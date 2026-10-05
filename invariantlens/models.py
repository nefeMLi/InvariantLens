"""Learned world models: a message-passing network (M1, and M2 trained on rotated data) and an EGNN-style one (M3)."""

import copy

import numpy as np
import torch
from torch import nn

from invariantlens.decisions import HORIZON, rollout
from invariantlens.physics import A_MAX, N, initial_state, step, uniform_disc

KINDS = ("M1", "M2", "M3")
EPOCHS, BATCH, LR = 200, 256, 1e-3


def mlp(n_in, width):
    return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())


class Net(nn.Module):
    """Message passing over all disc pairs, predicting (dx, dv) for every disc over one decision step.
    Plain: absolute coordinates go in and a linear head reads (dx, dv) off each disc.
    Equivariant: only invariants go in, and (dx, dv) are relative positions, velocities and the push weighted by
    learned scalars. Pair weights depend on the ordered pair, so momentum is not conserved by construction."""

    def __init__(self, equivariant, scale, width=64, layers=3):
        super().__init__()
        self.equivariant = equivariant
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))  # x, v, a, dx, dv
        self.embed = nn.Linear(4 if equivariant else 7, width)
        self.edges = nn.ModuleList(mlp(2 * width + 3 * equivariant, width) for _ in range(layers))
        self.nodes = nn.ModuleList(mlp(2 * width, width) for _ in range(layers))
        self.out = nn.Linear(width, 4)  # (dx, dv), or the weights of v_i and the push in each
        self.pair = nn.Linear(width, 2) if equivariant else None  # the weights of x_i - x_j in dx and dv

    def forward(self, s, a):
        sx, sv, sa, sdx, sdv = self.scale
        x, v = s[:, 0] / sx, s[:, 1] / sv
        agent = (torch.arange(N, device=s.device) == 0).to(s.dtype)[:, None].expand(len(s), N, 1)
        push = agent * a[:, None] / sa
        d, dv = x[:, :, None] - x[:, None], v[:, :, None] - v[:, None]
        if self.equivariant:
            h = torch.cat([(v * v).sum(-1, True), agent, (push * push).sum(-1, True), (v * push).sum(-1, True)], -1)
            pair = [(d * d).sum(-1, True), (d * dv).sum(-1, True), (dv * dv).sum(-1, True)]
        else:
            h, pair = torch.cat([x, v, push, agent], -1), []
        h, off = self.embed(h), 1 - torch.eye(N, dtype=s.dtype, device=s.device)[..., None]
        for edge, node in zip(self.edges, self.nodes):
            if self.training:
                m = edge(torch.cat([h[:, :, None].expand(-1, -1, N, -1), h[:, None].expand(-1, N, -1, -1), *pair], -1))
            else:  # the same first layer, applied to each disc once instead of to each pair: about twice as fast
                w, width = edge[0].weight, h.shape[-1]
                first = (h @ w[:, :width].T)[:, :, None] + (h @ w[:, width : 2 * width].T)[:, None] + edge[0].bias
                m = edge[1:](first + (torch.cat(pair, -1) @ w[:, 2 * width :].T if pair else 0))
            m = m * off
            h = h + node(torch.cat([h, m.sum(2)], -1))
        if self.equivariant:
            w, c = self.pair(m) * off, self.out(h)
            out = [
                (w[..., k, None] * d).sum(2) + c[..., 2 * k, None] * v + c[..., 2 * k + 1, None] * push for k in (0, 1)
            ]
        else:
            out = self.out(h).split(2, -1)
        return torch.stack([out[0] * sdx, out[1] * sdv], 1)


def transitions(n, seed):
    """(s, a, s') from n trajectories of HORIZON decision steps with i.i.d. uniform actions."""
    rng = np.random.default_rng(seed)
    s, a = np.stack([initial_state(rng) for _ in range(n)]), uniform_disc(rng, A_MAX, (n, HORIZON))
    states = rollout(step, s, a[:, None])[:, 0]
    return states[:, :-1].reshape(-1, 2, N, 2), a.reshape(-1, 2), states[:, 1:].reshape(-1, 2, N, 2)


def train(kind, seed, data, validation, epochs=EPOCHS):
    """One model trained on one-step MSE in scaled units, in float32; the epoch with the best validation MSE is kept."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    s, a, s1 = (torch.from_numpy(x).float() for x in data)
    vs, va, vs1 = (torch.from_numpy(x).float() for x in validation)
    delta = s1 - s
    net = Net(kind == "M3", torch.stack([s[:, 0].std(), s[:, 1].std(), a.std(), delta[:, 0].std(), delta[:, 1].std()]))
    unit = net.scale[3:, None, None]

    def loss(s, a, target):
        return ((net(s, a) - target) / unit).square().mean()

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, kept = float("inf"), None
    for _ in range(epochs):
        for i in torch.randperm(len(s)).split(BATCH):
            xs, xa, xt = s[i], a[i], delta[i]
            if kind == "M2":  # a fresh continuous rotation per sample
                t = torch.rand(len(i)) * 2 * torch.pi
                R = torch.stack([torch.stack([t.cos(), -t.sin()], -1), torch.stack([t.sin(), t.cos()], -1)], -2)
                xs, xa, xt = (torch.einsum("bij,b...j->b...i", R, y) for y in (xs, xa, xt))
            opt.zero_grad()
            loss(xs, xa, xt).backward()
            opt.step()
        schedule.step()
        with torch.no_grad():
            current = loss(vs, va, vs1 - vs).item()
        if current < best:
            best, kept = current, copy.deepcopy(net.state_dict())
    net.load_state_dict(kept)
    return net


def world_model(net, device="cpu"):
    """The trained network as a float64 step map on numpy arrays, like physics.step; it runs on device."""
    net = copy.deepcopy(net).double().eval().to(device)
    chunk = 4096 if torch.device(device).type == "cpu" else 16384

    def model(s, a):
        flat_s = torch.from_numpy(np.ascontiguousarray(s, dtype=float).reshape(-1, 2, N, 2)).to(device)
        flat_a = torch.from_numpy(np.ascontiguousarray(a, dtype=float).reshape(-1, 2)).to(device)
        with torch.no_grad():
            delta = torch.cat([net(flat_s[i : i + chunk], flat_a[i : i + chunk]) for i in range(0, len(flat_s), chunk)])
        return s + delta.cpu().numpy().reshape(s.shape)

    return model
