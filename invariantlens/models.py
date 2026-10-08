"""The learned world models: a message-passing network (M1, and M2 trained on rotated data) and an EGNN-style one
(M3) that turns with the scene by construction."""

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


def dot(u, w):
    return (u * w).sum(-1, True)


class Net(nn.Module):
    """Message passing over all pairs of discs. It predicts how far each disc moves, and how much its velocity
    changes, over one decision step."""

    def __init__(self, equivariant, scale, width=64, layers=3):
        super().__init__()
        self.equivariant = equivariant
        # input and output scales: x, v, a, dx, dv
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.embed = nn.Linear(4 if equivariant else 7, width)
        self.edges = nn.ModuleList(mlp(2 * width + 3 * equivariant, width) for _ in range(layers))
        self.nodes = nn.ModuleList(mlp(2 * width, width) for _ in range(layers))
        self.out = nn.Linear(width, 4)
        self.pair = nn.Linear(width, 2) if equivariant else None

    def forward(self, s, a):
        sx, sv, sa, sdx, sdv = self.scale
        x = s[:, 0] / sx
        v = s[:, 1] / sv
        agent = (torch.arange(N, device=s.device) == 0).to(s.dtype)[:, None].expand(len(s), N, 1)
        push = agent * a[:, None] / sa
        rel_x = x[:, :, None] - x[:, None]
        rel_v = v[:, :, None] - v[:, None]

        if self.equivariant:
            # only lengths and dot products, which stay the same when the scene turns
            h = torch.cat([dot(v, v), agent, dot(push, push), dot(v, push)], -1)
            pair = [dot(rel_x, rel_x), dot(rel_x, rel_v), dot(rel_v, rel_v)]
        else:
            h = torch.cat([x, v, push, agent], -1)
            pair = []

        h = self.embed(h)
        off_diagonal = 1 - torch.eye(N, dtype=s.dtype, device=s.device)[..., None]
        for edge, node in zip(self.edges, self.nodes):
            m = self.messages(edge, h, pair) * off_diagonal
            h = h + node(torch.cat([h, m.sum(2)], -1))

        if self.equivariant:
            # the outputs are weighted sums of x_i - x_j, v_i and the push, so they turn with the scene
            w = self.pair(m) * off_diagonal
            c = self.out(h)
            dx = (w[..., 0, None] * rel_x).sum(2) + c[..., 0, None] * v + c[..., 1, None] * push
            dv = (w[..., 1, None] * rel_x).sum(2) + c[..., 2, None] * v + c[..., 3, None] * push
        else:
            dx, dv = self.out(h).split(2, -1)
        return torch.stack([dx * sdx, dv * sdv], 1)

    def messages(self, edge, h, pair):
        """The edge network on every ordered pair of discs."""
        if self.training:
            h_i = h[:, :, None].expand(-1, -1, N, -1)
            h_j = h[:, None].expand(-1, N, -1, -1)
            return edge(torch.cat([h_i, h_j, *pair], -1))

        # the same thing, about twice as fast: the first layer is applied to each disc once, not to each pair
        w, width = edge[0].weight, h.shape[-1]
        first = (h @ w[:, :width].T)[:, :, None] + (h @ w[:, width : 2 * width].T)[:, None] + edge[0].bias
        if pair:
            first = first + torch.cat(pair, -1) @ w[:, 2 * width :].T
        return edge[1:](first)


def transitions(n, seed):
    """(s, a, s') from n rollouts of HORIZON steps under random pushes."""
    rng = np.random.default_rng(seed)
    s = np.stack([initial_state(rng) for _ in range(n)])
    a = uniform_disc(rng, A_MAX, (n, HORIZON))
    states = rollout(step, s, a[:, None])[:, 0]
    before = states[:, :-1].reshape(-1, 2, N, 2)
    after = states[:, 1:].reshape(-1, 2, N, 2)
    return before, a.reshape(-1, 2), after


def train(kind, seed, data, validation, epochs=EPOCHS):
    """Fit one model to one-step changes in float32 and keep the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    s, a, s1 = (torch.from_numpy(x).float() for x in data)
    vs, va, vs1 = (torch.from_numpy(x).float() for x in validation)
    delta = s1 - s
    scale = torch.stack([s[:, 0].std(), s[:, 1].std(), a.std(), delta[:, 0].std(), delta[:, 1].std()])
    net = Net(kind == "M3", scale)
    unit = net.scale[3:, None, None]

    def loss(s, a, target):
        return ((net(s, a) - target) / unit).square().mean()

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, kept = float("inf"), None
    for _ in range(epochs):
        for i in torch.randperm(len(s)).split(BATCH):
            xs, xa, xt = s[i], a[i], delta[i]
            if kind == "M2":
                # turn each sample by its own random angle
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
    """The network as a step function on numpy arrays, like physics.step, evaluated in float64."""
    net = copy.deepcopy(net).double().eval().to(device)
    chunk = 4096 if torch.device(device).type == "cpu" else 16384

    def model(s, a):
        flat_s = torch.from_numpy(np.ascontiguousarray(s, dtype=float).reshape(-1, 2, N, 2)).to(device)
        flat_a = torch.from_numpy(np.ascontiguousarray(a, dtype=float).reshape(-1, 2)).to(device)
        deltas = []
        with torch.no_grad():
            for i in range(0, len(flat_s), chunk):
                deltas.append(net(flat_s[i : i + chunk], flat_a[i : i + chunk]))
        return s + torch.cat(deltas).cpu().numpy().reshape(s.shape)

    return model
