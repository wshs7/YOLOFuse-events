# ultralytics/nn/modules/gates.py
import torch
import torch.nn as nn
import torch.nn.functional as F

METRIC_COLUMNS_16 = [
    "frame_muY", "frame_stdY", "frame_DR_p99_p1", "frame_p_sat", "frame_entropy",
    "frame_var_lap", "frame_grad_mean", "frame_colorfulness",
    "event_nz_density", "event_int_density", "event_onoff_ratio", "event_gini",
    "event_grid_cv", "event_entropy", "event_var_lap", "event_polar_bias",
]

METRIC_COLUMNS_8 = [
    "frame_muY",
    "frame_stdY",
    "frame_var_lap",
    "frame_grad_mean",
    "event_grid_cv",
    "event_int_density",
    "event_polar_bias",
    "event_var_lap",
]


def _lap_var(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) in [0,1]; return (B,) Laplacian variance."""
    lap = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    y = F.conv2d(x, lap, padding=1)
    return y.var(dim=[2, 3]).squeeze(1)


def _sobel_mean(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W); return (B,) of mean |grad|."""
    kx = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1., -2., -1.],
         [0.,  0.,  0.],
         [1.,  2.,  1.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy)
    return g.mean(dim=[1, 2, 3])


def _entropy_u8(x01, bins=256):
    """
    x01: (B,1,H,W) in [0,1]
    """
    prev_det = torch.are_deterministic_algorithms_enabled()

    if prev_det:
        torch.use_deterministic_algorithms(False, warn_only=True)

    try:
        B = x01.size(0)
        flat = x01.clamp(0, 1).reshape(B, -1)
        Hs = []
        for b in range(B):
            h = torch.histc(flat[b], bins=bins, min=0.0, max=1.0)
            p = h / (h.sum() + 1e-12)
            p = p[p > 0]
            Hs.append((-(p * torch.log2(p))).sum())
        Hs = torch.stack(Hs, dim=0)
    finally:
        if prev_det:
            torch.use_deterministic_algorithms(True, warn_only=True)

    return Hs


def _quantile_range(x01: torch.Tensor, q1=0.01, q2=0.99) -> torch.Tensor:
    """x01: (B,1,H,W); return (B,) q2 - q1."""
    flat = x01.reshape(x01.size(0), -1)
    lo = torch.quantile(flat, q1, dim=1)
    hi = torch.quantile(flat, q2, dim=1)
    return hi - lo


def _colorfulness(R: torch.Tensor, G: torch.Tensor, Bc: torch.Tensor) -> torch.Tensor:
    """R,G,Bc: (B,1,H,W); Hasler–Süsstrunk colorfulness."""
    Rm = R.squeeze(1)
    Gm = G.squeeze(1)
    Bm = Bc.squeeze(1)
    rg = Rm - Gm
    yb = (Rm + Gm) / 2 - Bm
    return torch.sqrt(rg.std(dim=[1, 2]) ** 2 + yb.std(dim=[1, 2]) ** 2) + \
        0.3 * torch.sqrt(rg.mean(dim=[1, 2]) ** 2 + yb.mean(dim=[1, 2]) ** 2)


def _gini(flat_nonneg: torch.Tensor) -> torch.Tensor:
    """flat_nonneg: (B,N) ≥0; return (B,) Gini."""
    B, N = flat_nonneg.shape
    x_sorted, _ = torch.sort(flat_nonneg, dim=1)
    s = x_sorted.sum(dim=1) + 1e-12
    idx = torch.arange(1, N + 1, device=flat_nonneg.device,
                       dtype=flat_nonneg.dtype).view(1, -1)
    w = (N + 1 - idx)
    lor = (w * x_sorted).sum(dim=1) / s
    return (N + 1 - 2 * lor) / N


def _grid_cv(E01: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """E01: (B,1,H,W) in [0,1]; CV over grid cells."""
    pooled = F.adaptive_avg_pool2d(E01, (grid, grid))
    cells = pooled.reshape(E01.size(0), -1)
    mean = cells.mean(dim=1)
    std = cells.std(dim=1, unbiased=False)
    return std / (mean + 1e-12)


class Plugin_module(nn.Module):

    def __init__(
        self,
        metric_dim: int = 8,
        hidden: int = 64,
        conv_channels: int = 16
    ):
        super().__init__()
        self.metric_dim = metric_dim

        self.conv_block = nn.Sequential(
            nn.Conv2d(5, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(conv_channels, 5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(5)
        )

        self.fe = nn.Sequential(
            nn.Linear(metric_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.head_group = nn.Linear(hidden, 2)
        self.head_film = nn.Linear(hidden, 10)
        self.res_scale = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.head_group.weight)
        nn.init.zeros_(self.head_group.bias)
        nn.init.zeros_(self.head_film.weight)
        nn.init.zeros_(self.head_film.bias)

    @torch.no_grad()
    def _compute_metrics_batch_full(self, img5: torch.Tensor) -> torch.Tensor:

        B, C, H, W = img5.shape

        P = img5[:, 0:1]
        N = img5[:, 1:2]
        R = img5[:, 2:3]
        G = img5[:, 3:4]
        Bc = img5[:, 4:5]

        scale = 255.0
        Pn, Nn = P / scale, N / scale
        E = (Pn + Nn).clamp(min=0.)

        Y = (0.299 * R + 0.587 * G + 0.114 * Bc).clamp(0, 1)
        muY = Y.mean(dim=[1, 2, 3])
        stdY = Y.std(dim=[1, 2, 3], unbiased=False)
        dr = _quantile_range(Y, 0.01, 0.99)
        psat = ((Y < 0.02).float().mean(dim=[1, 2, 3]) +
                (Y > 0.98).float().mean(dim=[1, 2, 3]))
        entY = _entropy_u8(Y)
        varlp = _lap_var(Y)
        gmean = _sobel_mean(Y)
        cf = _colorfulness(R, G, Bc)

        nz_density = (E > 0).float().mean(dim=[1, 2, 3])
        int_density = E.mean(dim=[1, 2, 3])
        onoff = (Pn.sum(dim=[1, 2, 3]) + 1e-12) / \
            (Nn.sum(dim=[1, 2, 3]) + 1e-12)
        flatE = E.reshape(B, -1)
        gg = _gini(flatE)
        cvgrid = _grid_cv(E)
        entE = _entropy_u8(E)
        varlpE = _lap_var(E)
        pbias = ((Pn - Nn) / (Pn + Nn + 1e-12)).mean(dim=[1, 2, 3])

        z16 = torch.stack([
            muY, stdY, dr, psat, entY, varlp, gmean, cf,
            nz_density, int_density, onoff, gg, cvgrid, entE, varlpE, pbias
        ], dim=1)

        return z16

    @torch.no_grad()
    def _compute_metrics_batch(self, img5: torch.Tensor) -> torch.Tensor:

        z16 = self._compute_metrics_batch_full(img5)

        # frame_muY:         0
        # frame_stdY:        1
        # frame_var_lap:     5
        # frame_grad_mean:   6
        # event_grid_cv:     12
        # event_int_density: 9
        # event_polar_bias:  15
        # event_var_lap:     14
        idx = [0, 1, 5, 6, 12, 9, 15, 14]
        z8 = z16[:, idx]
        return z8

    def forward(self, x):

        if not isinstance(x, (list, tuple)):
            return x

        if len(x) < 2:
            return x

        event = x[0]
        frame = x[1]

        B, _, H, W = event.shape

        img5 = torch.cat([event, frame], dim=1)

        fx = self.conv_block(img5)

        z = self._compute_metrics_batch(img5).to(
            img5.dtype).to(img5.device)

        h = self.fe(z)

        group_logits = self.head_group(h)
        g = torch.sigmoid(group_logits)
        g = g / (g.sum(dim=1, keepdim=True) + 1e-12)
        alpha_event = g[:, 0:1].view(B, 1, 1, 1)
        alpha_frame = g[:, 1:2].view(B, 1, 1, 1)

        film_params = self.head_film(h)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=1)

        gamma = 1.0 + 0.1 * gamma
        beta = 0.1 * beta

        gamma = gamma.view(B, 5, 1, 1)
        beta = beta.view(B, 5, 1, 1)

        fx_film = gamma * fx + beta

        event_part = fx_film[:, 0:2] * alpha_event
        frame_part = fx_film[:, 2:5] * alpha_frame
        fused = torch.cat([event_part, frame_part], dim=1)

        out5 = img5 + self.res_scale * fused

        out_event = out5[:, 0:2]
        out_frame = out5[:, 2:5]

        y = list(x)
        y[0] = out_event
        y[1] = out_frame
        return y