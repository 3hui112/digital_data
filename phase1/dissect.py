"""Phase 1 Task 1.5 with starter checkpoints (runs/<model>/seed<S>/seed_<S>/best.pt).
  (a) shape trace through starter MambaBlock
  (b) LTI: block.ssm == causal_conv(direct) == causal_conv(fft, padded); unpadded FFT = circular (wrong)
  (c) finite-length kernel frequency responses (truncation of the recurrent impulse response)
  (d) selective model: Δ per layer and A-dependent quantities (exp(ΔA), half-life ln2/(-ΔA)) on VALIDATION windows
"""
from common import FIG, HAR_CLASSES, RES, RUNS

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from evaluate import evaluation_data, load_checkpoint
from mamba_block import MambaBlock, causal_conv
from common import DATA

OUT = {}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(model, seed):
    return RUNS / model / f"seed{seed}" / f"seed_{seed}" / "best.pt"


def shape_trace():
    blk = MambaBlock(d_model=32, d_state=16)
    x = torch.randn(1, 128, 32); rows = []
    xz = blk.in_proj(x); rows.append(("in_proj: expand to 2·E·D", list(xz.shape)))
    u, z = xz.chunk(2, -1); rows.append(("chunk → main u / gate z", list(u.shape)))
    u = blk.conv1d(u.transpose(1, 2))[:, :, :128].transpose(1, 2); rows.append(("depthwise causal conv1d k=4 (+crop)", list(u.shape)))
    u = F.silu(u); rows.append(("SiLU", list(u.shape)))
    xd = blk.x_proj(u); rows.append(("x_proj → [Δ_raw | B | C]", list(xd.shape)))
    r, B, C = xd.split([blk.dt_rank, blk.d_state, blk.d_state], -1)
    rows += [("  Δ_raw (low rank r)", list(r.shape)), ("  B_t", list(B.shape)), ("  C_t", list(C.shape))]
    dl = F.softplus(blk.dt_proj(r)); rows.append(("dt_proj + softplus → Δ_t", list(dl.shape)))
    rows.append(("A = -exp(A_log)", list(blk.A.shape)))
    rows.append(("state h_t", [1, blk.d_inner, blk.d_state]))
    y = blk.ssm(u); rows.append(("ssm: y_t = Σ_n C_t h_t + D u_t", list(y.shape)))
    y = y * F.silu(z); rows.append(("gate ⊙ SiLU(z)", list(y.shape)))
    rows.append(("out_proj → D", list(blk.out_proj(y).shape)))
    OUT["shape_trace"] = rows


def lti_checks(model):
    blk = model.layers[0].mixer
    torch.manual_seed(0)
    u = torch.randn(8, 128, blk.d_inner, device=dev)
    y_rec = blk.ssm(u)
    K = blk.lti_kernel(128)                                       # includes D at lag 0
    y_dir, y_fft = causal_conv(u, K, "direct"), causal_conv(u, K, "fft")
    y_circ = torch.fft.irfft(torch.fft.rfft(u, n=128, dim=1) * torch.fft.rfft(K, n=128, dim=0)[None], n=128, dim=1)
    rel = lambda a: float((a - y_rec).norm() / y_rec.norm())
    OUT["lti_equivalence"] = dict(direct=rel(y_dir), fft_padded=rel(y_fft), fft_circular_no_pad=rel(y_circ))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].semilogy((y_circ - y_rec).abs().mean((0, 2)).cpu() + 1e-12, label="FFT, no padding (circular)")
    axes[0].semilogy((y_fft - y_rec).abs().mean((0, 2)).cpu() + 1e-12, label="FFT, zero-padded (linear)")
    axes[0].set_title("|conv − recurrence| per t (layer 1, trained LTI)"); axes[0].set_xlabel("t"); axes[0].legend(fontsize=8)
    Kl = blk.lti_kernel(4096, include_skip=False).cpu().numpy()
    amax = torch.exp(F.softplus(blk.dt_bias)[:, None] * blk.A).max(1).values.cpu().numpy()
    chans = np.argsort(amax)[np.linspace(0, len(amax) - 1, 4).astype(int)]
    tail = {}
    for i, c in enumerate(chans):
        col = plt.cm.plasma(i / 4)
        axes[1].plot(Kl[:128, c] / np.abs(Kl[:, c]).max(), c=col, label=f"ch{c} max a={amax[c]:.3f}")
        for L, ls in [(16, ":"), (128, "-"), (4096, "--")]:
            H = np.fft.rfft(Kl[:L, c], n=8192); f = np.fft.rfftfreq(8192, 1 / 50)
            axes[2].plot(f, 20 * np.log10(np.abs(H) + 1e-12), c=col, ls=ls, lw=1)
        tail[int(c)] = float((Kl[128:, c] ** 2).sum() / (Kl[:, c] ** 2).sum())
    axes[1].set_title("normalised SSM kernels (no skip), 128 lags"); axes[1].set_xlabel("lag"); axes[1].legend(fontsize=7)
    axes[2].set_title("|FFT| of length-L kernel, dB  (':' 16, '-' 128, '--' 4096)"); axes[2].set_xlabel("Hz (fs=50)"); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "t15_lti_equivalence_kernels.png", dpi=130); plt.close(fig)
    OUT["lti_kernel_energy_beyond_128"] = tail


def delta_stats(model, x, y, tag):
    model.set_record(True)
    per_layer = [[] for _ in model.layers]
    for i in range(0, len(x), 256):
        model(torch.from_numpy(x[i:i + 256]).to(dev))
        for j, rec in enumerate(model.records):
            per_layer[j].append(rec["delta"])                     # (b, l, h), CPU
    model.set_record(False)
    fig, axes = plt.subplots(len(per_layer), 4, figsize=(19, 3.6 * len(per_layer))); out = {}
    for j, dl in enumerate(per_layer):
        D = torch.cat(dl); A = model.layers[j].mixer.A.detach().cpu()
        Dm = D.mean((0, 1)); negdA = Dm[:, None] * (-A); hl = np.log(2) / negdA
        sub = D.flatten()[torch.randperm(D.numel())[:200000]].numpy()
        axes[j, 0].hist(np.log10(sub), 100); axes[j, 0].set_title(f"layer {j+1}: log10 Δ_t (all t, channels)")
        axes[j, 1].hist(torch.exp(-negdA).flatten().numpy(), 100, color="tab:green"); axes[j, 1].set_title("a = exp(Δ̄·A_n)")
        axes[j, 2].hist(np.log10(hl.flatten().numpy()), 100, color="tab:orange"); axes[j, 2].set_title("log10 half-life [samples]")
        axes[j, 3].scatter(np.repeat(Dm.numpy(), A.shape[1]), negdA.flatten().numpy(), s=2, alpha=.3)
        axes[j, 3].set_xscale("log"); axes[j, 3].set_yscale("log")
        axes[j, 3].set_xlabel("channel mean Δ"); axes[j, 3].set_ylabel("-Δ̄A_n"); axes[j, 3].set_title("same Δ, different decay per state")
        out[f"layer{j+1}"] = dict(delta_median=float(np.median(sub)), delta_p05=float(np.quantile(sub, .05)),
                                  delta_p95=float(np.quantile(sub, .95)), half_life_median=float(hl.median()),
                                  frac_half_life_gt_128=float((hl > 128).float().mean()),
                                  frac_half_life_lt_1=float((hl < 1).float().mean()),
                                  delta_time_cv=float((D.std(1) / D.mean(1)).mean()),
                                  delta_mean_by_class={HAR_CLASSES[k]: float(D[torch.from_numpy(y == k)].mean()) for k in range(6)})
    fig.suptitle(f"Task 1.5  Δ and A-dependent decay ({tag}, validation)")
    fig.tight_layout(); fig.savefig(FIG / f"t15_delta_{tag}.png", dpi=130); plt.close(fig)
    OUT[f"delta_{tag}"] = out


@torch.no_grad()
def main():
    shape_trace()
    for name, seed in [("lti", 0), ("mamba", 0)]:
        p = ckpt_path(name, seed)
        if not p.exists():
            print("missing", p); continue
        model, ck = load_checkpoint(p, dev)
        x, y = evaluation_data(ck, DATA)
        if name == "lti":
            lti_checks(model)
        delta_stats(model, x, y, f"{name}_s{seed}")
    (RES / "phase1_dissect.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps({k: v for k, v in OUT.items() if k != "shape_trace"}, indent=1))
    for s, sh in OUT["shape_trace"]:
        print(f"{s:<42} {sh}")


if __name__ == "__main__":
    main()
