"""Phase 1 Tasks 1.1, 1.2, 1.2b, 1.3 (figures/numbers; no deep learning). Uses starter signal_utils only.
Spectra: starter power_spectrum (one-sided PSD, periodic Hann, constant detrend), averaged as POWER over windows,
TRAINING subjects only. The EMA classifier itself is starter train_ema.py (runs/ema); here we add plots and a raw-feature control.
"""
from common import FIG, FS, GROUPS, HAR_CHANNELS, HAR_CLASSES, RES, dev_split

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signal_utils import alias_frequency, downsample, ema_features, power_spectrum, upsample

OUT = {}
COLORS = plt.cm.tab10(np.arange(6))
ALPHAS = [0.1, 0.5, 0.9, 0.99]          # starter train_ema.py default; used for the reported baseline


def frac(f, P, lo, hi):
    return float(P[..., (f >= lo) & (f < hi)].sum() / P.sum())


def task_1_1(d):
    X, y = d["X_train"], d["y_train"]
    f, P = power_spectrum(X, FS)                                   # (N, F, 9)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ci, ch in enumerate(HAR_CHANNELS):
        ax = axes[ci // 3, ci % 3]
        for k, c in enumerate(HAR_CLASSES):
            ax.semilogy(f, P[y == k, :, ci].mean(0), color=COLORS[k], label=c, lw=1.2)
        ax.axvline(20, ls="--", c="gray", lw=.8); ax.set_title(ch); ax.grid(alpha=.3)
    for ax in axes[-1]: ax.set_xlabel("Hz")
    for ax in axes[:, 0]: ax.set_ylabel("mean PSD")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Task 1.1  activity-wise PSD per channel (Hann, constant detrend, power-averaged; dashed = dataset 20 Hz LPF)")
    fig.tight_layout(); fig.savefig(FIG / "t11_spectra_channels.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); stats = {}
    for gi, (g, idx) in enumerate(GROUPS.items()):
        for k, c in enumerate(HAR_CLASSES):
            Pg = P[y == k][:, :, idx].mean((0, 2))
            axes[gi].semilogy(f, Pg, color=COLORS[k], label=c)
            stats[f"{g}/{c}"] = dict(peak_hz=float(f[1:][np.argmax(Pg[1:])]), frac_0_3=frac(f, Pg, 0, 3),
                                     frac_3_10=frac(f, Pg, 3, 10), frac_10_20=frac(f, Pg, 10, 20),
                                     frac_20_25=frac(f, Pg, 20, 26), total=float(Pg.sum()))
        axes[gi].axvline(20, ls="--", c="gray", lw=.8); axes[gi].set_title(g); axes[gi].set_xlabel("Hz"); axes[gi].grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "t11_spectra_groups.png", dpi=130); plt.close(fig)
    Pb = P[:, :, :6].mean((0, 2))
    OUT["1.1"] = dict(bin_hz=float(f[1]), group_class=stats,
                      body6_frac_above={fn: frac(f, Pb, fn, 26) for fn in [12.5, 6.25, 3.125]})


RATES = {2: 25.0, 4: 12.5, 8: 6.25}


def task_1_2(d):
    X, y = d["X_train"][:, :, :6], d["y_train"]                    # body acc + gyro (gravity removed)
    f0, P0 = power_spectrum(X, FS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); out = {}
    for j, (M, fs) in enumerate(RATES.items()):
        for row, cls, title in [(0, [0, 1, 2], "dynamic"), (1, [3, 4, 5], "static")]:
            sel = np.isin(y, cls); ax = axes[row, j]
            ax.semilogy(f0, P0[sel].mean((0, 2)), c="k", lw=1, label="50 Hz")
            for meth, c in [("naive", "tab:red"), ("aa", "tab:blue")]:
                f, P = power_spectrum(downsample(X[sel], M, meth), fs)
                ax.semilogy(f, P.mean((0, 2)), c=c, lw=1.4, label=f"{meth} {fs} Hz")
            ax.axvline(fs / 2, ls="--", c="gray"); ax.set_xlim(0, 25); ax.grid(alpha=.3)
            ax.set_title(f"{fs} Hz (Nyquist {fs/2}) — {title}", fontsize=9)
        f, Pn = power_spectrum(downsample(X, M, "naive"), fs); _, Pa = power_spectrum(downsample(X, M, "aa"), fs)
        Pn, Pa = Pn.mean((0, 2)), Pa.mean((0, 2))
        out[f"{fs}Hz"] = dict(rel_l1_naive_vs_aa=float(np.abs(Pn - Pa).sum() / Pa.sum()),
                              frac_orig_power_above_new_nyquist=frac(f0, P0.mean((0, 2)), fs / 2, 26))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Task 1.2  naive x[::M] vs anti-aliased (Kaiser FIR, zero-phase polyphase) — body_acc+gyro")
    fig.tight_layout(); fig.savefig(FIG / "t12_downsample_spectra.png", dpi=130); plt.close(fig)
    OUT["1.2"] = out


def task_1_2b(d):
    X = d["X_train"][:, :, 0:1].astype(np.float64)                 # body_acc_x
    t = np.arange(X.shape[1]) / FS
    ph = np.random.default_rng(0).uniform(0, 2 * np.pi, (len(X), 1, 1))
    Xt = X + 2 * X.std() * np.sin(2 * np.pi * 18 * t[None, :, None] + ph)
    f0, P0 = power_spectrum(Xt, FS); P0m = P0.mean((0, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); meas = {}
    for j, (M, fs) in enumerate(RATES.items()):
        pa = alias_frequency(18, fs); m = meas[f"{fs}Hz"] = dict(predicted_alias_hz=pa)
        ax = axes[0, j]; ax.semilogy(f0, P0m, c="k", lw=1, label="50 Hz + 18 Hz tone")
        for meth, c in [("naive", "tab:red"), ("aa", "tab:blue")]:
            f, P = power_spectrum(downsample(Xt, M, meth), fs); Pm = P.mean((0, 2))
            k = np.argmin(np.abs(f - pa)); lo, hi = max(k - 2, 1), k + 3
            m[f"{meth}_peak_near_pred_hz"] = float(f[lo:hi][np.argmax(Pm[lo:hi])])
            m[f"{meth}_psd_at_alias"] = float(Pm[k])
            ax.semilogy(f, Pm, c=c, lw=1.4, label=meth)
        ax.axvline(pa, c="tab:red", ls=":", label=f"predicted alias {pa} Hz"); ax.axvline(fs / 2, c="gray", ls="--")
        ax.set_xlim(0, 25); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_title(f"to {fs} Hz")
        ax = axes[1, j]
        for meth, c in [("naive", "tab:red"), ("aa", "tab:blue")]:
            f, P = power_spectrum(upsample(downsample(Xt, M, meth), M, "poly"), FS); Pm = P.mean((0, 2))
            k18, ka = np.argmin(np.abs(f - 18)), np.argmin(np.abs(f - pa))
            m[f"up_{meth}_18hz_ratio"] = float(Pm[k18] / P0m[k18]); m[f"up_{meth}_alias_ratio"] = float(Pm[ka] / P0m[ka])
            ax.semilogy(f, Pm, c=c, lw=1.4, label=f"{meth} → upsample poly ×{M}")
        ax.semilogy(f0, P0m, c="k", lw=1, label="original"); ax.axvline(18, c="k", ls=":"); ax.axvline(pa, c="tab:red", ls=":")
        ax.set_xlim(0, 25); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_xlabel("Hz"); ax.set_title(f"{fs} Hz → back to 50 Hz")
        # long pure tone: exact check
        tt = np.arange(2000) / FS; fp, Pp = power_spectrum(np.sin(2 * np.pi * 18 * tt)[::M, None], fs)
        m["pure_tone_peak_hz"] = float(fp[np.argmax(Pp[:, 0])])
    fig.suptitle("Task 1.2b  body_acc_x + 18 Hz tone: aliasing, and interpolation cannot recover it")
    fig.tight_layout(); fig.savefig(FIG / "t12b_tone_alias.png", dpi=130); plt.close(fig)
    OUT["1.2b"] = meas


def task_1_3(d):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4)); k = np.arange(60); info = {}
    for i, a in enumerate(ALPHAS):
        c = plt.cm.viridis(i / len(ALPHAS))
        axes[0].plot(k, (1 - a) * a ** k, c=c, marker=".", ms=3, label=f"a={a}")
        w, H = signal.freqz([1 - a], [1, -a], worN=512, fs=FS)
        axes[1].plot(w, 20 * np.log10(np.abs(H)), c=c, label=f"a={a}")
        delta = -np.log(a); cosw = 1 - (1 - a) ** 2 / (2 * a)
        info[str(a)] = dict(delta_if_A_minus1=float(delta), half_life_samples=float(np.log(2) / delta),
                            time_const_s=float(1 / delta / FS),
                            cutoff_3db_hz=float(FS / (2 * np.pi) * np.arccos(cosw)) if cosw >= -1 else None)
    axes[0].set_title("EMA impulse response (1-a)a^k"); axes[0].set_xlabel("lag"); axes[0].legend(fontsize=7)
    axes[1].set_title("|H(f)| dB (fs=50 Hz)"); axes[1].set_ylim(-45, 2); axes[1].set_xlabel("Hz"); axes[1].grid(alpha=.3)
    dd = np.logspace(-3, 1, 200)
    for A in [-1, -4, -16]: axes[2].semilogx(dd, np.exp(dd * A), label=f"A={A}")
    for a in ALPHAS: axes[2].axhline(a, c="gray", lw=.5)
    axes[2].set_title("a = exp(ΔA): decay set by the product ΔA"); axes[2].set_xlabel("Δ"); axes[2].legend()
    fig.tight_layout(); fig.savefig(FIG / "t13_ema_responses.png", dpi=130); plt.close(fig)

    # control: same pipeline on raw-signal statistics only (no EMA bank). a=0 => identity filter.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    clf.fit(ema_features(d["X_train"], [0.0]), d["y_train"]); p = clf.predict(ema_features(d["X_val"], [0.0]))
    OUT["1.3"] = dict(alphas=ALPHAS, ema_info=info, raw_only_control=dict(
        acc=float(accuracy_score(d["y_val"], p)), macro_f1=float(f1_score(d["y_val"], p, average="macro")),
        cm=confusion_matrix(d["y_val"], p).tolist()))


if __name__ == "__main__":
    d = dev_split()
    task_1_1(d); task_1_2(d); task_1_2b(d); task_1_3(d)
    (RES / "phase1_signal.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps({k: OUT[k] for k in ["1.2", "1.2b"]}, indent=1))
    print("1.3 raw-only control", {k: v for k, v in OUT["1.3"]["raw_only_control"].items() if k != "cm"})
