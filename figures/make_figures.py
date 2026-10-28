#!/usr/bin/env python3
"""
make_figures.py — generate the figures for the report.

Produces (in --out-dir, default figures/out/):
    fig_mae_by_street.pdf  — MAE per street per target (win/tie/equity)
    fig_pareto.pdf         — speed vs accuracy (log x, linear y)
    fig_residuals.pdf      — distribution of equity residuals per street
    fig_calibration.pdf    — predicted vs actual equity, binned deciles

Data sources:
    data/test_dataset.csv      — 15,000 labeled held-out states
    data/approx_weights_s*.bin — per-street model weights

Usage:
    python figures/make_figures.py
    python figures/make_figures.py --out-dir Figures    # for the report folder
"""
import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_FEATURES = 62
STREETS = ["Preflop", "Flop", "Turn", "River"]

# ---- LaTeX-ish styling: serif font, Computer Modern math ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 100,
})


# ============================================================ model loading

def load_pkap(path):
    """Load a PKAP weights file into a list of (W, b) float64 arrays."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"PKAP":
            raise ValueError(f"{path}: not a PKAP file (got {magic!r})")
        (n_layers,) = struct.unpack("<i", f.read(4))
        layers = []
        for _ in range(n_layers):
            i, o = struct.unpack("<ii", f.read(8))
            W = np.frombuffer(f.read(i * o * 4), dtype="<f4").reshape(i, o)
            b = np.frombuffer(f.read(o * 4), dtype="<f4")
            # astype(np.float64) makes a writable copy (frombuffer views are read-only)
            layers.append((W.astype(np.float64), b.astype(np.float64)))
    return layers


def forward(layers, X):
    """Forward pass matching src/approx.cpp: ELU hidden, sigmoid output."""
    h = X
    for i, (W, b) in enumerate(layers):
        z = h @ W + b
        if i == len(layers) - 1:
            h = 1.0 / (1.0 + np.exp(-z))                          # sigmoid
        else:
            h = np.where(z > 0, z, np.expm1(np.minimum(z, 0)))    # ELU
    return h


def load_csv(path):
    """Read a CSV written by `poker gen-dataset`. Returns X, Y, street, se."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1,
                     usecols=range(2, 6 + N_FEATURES))
    street = arr[:, 0].astype(int)
    Y = arr[:, 1:3]                        # win, tie
    se = arr[:, 3]
    X = arr[:, 4:].astype(np.float64)      # 62 features
    return X, Y, street, se


def predict(weights_dir, X, street):
    """Per-street forward pass. Returns (N, 2) array of (P(win), P(tie))."""
    nets = [load_pkap(weights_dir / f"approx_weights_s{s}.bin") for s in range(4)]
    P = np.zeros((len(X), 2))
    for s in range(4):
        mask = street == s
        if mask.any():
            P[mask] = forward(nets[s], X[mask])
    return P


# ============================================================ figures

def fig_mae_by_street(P, Y, street, out):
    """Grouped bar chart: MAE win/tie/equity per street, plus overall line."""
    metrics = {
        "Win":    np.abs(P[:, 0] - Y[:, 0]),
        "Tie":    np.abs(P[:, 1] - Y[:, 1]),
        "Equity": np.abs((P[:, 0] + P[:, 1] / 2) - (Y[:, 0] + Y[:, 1] / 2)),
    }
    colors = {"Win": "#4C72B0", "Tie": "#DD8452", "Equity": "#55A868"}
    x = np.arange(len(STREETS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ymax = max(100.0 * m.mean() for m in metrics.values()) * 1.35
    for i, (name, err) in enumerate(metrics.items()):
        per_street = [100.0 * err[street == s].mean() for s in range(4)]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, per_street, width,
                      label=name, color=colors[name],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, per_street):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    overall = 100.0 * metrics["Equity"].mean()
    ax.axhline(overall, color="black", linestyle="--", linewidth=0.8,
               label=f"Overall equity ({overall:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(STREETS)
    ax.set_ylabel("MAE (%)")
    ax.set_ylim(0, ymax)
    ax.set_title("Network accuracy by street (15,000 held-out states)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pareto(out):
    """Speed vs accuracy for the three methods listed in Table II."""
    # Data taken from the README's measured-results table.
    points = [
        ("Per-street net",   5,  1.80, "#55A868", (10, -14)),
        ("MC, 1000 show.", 277,  1.10, "#4C72B0", (-12, 10)),
        ("MC, ~300 show.",  80,  2.03, "#DD8452", (10, 6)),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for label, us, mae, c, xy in points:
        ax.scatter([us], [mae], s=90, color=c, edgecolor="black",
                   linewidth=0.6, zorder=3)
        ax.annotate(label, (us, mae), textcoords="offset points",
                    xytext=xy, fontsize=8)

    # Equal-accuracy reference: network (1.80% at 5 µs) vs MC (2.03% at 80 µs)
    ax.plot([5, 80], [1.80, 2.03], color="gray", linestyle=":",
            linewidth=0.8, zorder=1)
    ax.text(20, 1.86, "≈16× at equal accuracy", fontsize=7,
            ha="center", va="top", style="italic", color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("microseconds per prediction (log scale)")
    ax.set_ylabel("Equity MAE (%)")
    ax.set_xlim(3, 500)
    ax.set_ylim(0.9, 2.3)
    ax.set_title("Speed vs accuracy tradeoff")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_residuals(P, Y, street, out):
    """2x2 grid of equity-residual histograms, one per street."""
    residual = 100.0 * ((P[:, 0] + P[:, 1] / 2) - (Y[:, 0] + Y[:, 1] / 2))

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2),
                             sharex=True, sharey=False)
    axes = axes.flatten()
    bins = np.linspace(-10, 10, 41)     # 0.5% bins
    for s in range(4):
        ax = axes[s]
        r = residual[street == s]
        mean, std = r.mean(), r.std()
        ax.hist(r, bins=bins, color="#4C72B0",
                edgecolor="white", linewidth=0.3, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(mean, color="red", linestyle="--", linewidth=0.8,
                   label=f"mean {mean:+.2f}")
        ax.set_title(f"{STREETS[s]}  (n={len(r):,}, σ={std:.2f}%)")
        ax.set_xlim(-10, 10)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    for ax in axes[2:]:
        ax.set_xlabel("predicted − actual equity (%)")
    for ax in axes[::2]:
        ax.set_ylabel("count")
    fig.suptitle("Equity residuals on the held-out test set")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_calibration(P, Y, out):
    """Calibration plot: predicted equity (deciles) vs mean observed equity."""
    pred = P[:, 0] + P[:, 1] / 2
    actual = Y[:, 0] + Y[:, 1] / 2

    # Equal-count bins so each point has roughly the same sample size
    order = np.argsort(pred)
    n_bins = 20
    edges = np.linspace(0, len(pred), n_bins + 1).astype(int)
    bin_pred, bin_obs, bin_n = [], [], []
    for k in range(n_bins):
        idx = order[edges[k]:edges[k + 1]]
        if len(idx) < 30:
            continue
        bin_pred.append(pred[idx].mean())
        bin_obs.append(actual[idx].mean())
        bin_n.append(len(idx))

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8,
            linestyle="--", label="ideal")
    ax.plot(bin_pred, bin_obs, marker="o", markersize=4,
            color="#55A868", linewidth=1.2, label="network")
    ax.set_xlabel("predicted equity")
    ax.set_ylabel("observed equity (test-set mean)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Calibration on the held-out test set")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/test_dataset.csv")
    ap.add_argument("--weights-dir", default="data")
    ap.add_argument("--out-dir", default="figures/out")
    args = ap.parse_args()

    data = Path(args.data)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data.exists():
        raise SystemExit(f"missing {data}")
    for s in range(4):
        wp = weights_dir / f"approx_weights_s{s}.bin"
        if not wp.exists():
            raise SystemExit(f"missing {wp}")

    print(f"loading {data}")
    X, Y, street, se = load_csv(data)
    print(f"  {len(X):,} rows, counts per street = {np.bincount(street).tolist()}")

    print(f"loading weights from {weights_dir}")
    P = predict(weights_dir, X, street)

    fig_mae_by_street(P, Y, street, out_dir / "fig_mae_by_street.pdf")
    fig_pareto(out_dir / "fig_pareto.pdf")
    fig_residuals(P, Y, street, out_dir / "fig_residuals.pdf")
    fig_calibration(P, Y, out_dir / "fig_calibration.pdf")

    # Print the same numbers Table I of the report cites, so the user can cross-check.
    print("\nMAE by street (compare to Table I of the report):")
    for s in range(4):
        m = street == s
        mw = 100.0 * np.abs(P[m, 0] - Y[m, 0]).mean()
        mt = 100.0 * np.abs(P[m, 1] - Y[m, 1]).mean()
        me = 100.0 * np.abs((P[m, 0] + P[m, 1] / 2)
                            - (Y[m, 0] + Y[m, 1] / 2)).mean()
        print(f"  {STREETS[s]:8s}  win {mw:.2f}%  tie {mt:.2f}%  equity {me:.2f}%")
    me_all = 100.0 * np.abs((P[:, 0] + P[:, 1] / 2)
                            - (Y[:, 0] + Y[:, 1] / 2)).mean()
    print(f"  {'Overall':8s}  equity {me_all:.2f}%")


if __name__ == "__main__":
    main()