"""
train_approx.py - learn to approximate "equity vs one random opponent" with a small
neural network, then export the weights for C++ inference (src/approx.cpp).

INPUT  : 62 cheap features of (hero hole cards, board) written by `poker gen-dataset`
         (src/features.cpp). No opponent enumeration is needed to compute them.
TARGET : two numbers per state: P(win) and P(tie) vs ONE random opponent, produced by
         the C++ equity engine (exact on turn/river, stratified Monte Carlo on
         preflop/flop). Equity is then win + tie/2.
MODEL  : fully connected net 62 -> 64 -> 32 -> 2, ELU hidden units, sigmoid outputs,
         mean-squared-error loss, Adam optimiser. This is the same recipe as the
         reference paper (p-24-12-2, ELU, sigmoid, Adam, MSE), made a little wider
         because our feature vector is larger.

Why plain numpy and not PyTorch: this sandbox has no PyTorch. The network is small
enough that the hand-written forward/backward pass below is short and, being explicit,
easy to check. In your own environment the same architecture is ~15 lines of PyTorch.

Why synthetic data: there is no dataset of "true equities" to download. The labels come
from simulating the exact process we want to learn (a uniformly random deal), so they
are unbiased samples of the truth, with a known and reported noise level (`eq_se`).

Usage:  python train_approx.py --train train.csv --test test.csv --out approx_weights.bin
"""
import argparse
import struct
import time

import numpy as np

N_FEATURES = 62
STREETS = ["preflop", "flop", "turn", "river"]
CATEGORIES = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House",
              "Quads", "Straight Flush"]


# --------------------------------------------------------------------------- data
def load_csv(path):
    """Read a CSV written by `poker gen-dataset`. Returns (X, Y, street, label_se)."""
    # columns: hero, board (text, skipped) | street, win, tie, eq_se | f0..f61
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(2, 6 + N_FEATURES))
    street = arr[:, 0].astype(int)
    Y = arr[:, 1:3]                       # P(win), P(tie)
    se = arr[:, 3]
    X = arr[:, 4:]
    return X.astype(np.float64), Y.astype(np.float64), street, se


# --------------------------------------------------------------------------- model
def elu(z):
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """Fully connected net; hidden layers ELU, output layer sigmoid."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, X, keep=False):
        """Return predictions (and the per-layer activations if keep=True, for backprop)."""
        acts = [X]
        h = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = sigmoid(z) if l == len(self.W) - 1 else elu(z)
            acts.append(h)
        return (h, acts) if keep else h

    def gradients(self, X, T):
        """MSE loss gradients for one batch (manual backprop)."""
        Y, acts = self.forward(X, keep=True)
        n = X.shape[0] * T.shape[1]
        delta = 2.0 * (Y - T) / n * Y * (1 - Y)          # d loss / d (pre-sigmoid output)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = acts[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (acts[l] + 1.0 * (acts[l] <= 0))  # ELU': 1 if z>0 else a+1
        return gW, gb

    def params(self):
        return self.W + self.b


class Adam:
    """Adam optimiser (Kingma & Ba) updating the arrays in place."""

    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t, self.b1, self.b2, self.eps = 0, b1, b2, eps

    def step(self, params, grads, lr):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + self.eps)


def equity(Y):
    return Y[:, 0] + 0.5 * Y[:, 1]


# --------------------------------------------------------------------------- export
def export_weights(net, path):
    """Write the PKAP binary format read by src/approx.cpp (float32, little endian)."""
    with open(path, "wb") as f:
        f.write(b"PKAP")
        f.write(struct.pack("<i", len(net.W)))
        for W, b in zip(net.W, net.b):
            f.write(struct.pack("<ii", W.shape[0], W.shape[1]))
            f.write(W.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())


# --------------------------------------------------------------------------- reporting
def report(name, net, X, Y, street, se, baseline_means):
    P = net.forward(X)
    out = []
    ae = np.abs(equity(P) - equity(Y))
    out.append(f"{name}: {len(X)} rows")
    out.append(f"  MAE  win {100*np.abs(P[:,0]-Y[:,0]).mean():.2f}%  tie {100*np.abs(P[:,1]-Y[:,1]).mean():.2f}%"
               f"  equity {100*ae.mean():.2f}%")
    out.append("  equity error percentiles: " + "  ".join(
        f"p{q}={100*np.percentile(ae, q):.2f}%" for q in (50, 90, 99)) + f"  max={100*ae.max():.2f}%")
    out.append("  share of predictions within: " + "  ".join(
        f"{t}%: {100*(ae <= t/100).mean():.1f}%" for t in (1, 2, 5, 10)))
    base = np.abs(baseline_means[street] - equity(Y)).mean()
    out.append(f"  baseline (predict the per-street average equity): MAE {100*base:.2f}%")
    out.append(f"  label noise: mean reported label SE {100*se.mean():.2f}% "
               f"(expected MAE floor from noise alone ~{100*0.8*se.mean():.2f}%)")
    out.append("  by street (equity MAE):  " + "  ".join(
        f"{STREETS[s]} {100*ae[street == s].mean():.2f}%" for s in range(4) if (street == s).any()))
    cat = X[:, 40:49].argmax(axis=1)
    out.append("  by hero's made hand (equity MAE, rows):")
    for c in range(9):
        m = cat == c
        if m.sum() >= 50:
            out.append(f"     {CATEGORIES[c]:<14} {100*ae[m].mean():5.2f}%   n={m.sum()}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="approx_weights.bin")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, Y, street, se = load_csv(args.train)
    Xt, Yt, st_t, se_t = load_csv(args.test)
    print(f"loaded train {X.shape}, test {Xt.shape}")

    # 90/10 train/validation split (validation picks the best epoch; test is never touched)
    perm = rng.permutation(len(X))
    n_val = len(X) // 10
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X[ti], Y[ti]
    Xv, Yv = X[vi], Y[vi]

    sizes = [N_FEATURES] + args.hidden + [2]
    net = MLP(sizes, rng)
    opt = Adam(net.params())
    n_params = sum(p.size for p in net.params())
    print(f"architecture {sizes}, {n_params} parameters ({n_params*4/1024:.1f} KB as float32)")

    best_val, best_state = 1e9, None
    t0 = time.time()
    for ep in range(args.epochs):
        lr = 1e-4 + 0.5 * (args.lr - 1e-4) * (1 + np.cos(np.pi * ep / args.epochs))  # cosine decay
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            gW, gb = net.gradients(Xtr[idx], Ytr[idx])
            opt.step(net.params(), gW + gb, lr)
        if ep % 5 == 4 or ep == args.epochs - 1:
            val = np.abs(equity(net.forward(Xv)) - equity(Yv)).mean()
            if val < best_val:
                best_val, best_state = val, [p.copy() for p in net.params()]
            print(f"epoch {ep+1:3d}  lr {lr:.5f}  validation equity MAE {100*val:.3f}%  ({time.time()-t0:.0f}s)", flush=True)

    for p, s in zip(net.params(), best_state):
        p[:] = s

    base = np.array([equity(Y)[street == s].mean() for s in range(4)])
    text = report("TEST SET (never used for training or epoch selection)", net, Xt, Yt, st_t, se_t, base)
    print("\n" + text)
    export_weights(net, args.out)
    print(f"\nwrote {args.out}")
    with open(args.out + ".report.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
