"""
train_approx.py - learn to approximate "equity vs one random opponent" with a small
neural network, then export the weights for C++ inference (src/approx.cpp).

INPUT  : 62 cheap features of (hero hole cards, board) written by `poker gen-dataset`
         (src/features.cpp). No opponent enumeration is needed to compute them.
TARGET : two numbers per state: P(win) and P(tie) vs ONE random opponent, produced by
         the C++ equity engine (exact on turn/river, stratified Monte Carlo on
         preflop/flop). Equity is then win + tie/2.
MODEL  : fully connected net 62 -> 128 -> 64 -> 2, ELU hidden units, sigmoid outputs,
         mean-squared-error loss, Adam optimiser.

Usage (single model):
  python train_approx.py --train train.csv --test data/test_dataset.csv

Usage (per-street, the shipped configuration):
  python train_approx.py --train train.csv --test data/test_dataset.csv \
                         --per-street --hidden 128 64 --epochs 300 --seed 2

Usage (K-fold CV, report only, no export):
  python train_approx.py --train train.csv --test data/test_dataset.csv --kfold 5

Default output is data/approx_weights.bin (single) or data/approx_weights_s0..s3.bin (per-street).
"""
import argparse
import os
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
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"PKAP")
        f.write(struct.pack("<i", len(net.W)))
        for W, b in zip(net.W, net.b):
            f.write(struct.pack("<ii", W.shape[0], W.shape[1]))
            f.write(W.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())


# --------------------------------------------------------------------------- training
def lr_at(ep, args):
    """Linear warmup then cosine decay to --min-lr."""
    if ep < args.warmup:
        return args.lr * (ep + 1) / max(1, args.warmup)
    t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
    t = min(max(t, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * t))


def train_one(Xtr, Ytr, Xv, Yv, sizes, args, rng, verbose=True, tag=""):
    """Train one network. Returns (net, best_val_mae, best_epoch)."""
    net = MLP(sizes, rng)
    opt = Adam(net.params())
    best_val, best_state, best_ep = 1e9, None, -1
    since_best = 0
    t0 = time.time()
    for ep in range(args.epochs):
        lr = lr_at(ep, args)
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            gW, gb = net.gradients(Xtr[idx], Ytr[idx])
            opt.step(net.params(), gW + gb, lr)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            val = np.abs(equity(net.forward(Xv)) - equity(Yv)).mean()
            if val < best_val:
                best_val, best_state, best_ep = val, [p.copy() for p in net.params()], ep + 1
                since_best = 0
            else:
                since_best += 5
            if verbose:
                print(f"{tag}epoch {ep+1:3d}  lr {lr:.5f}  val equity MAE {100*val:.3f}%  ({time.time()-t0:.0f}s)", flush=True)
            if since_best >= args.patience:
                if verbose:
                    print(f"{tag}early stop: no improvement for {args.patience} epochs")
                break
    for p, s in zip(net.params(), best_state):
        p[:] = s
    return net, best_val, best_ep


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
    ap.add_argument("--out", default="data/approx_weights.bin",
                    help="weights file base name; per-street writes <base>_s0.bin ... <base>_s3.bin")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--per-street", action="store_true")
    ap.add_argument("--kfold", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, Y, street, se = load_csv(args.train)
    Xt, Yt, st_t, se_t = load_csv(args.test)
    print(f"loaded train {X.shape}, test {Xt.shape}")

    # 90/10 train/validation split
    perm = rng.permutation(len(X))
    n_val = len(X) // 10
    vi, ti = perm[:n_val], perm[n_val:]

    sizes = [N_FEATURES] + args.hidden + [2]
    n_params = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))
    print(f"architecture {sizes}, {n_params} parameters ({n_params*4/1024:.1f} KB as float32)")

    if args.kfold > 1 and args.per_street:
        raise SystemExit("--per-street and --kfold are not supported together")

    # ---------------- K-fold cross-validation (report only, no export) ----
    if args.kfold > 1:
        n = len(X)
        print(f"running {args.kfold}-fold cross-validation on {n} training rows")
        idx_all = rng.permutation(n)
        folds = np.array_split(idx_all, args.kfold)
        fold_maes = []
        for k in range(args.kfold):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(args.kfold) if j != k])
            net, bv, be = train_one(X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx],
                                    sizes, args, rng, verbose=False)
            fold_maes.append(bv)
            print(f"  fold {k+1}/{args.kfold}: best epoch {be}, val equity MAE {100*bv:.3f}%")
        print(f"  CV equity MAE: mean {100*np.mean(fold_maes):.3f}%  std {100*np.std(fold_maes):.3f}%")
        return

    # ---------------- Per-street mode: 4 models, one per street ----------
    if args.per_street:
        print(f"per-street mode: training 4 models of architecture {sizes}")
        base = args.out[:-4] if args.out.endswith(".bin") else args.out
        nets = []
        for s in range(4):
            mask_tr = (street[ti] == s)
            mask_v = (street[vi] == s)
            mask_te = (st_t == s)
            if mask_tr.sum() < 100:
                raise SystemExit(f"street {STREETS[s]}: only {mask_tr.sum()} training rows, refusing to train")
            print(f"\n=== {STREETS[s]}: {mask_tr.sum()} train / {mask_v.sum()} val / {mask_te.sum()} test ===")
            net, bv, be = train_one(X[ti][mask_tr], Y[ti][mask_tr], X[vi][mask_v], Y[vi][mask_v],
                                    sizes, args, rng, verbose=True, tag=f"[{STREETS[s]}] ")
            print(f"  best epoch {be}, val MAE {100*bv:.3f}%")
            nets.append(net)
            path = f"{base}_s{s}.bin"
            export_weights(net, path)
            print(f"  wrote {path}")
        print("\nPER-STREET TEST SET")
        total_ae = []
        for s in range(4):
            mask = st_t == s
            P = nets[s].forward(Xt[mask])
            ae = np.abs(equity(P) - equity(Yt[mask]))
            total_ae.append(ae)
            print(f"  {STREETS[s]:8s} n={mask.sum():6d}"
                  f"  MAE win {100*np.abs(P[:,0]-Yt[mask,0]).mean():.2f}%"
                  f"  tie {100*np.abs(P[:,1]-Yt[mask,1]).mean():.2f}%"
                  f"  equity {100*ae.mean():.2f}%")
        print(f"  overall  MAE equity {100*np.concatenate(total_ae).mean():.2f}%")
        return

    # ---------------- Single-model path ------------------------------------
    net, bv, be = train_one(X[ti], Y[ti], X[vi], Y[vi], sizes, args, rng, verbose=True)
    print(f"best epoch {be}, val MAE {100*bv:.3f}%")

    base = np.array([equity(Y)[street == s].mean() for s in range(4)])
    text = report("TEST SET (never used for training or epoch selection)", net, Xt, Yt, st_t, se_t, base)
    print("\n" + text)
    export_weights(net, args.out)
    print(f"\nwrote {args.out}")
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out + ".report.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()