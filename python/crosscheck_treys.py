"""
crosscheck_treys.py - independent check of the C++ hand evaluator against `treys`.

The exhaustive test in tests/validate.cpp proves the C++ evaluator classifies every
7-card hand correctly. This script additionally checks the ORDER between hands: for
random pairs of 7-card hands, "which hand wins?" must agree with treys (a separate
implementation, where a LOWER score is better).

Usage:  python crosscheck_treys.py /path/to/poker_binary [n_pairs]
Needs:  pip install treys
"""
import random
import subprocess
import sys

from treys import Card, Evaluator

RANKS, SUITS = "23456789TJQKA", "cdhs"


def main():
    exe = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    rng = random.Random(7)
    deck = [r + s for r in RANKS for s in SUITS]

    hands = []
    for _ in range(2 * n):
        hands.append(rng.sample(deck, 7))
    stdin = "\n".join("".join(h) for h in hands) + "\n"
    out = subprocess.run([exe, "score"], input=stdin, capture_output=True, text=True, check=True).stdout
    mine = [int(line.split()[0]) for line in out.strip().split("\n")]

    ev = Evaluator()
    theirs = []
    for h in hands:
        cards = [Card.new(c) for c in h]
        theirs.append(ev.evaluate(cards[:5], cards[5:]))

    agree = 0
    for i in range(n):
        a, b = 2 * i, 2 * i + 1
        m = (mine[a] > mine[b]) - (mine[a] < mine[b])           # +1 = first hand wins
        t = (theirs[a] < theirs[b]) - (theirs[a] > theirs[b])    # treys: lower is better
        agree += (m == t)
    print(f"{agree}/{n} pairwise comparisons agree with treys")
    sys.exit(0 if agree == n else 1)


if __name__ == "__main__":
    main()
