# assignment01_part1.py
# Part 1: Apriori + Association Rules on the toy dataset
# - Reusable core (apriori, generate_rules, etc.)
# - Runs the toy example with Smin=0.1 and Cmin=0.3
# - Saves CSVs and a scatter plot (x=confidence, y=support, size ∝ lift)

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Set, Dict, FrozenSet

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Core: supports, Apriori, rules
# -----------------------------

def support(itemset: FrozenSet[str], transactions: List[Set[str]]) -> float:
    """Relative support of an itemset."""
    if not transactions:
        return 0.0
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)


def _generate_candidates(prev_frequents: List[FrozenSet[str]], k: int):
    """Join step: from frequent (k-1)-itemsets, propose k-itemset candidates."""
    candidates = set()
    prev_list = list(prev_frequents)
    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            union = prev_list[i] | prev_list[j]
            if len(union) == k:
                candidates.add(frozenset(union))
    return candidates


def _prune(candidates, prev_frequents_set, k: int):
    """Prune step: every (k-1)-subset of a candidate must be frequent."""
    pruned = set()
    for c in candidates:
        ok = True
        for subset in itertools.combinations(c, k - 1):
            if frozenset(subset) not in prev_frequents_set:
                ok = False
                break
        if ok:
            pruned.add(c)
    return pruned


def apriori(transactions: List[Set[str]], minsup: float) -> Dict[int, Dict[FrozenSet[str], float]]:
    """Return dict level->(itemset->support) of all frequent itemsets."""
    # L1
    item_counts = Counter()
    for t in transactions:
        for i in t:
            item_counts[i] += 1
    n = len(transactions)
    L: Dict[int, Dict[FrozenSet[str], float]] = {}
    L1 = {frozenset([i]): c / n for i, c in item_counts.items() if (c / n) >= minsup}
    if not L1:
        return {}
    L[1] = L1

    # Lk, k>=2
    k = 2
    prev = list(L1.keys())
    while prev:
        Ck = _generate_candidates(prev, k)
        Ck = _prune(Ck, set(prev), k)
        Lk = {}
        for c in Ck:
            s = support(c, transactions)
            if s >= minsup:
                Lk[c] = s
        if not Lk:
            break
        L[k] = Lk
        prev = list(Lk.keys())
        k += 1
    return L


@dataclass
class Rule:
    antecedent: FrozenSet[str]
    consequent: FrozenSet[str]
    support: float
    confidence: float
    lift: float


def generate_rules(frequents: Dict[int, Dict[FrozenSet[str], float]],
                   transactions: List[Set[str]],
                   minconf: float) -> List[Rule]:
    """Create rules A->B from frequent itemsets with confidence>=minconf."""
    # Lookup of supports (with fallback if some subset wasn’t stored)
    sup = {}
    for _, d in frequents.items():
        sup.update(d)

    def sup_get(X: FrozenSet[str]) -> float:
        return sup.get(X, support(X, transactions))

    rules: List[Rule] = []
    for k, d in frequents.items():
        if k < 2:
            continue
        for itemset, sup_xy in d.items():
            items = list(itemset)
            for r in range(1, len(items)):  # all non-empty proper antecedents
                for A in itertools.combinations(items, r):
                    A = frozenset(A)
                    B = itemset - A
                    if not B:
                        continue
                    sup_x = sup_get(A)
                    if sup_x == 0:
                        continue
                    conf = sup_xy / sup_x
                    if conf >= minconf:
                        sup_y = sup_get(B)
                        lift = conf / sup_y if sup_y > 0 else float("inf")
                        rules.append(Rule(A, B, sup_xy, conf, lift))
    # Sort by interestingness
    rules.sort(key=lambda r: (r.lift, r.confidence, r.support), reverse=True)
    return rules


# -----------------------------
# Helpers: tabular + plotting
# -----------------------------

def frequents_to_df(freqs: Dict[int, Dict[FrozenSet[str], float]]) -> pd.DataFrame:
    rows = []
    for k, d in freqs.items():
        for iset, supv in d.items():
            rows.append({
                "k": k,
                "itemset": ", ".join(sorted(iset)),
                "support": supv
            })
    return pd.DataFrame(rows).sort_values(["k", "support"], ascending=[True, False]).reset_index(drop=True)


def rules_to_df(rules: List[Rule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "antecedent": ", ".join(sorted(r.antecedent)),
        "consequent": ", ".join(sorted(r.consequent)),
        "support": r.support,
        "confidence": r.confidence,
        "lift": r.lift
    } for r in rules])


def plot_rules_scatter(df_rules: pd.DataFrame, title: str, out_png: Path):
    if df_rules.empty:
        return
    plt.figure()
    plt.scatter(df_rules["confidence"], df_rules["support"], s=df_rules["lift"] * 30.0)
    plt.xlabel("Confidence")
    plt.ylabel("Support")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Part 1: Toy dataset (runnable)
# -----------------------------

def main():
    # Toy transactions from the assignment (players’ inventories)
    transactions = [
        {"Elixir", "Shield"},                              # Player 1
        {"Gem", "Shield"},                                 # Player 2
        {"Elixir", "Sword"},                               # Player 3
        {"Elixir", "Wand", "Shield"},                      # Player 4
        {"Elixir", "Sword"},                               # Player 5
        {"Wand", "Shield"},                                # Player 6
        {"Elixir", "Wand", "Shield", "Sword"},             # Player 7
        {"Elixir", "Giant Wand"},                          # Player 8
    ]

    Smin = 0.1  # minimum support
    Cmin = 0.3  # minimum confidence

    freqs = apriori([set(t) for t in transactions], Smin)
    rules = generate_rules(freqs, [set(t) for t in transactions], Cmin)

    outdir = Path("artifacts_part1")
    outdir.mkdir(exist_ok=True, parents=True)

    df_freqs = frequents_to_df(freqs)
    df_rules = rules_to_df(rules)

    df_freqs.to_csv(outdir / "part1_frequents.csv", index=False)
    df_rules.to_csv(outdir / "part1_rules.csv", index=False)

    plot_rules_scatter(df_rules, "Part 1 — Rules (size ∝ Lift)", outdir / "part1_rules_scatter.png")

    print(f"Saved: {outdir/'part1_frequents.csv'}")
    print(f"Saved: {outdir/'part1_rules.csv'}")
    print(f"Saved: {outdir/'part1_rules_scatter.png'}")


if __name__ == "__main__":
    main()
