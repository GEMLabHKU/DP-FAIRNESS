#!/usr/bin/env python
"""Generate final publication-ready tables from ALL 140 experiments."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

RUNS = Path("outputs/runs")

def load():
    rows = []
    for d in RUNS.iterdir():
        if not d.name.startswith("v2_"):
            continue
        mp, cp = d / "metrics.json", d / "config.json"
        if not mp.exists() or not cp.exists():
            continue
        m = json.loads(mp.read_text())
        c = json.loads(cp.read_text())
        sep = None
        ap, mbp = d / "attack_outputs.npy", d / "membership.npy"
        if ap.exists() and mbp.exists():
            try:
                ao, mb = np.load(ap), np.load(mbp)
                sep = float(np.mean(ao[mb==1]) - np.mean(ao[mb==0]))
            except Exception:
                pass
        rows.append(dict(
            rid=d.name, ds=c.get("dataset"), model=c.get("model"),
            var=c.get("model_variant") or "N/A",
            train=c.get("train_defense") or "none",
            release=c.get("release_defense") or "none",
            eps=c.get("eps"),
            auc=m.get("test_auc"), mia=m.get("mia_auc"),
            tpr=m.get("worst_group_tpr_gap"), fpr=m.get("worst_group_fpr_gap"),
            sep=sep))
    return rows

def a(vals):
    v = [x for x in vals if x is not None]
    return (np.mean(v), np.std(v)) if v else (None, None)

def f(m, s, d=3):
    return f"{m:.{d}f} +/- {s:.{d}f}" if m is not None else "---"

def fl(m, s, d=3):
    return f"${m:.{d}f} \\pm {s:.{d}f}$" if m is not None else "---"

def main():
    rows = load()
    g = defaultdict(list)
    for r in rows:
        g[(r["ds"],r["model"],r["var"],r["train"],r["release"],r["eps"])].append(r)
    
    print(f"Total: {len(rows)} runs, {len(g)} conditions\n")

    # ---- TABLE 1: OULAD Main ----
    print("="*100)
    print("TABLE 1: OULAD Main Results (Training-Time Privacy)")
    print("="*100)
    t1 = [
        (("OULAD","LR","N/A","none","none",None),     "LR Baseline", "--"),
        (("OULAD","MLP","small","none","none",None),   "MLP Baseline","--"),
        (("OULAD","MLP","small","DP-SGD","none",1),    "MLP DP-SGD",  "1"),
        (("OULAD","MLP","small","DP-SGD","none",5),    "MLP DP-SGD",  "5"),
        (("OULAD","MLP","small","DP-SGD","none",10),   "MLP DP-SGD",  "10"),
    ]
    print(f"{'Defense':<18} {'eps':>4} {'n':>3} {'Test AUC':<16} {'MIA AUC':<16} {'TPR Gap':<16} {'FPR Gap':<16}")
    print("-"*95)
    for k, lab, ep in t1:
        rs = g.get(k, [])
        print(f"{lab:<18} {ep:>4} {len(rs):>3} {f(*a([r['auc'] for r in rs])):<16} {f(*a([r['mia'] for r in rs])):<16} {f(*a([r['tpr'] for r in rs])):<16} {f(*a([r['fpr'] for r in rs])):<16}")

    # ---- TABLE 2: Layered Defense ----
    print(f"\n{'='*100}")
    print("TABLE 2: Layered Defense — Effect of Output Perturbation (OULAD MLP-small)")
    print("="*100)
    pairs = [
        ("Baseline",    ("OULAD","MLP","small","none","none",None),
                        ("OULAD","MLP","small","none","output_perturbation",None)),
        ("DP-SGD e=1",  ("OULAD","MLP","small","DP-SGD","none",1),
                        ("OULAD","MLP","small","DP-SGD","output_perturbation",1)),
        ("DP-SGD e=5",  ("OULAD","MLP","small","DP-SGD","none",5),
                        ("OULAD","MLP","small","DP-SGD","output_perturbation",5)),
    ]
    print(f"{'Training':<14} {'Perturb':<8} {'n':>3} {'Test AUC':<16} {'MIA AUC':<16} {'TPR Gap':<16} {'FPR Gap':<16}")
    print("-"*95)
    for lab, k_no, k_yes in pairs:
        for tag, k in [("No", k_no), ("Yes", k_yes)]:
            rs = g.get(k, [])
            lbl = lab if tag == "No" else ""
            print(f"{lbl:<14} {tag:<8} {len(rs):>3} {f(*a([r['auc'] for r in rs])):<16} {f(*a([r['mia'] for r in rs])):<16} {f(*a([r['tpr'] for r in rs])):<16} {f(*a([r['fpr'] for r in rs])):<16}")
        print()

    # ---- TABLE 3: UCI697 ----
    print(f"{'='*100}")
    print("TABLE 3: UCI697 Cross-Dataset Validation")
    print("="*100)
    t3 = [
        (("UCI697","MLP","small","none","none",None),   "Baseline","--"),
        (("UCI697","MLP","small","DP-SGD","none",1),    "DP-SGD",  "1"),
        (("UCI697","MLP","small","DP-SGD","none",5),    "DP-SGD",  "5"),
        (("UCI697","MLP","small","DP-SGD","none",10),   "DP-SGD",  "10"),
    ]
    print(f"{'Defense':<18} {'eps':>4} {'n':>3} {'Test AUC':<16} {'MIA AUC':<16}")
    print("-"*60)
    for k, lab, ep in t3:
        rs = g.get(k, [])
        print(f"{lab:<18} {ep:>4} {len(rs):>3} {f(*a([r['auc'] for r in rs])):<16} {f(*a([r['mia'] for r in rs])):<16}")
    print("Note: UCI697 lacks demographics; fairness metrics intentionally omitted.")

    # ---- TABLE 4: Overfitting/MIA ----
    print(f"\n{'='*100}")
    print("TABLE 4: Overfitting Explanation (OULAD MLP-small)")
    print("="*100)
    t4 = [
        (("OULAD","MLP","small","none","none",None),   "Baseline"),
        (("OULAD","MLP","small","DP-SGD","none",1),    "DP-SGD e=1"),
        (("OULAD","MLP","small","DP-SGD","none",5),    "DP-SGD e=5"),
        (("OULAD","MLP","small","DP-SGD","none",10),   "DP-SGD e=10"),
    ]
    print(f"{'Condition':<18} {'n':>3} {'Test AUC':<16} {'MIA AUC':<16} {'Mem-NonMem Sep':<20}")
    print("-"*80)
    for k, lab in t4:
        rs = g.get(k, [])
        print(f"{lab:<18} {len(rs):>3} {f(*a([r['auc'] for r in rs])):<16} {f(*a([r['mia'] for r in rs])):<16} {f(*a([r['sep'] for r in rs]),4):<20}")

    # ---- LATEX OUTPUT ----
    print(f"\n\n{'#'*100}")
    print("# LATEX TABLES (copy-paste ready)")
    print('#'*100)

    # LaTeX Table 1
    print(r"""
\begin{table}[t]
\centering
\caption{Privacy-Utility-Fairness Results on OULAD (5 seeds per condition)}
\label{tab:oulad}
\begin{tabular}{llccccc}
\toprule
Model & Defense & $\epsilon$ & Test AUC & MIA AUC & TPR Gap & FPR Gap \\
\midrule""")
    for k, lab, ep in t1:
        rs = g.get(k, [])
        mod = lab.split()[0]
        defe = " ".join(lab.split()[1:])
        print(f"{mod} & {defe} & {ep} & {fl(*a([r['auc'] for r in rs]))} & {fl(*a([r['mia'] for r in rs]))} & {fl(*a([r['tpr'] for r in rs]))} & {fl(*a([r['fpr'] for r in rs]))} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # LaTeX Table 2 (Layered)
    print(r"""
\begin{table}[t]
\centering
\caption{Effect of Output Perturbation on Fairness (OULAD MLP-small)}
\label{tab:layered}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
Training & Perturb & Test AUC & MIA AUC & TPR Gap & FPR Gap \\
\midrule""")
    for lab, k_no, k_yes in pairs:
        for tag, k in [("No", k_no), ("Yes", k_yes)]:
            rs = g.get(k, [])
            lbl = lab if tag == "No" else ""
            if not rs:
                continue
            print(f"{lbl} & {tag} & {fl(*a([r['auc'] for r in rs]))} & {fl(*a([r['mia'] for r in rs]))} & {fl(*a([r['tpr'] for r in rs]))} & {fl(*a([r['fpr'] for r in rs]))} \\\\")
        print(r"\addlinespace")
    print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")

    # LaTeX Table 3 (UCI697)
    print(r"""
\begin{table}[t]
\centering
\caption{UCI697 Cross-Dataset Privacy-Utility Validation (MLP-small, 5 seeds)}
\label{tab:uci697}
\begin{tabular}{lccc}
\toprule
Defense & $\epsilon$ & Test AUC & MIA AUC \\
\midrule""")
    for k, lab, ep in t3:
        rs = g.get(k, [])
        print(f"{lab} & {ep} & {fl(*a([r['auc'] for r in rs]))} & {fl(*a([r['mia'] for r in rs]))} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print(f"\n{'='*100}")
    print(f"TOTAL: {len(rows)} experiments across {len(g)} conditions")
    print(f"  OULAD: {sum(1 for r in rows if r['ds']=='OULAD')}")
    print(f"  UCI697: {sum(1 for r in rows if r['ds']=='UCI697')}")

if __name__ == "__main__":
    main()
