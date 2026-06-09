#!/usr/bin/env python
"""Final analysis: aggregate ALL 140 experiments and produce publication tables."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

RUNS = Path("outputs/runs")

def load_runs():
    rows = []
    for d in RUNS.iterdir():
        if not d.name.startswith("v2_"):
            continue
        mp = d / "metrics.json"
        cp = d / "config.json"
        if not mp.exists() or not cp.exists():
            continue
        m = json.loads(mp.read_text())
        c = json.loads(cp.read_text())

        # also load member/nonmember separation if available
        sep = None
        ap = d / "attack_outputs.npy"
        mbp = d / "membership.npy"
        if ap.exists() and mbp.exists():
            try:
                ao = np.load(ap)
                mb = np.load(mbp)
                mem = ao[mb == 1]
                non = ao[mb == 0]
                if len(mem) > 0 and len(non) > 0:
                    sep = float(np.mean(mem) - np.mean(non))
            except Exception:
                pass

        rows.append(dict(
            run_id=d.name,
            dataset=c.get("dataset"),
            model=c.get("model"),
            variant=c.get("model_variant") or "N/A",
            train=c.get("train_defense") or "none",
            release=c.get("release_defense") or "none",
            eps=c.get("eps"),
            test_auc=m.get("test_auc"),
            mia_auc=m.get("mia_auc"),
            tpr_gap=m.get("worst_group_tpr_gap"),
            fpr_gap=m.get("worst_group_fpr_gap"),
            fnr_gap=m.get("worst_group_fnr_gap"),
            separation=sep,
        ))
    return rows


def agg(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v))


def fmt(mean, std, dp=3):
    if mean is None:
        return "---"
    return f"{mean:.{dp}f} +/- {std:.{dp}f}"


def fmt_latex(mean, std, dp=3):
    if mean is None:
        return "---"
    return f"${mean:.{dp}f} \\pm {std:.{dp}f}$"


def main():
    rows = load_runs()
    print(f"Total runs loaded: {len(rows)}")

    # group by condition key
    groups = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["model"], r["variant"], r["train"], r["release"], r["eps"])
        groups[key].append(r)

    print(f"Unique conditions: {len(groups)}\n")

    # ======================================================================
    # TABLE 1: OULAD main results (no perturb)
    # ======================================================================
    print("=" * 90)
    print("TABLE 1: OULAD Privacy-Utility-Fairness (no output perturbation)")
    print("=" * 90)
    oulad_keys = [
        ("OULAD", "LR",  "N/A",   "none",   "none", None),
        ("OULAD", "MLP", "small", "none",   "none", None),
        ("OULAD", "MLP", "small", "DP-SGD", "none", 1),
        ("OULAD", "MLP", "small", "DP-SGD", "none", 5),
        ("OULAD", "MLP", "small", "DP-SGD", "none", 10),
    ]
    header = f"{'Condition':<25} {'n':>3} {'Test AUC':<17} {'MIA AUC':<17} {'TPR Gap':<17} {'FPR Gap':<17}"
    print(header)
    print("-" * 90)
    for k in oulad_keys:
        rs = groups.get(k, [])
        label = "LR Baseline" if k[1] == "LR" else \
                "MLP Baseline" if k[3] == "none" else \
                f"MLP DP-SGD e={k[5]}"
        ta_m, ta_s = agg([r["test_auc"] for r in rs])
        ma_m, ma_s = agg([r["mia_auc"] for r in rs])
        tg_m, tg_s = agg([r["tpr_gap"] for r in rs])
        fg_m, fg_s = agg([r["fpr_gap"] for r in rs])
        print(f"{label:<25} {len(rs):>3} {fmt(ta_m, ta_s):<17} {fmt(ma_m, ma_s):<17} {fmt(tg_m, tg_s):<17} {fmt(fg_m, fg_s):<17}")
    print()

    # ======================================================================
    # TABLE 2: Layered defense comparison
    # ======================================================================
    print("=" * 90)
    print("TABLE 2: Layered Defense - Effect of Output Perturbation")
    print("=" * 90)
    layered_pairs = [
        (("OULAD","MLP","small","none","none",None),
         ("OULAD","MLP","small","none","output_perturbation",None),
         "Baseline"),
        (("OULAD","MLP","small","DP-SGD","none",1),
         ("OULAD","MLP","small","DP-SGD","output_perturbation",1),
         "DP-SGD e=1"),
        (("OULAD","MLP","small","DP-SGD","none",5),
         ("OULAD","MLP","small","DP-SGD","output_perturbation",5),
         "DP-SGD e=5"),
    ]
    print(f"{'Condition':<15} {'Perturb':<8} {'n':>3} {'Test AUC':<17} {'MIA AUC':<17} {'TPR Gap':<17} {'FPR Gap':<17}")
    print("-" * 100)
    for k_no, k_yes, label in layered_pairs:
        for perturb, k in [("No", k_no), ("Yes", k_yes)]:
            rs = groups.get(k, [])
            ta_m, ta_s = agg([r["test_auc"] for r in rs])
            ma_m, ma_s = agg([r["mia_auc"] for r in rs])
            tg_m, tg_s = agg([r["tpr_gap"] for r in rs])
            fg_m, fg_s = agg([r["fpr_gap"] for r in rs])
            lbl = label if perturb == "No" else ""
            print(f"{lbl:<15} {perturb:<8} {len(rs):>3} {fmt(ta_m, ta_s):<17} {fmt(ma_m, ma_s):<17} {fmt(tg_m, tg_s):<17} {fmt(fg_m, fg_s):<17}")
        print()

    # ======================================================================
    # TABLE 3: UCI697 validation
    # ======================================================================
    print("=" * 90)
    print("TABLE 3: UCI697 Cross-Dataset Validation (privacy-utility only)")
    print("=" * 90)
    uci_keys = [
        ("UCI697", "MLP", "small", "none",   "none", None),
        ("UCI697", "MLP", "small", "DP-SGD", "none", 1),
        ("UCI697", "MLP", "small", "DP-SGD", "none", 5),
        ("UCI697", "MLP", "small", "DP-SGD", "none", 10),
    ]
    print(f"{'Condition':<25} {'n':>3} {'Test AUC':<17} {'MIA AUC':<17}")
    print("-" * 60)
    for k in uci_keys:
        rs = groups.get(k, [])
        label = "Baseline" if k[3] == "none" else f"DP-SGD e={k[5]}"
        ta_m, ta_s = agg([r["test_auc"] for r in rs])
        ma_m, ma_s = agg([r["mia_auc"] for r in rs])
        print(f"{label:<25} {len(rs):>3} {fmt(ta_m, ta_s):<17} {fmt(ma_m, ma_s):<17}")
    print()

    # ======================================================================
    # TABLE 4: Overfitting / MIA separation explanation
    # ======================================================================
    print("=" * 90)
    print("TABLE 4: Overfitting Explanation - Member vs Non-Member Separation")
    print("=" * 90)
    sep_keys = [
        ("OULAD","MLP","small","none","none",None, "Baseline"),
        ("OULAD","MLP","small","DP-SGD","none",1, "DP-SGD e=1"),
        ("OULAD","MLP","small","DP-SGD","none",5, "DP-SGD e=5"),
        ("OULAD","MLP","small","DP-SGD","none",10, "DP-SGD e=10"),
    ]
    print(f"{'Condition':<25} {'n':>3} {'Test AUC':<17} {'MIA AUC':<17} {'Mem-NonMem Sep':<17}")
    print("-" * 80)
    for *k, label in sep_keys:
        k = tuple(k)
        rs = groups.get(k, [])
        ta_m, ta_s = agg([r["test_auc"] for r in rs])
        ma_m, ma_s = agg([r["mia_auc"] for r in rs])
        sp_m, sp_s = agg([r["separation"] for r in rs])
        print(f"{label:<25} {len(rs):>3} {fmt(ta_m, ta_s):<17} {fmt(ma_m, ma_s):<17} {fmt(sp_m, sp_s, 4):<17}")
    print()

    # ======================================================================
    # LATEX TABLES
    # ======================================================================
    print("=" * 90)
    print("LATEX: Table 1 - OULAD Main Results")
    print("=" * 90)
    print(r"""\begin{table}[t]
\centering
\caption{Privacy-Utility-Fairness Results on OULAD (MLP-small, 5 seeds)}
\label{tab:oulad_results}
\begin{tabular}{lcccccc}
\toprule
Defense & $\epsilon$ & Test AUC & MIA AUC & TPR Gap & FPR Gap \\
\midrule""")
    for k in oulad_keys:
        rs = groups.get(k, [])
        if k[1] == "LR":
            label = "LR Baseline"
        elif k[3] == "none":
            label = "MLP Baseline"
        else:
            label = f"MLP DP-SGD"
        eps_str = str(k[5]) if k[5] else "--"
        ta = fmt_latex(*agg([r["test_auc"] for r in rs]))
        ma = fmt_latex(*agg([r["mia_auc"] for r in rs]))
        tg = fmt_latex(*agg([r["tpr_gap"] for r in rs]))
        fg = fmt_latex(*agg([r["fpr_gap"] for r in rs]))
        print(f"{label} & {eps_str} & {ta} & {ma} & {tg} & {fg} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    print()

    print("=" * 90)
    print("LATEX: Table 2 - UCI697 Validation")
    print("=" * 90)
    print(r"""\begin{table}[t]
\centering
\caption{UCI697 Cross-Dataset Privacy-Utility Validation (MLP-small, 5 seeds)}
\label{tab:uci697_results}
\begin{tabular}{lccc}
\toprule
Defense & $\epsilon$ & Test AUC & MIA AUC \\
\midrule""")
    for k in uci_keys:
        rs = groups.get(k, [])
        label = "Baseline" if k[3] == "none" else "DP-SGD"
        eps_str = str(k[5]) if k[5] else "--"
        ta = fmt_latex(*agg([r["test_auc"] for r in rs]))
        ma = fmt_latex(*agg([r["mia_auc"] for r in rs]))
        print(f"{label} & {eps_str} & {ta} & {ma} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    print()

    print("=" * 90)
    print("LATEX: Table 3 - Layered Defense Effect")
    print("=" * 90)
    print(r"""\begin{table}[t]
\centering
\caption{Effect of Output Perturbation on Fairness (OULAD MLP-small)}
\label{tab:layered_defense}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Training & Output Perturb & Test AUC & MIA AUC & TPR Gap & FPR Gap \\
\midrule""")
    for k_no, k_yes, label in layered_pairs:
        for perturb, k in [("No", k_no), ("Yes", k_yes)]:
            rs = groups.get(k, [])
            ta = fmt_latex(*agg([r["test_auc"] for r in rs]))
            ma = fmt_latex(*agg([r["mia_auc"] for r in rs]))
            tg = fmt_latex(*agg([r["tpr_gap"] for r in rs]))
            fg = fmt_latex(*agg([r["fpr_gap"] for r in rs]))
            lbl = label if perturb == "No" else ""
            print(f"{lbl} & {perturb} & {ta} & {ma} & {tg} & {fg} \\\\")
        if label != "DP-SGD e=5":
            print(r"\addlinespace")
    print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")
    print()

    # Summary
    print("=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)
    total = len(rows)
    uci = sum(1 for r in rows if r["dataset"] == "UCI697")
    oulad = sum(1 for r in rows if r["dataset"] == "OULAD")
    print(f"Total experiments: {total}")
    print(f"  OULAD: {oulad}")
    print(f"  UCI697: {uci}")
    print(f"Unique conditions: {len(groups)}")


if __name__ == "__main__":
    main()
