"""Build publication-safe tables from verified v2 artifacts only."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RUNS = Path("outputs/runs")


def load_all():
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.name.startswith("v2_"):
            continue
        cp, mp = d / "config.json", d / "metrics.json"
        if not cp.exists() or not mp.exists():
            continue
        c = json.loads(cp.read_text())
        m = json.loads(mp.read_text())
        rows.append({"id": d.name, "c": c, "m": m})
    return rows


def is_clean_oulad_mlp_no_release(row):
    c = row["c"]
    if c.get("dataset") != "OULAD":
        return False
    if c.get("model") != "MLP" or c.get("model_variant") != "small":
        return False
    if c.get("publish_defense"):
        return False
    rel = c.get("release_defense")
    if rel and rel != "none":
        return False
    return True


def is_lr_baseline(row):
    c = row["c"]
    if c.get("dataset") != "OULAD":
        return False
    if c.get("model") != "LR":
        return False
    if c.get("publish_defense"):
        return False
    if c.get("release_defense") and c.get("release_defense") != "none":
        return False
    if (c.get("train_defense") or "none") != "none":
        return False
    return True


def is_uci697_mlp_clean(row):
    c = row["c"]
    if c.get("dataset") != "UCI697":
        return False
    if c.get("model") != "MLP" or c.get("model_variant") != "small":
        return False
    if c.get("publish_defense"):
        return False
    if c.get("release_defense") and c.get("release_defense") != "none":
        return False
    return True


def agg(ids, rows_by_id, keys):
    rs = [rows_by_id[i] for i in ids]
    out = {}
    for k in keys:
        vals = [r["m"][k] for r in rs if r["m"].get(k) is not None]
        if vals:
            out[k] = (np.mean(vals), np.std(vals), len(vals))
        else:
            out[k] = None
    return out


def fmt_cell(triple, nd=3):
    if not triple:
        return "---"
    mu, sd, n = triple
    return f"{mu:.{nd}f} ± {sd:.{nd}f} (n={n})"


def separation_from_run(row):
    """Member minus non-member mean attack score."""
    rid = row["id"]
    d = RUNS / rid
    ap, mbp = d / "attack_outputs.npy", d / "membership.npy"
    if not ap.exists() or not mbp.exists():
        return None
    ao = np.load(ap)
    mb = np.load(mbp)
    mem, non = ao[mb == 1], ao[mb == 0]
    if len(mem) < 1 or len(non) < 1:
        return None
    return float(np.mean(mem) - np.mean(non))


def main():
    rows = load_all()
    by_id = {r["id"]: r for r in rows}

    # ---- Group OULAD MLP clean ----
    oulad_mlp = [r for r in rows if is_clean_oulad_mlp_no_release(r)]
    by_cond = defaultdict(list)
    for r in oulad_mlp:
        td = r["c"].get("train_defense") or "none"
        eps = r["c"].get("eps")
        by_cond[(td, eps)].append(r["id"])

    print("=== OULAD MLP-small clean groups ===")
    for k in sorted(by_cond.keys(), key=lambda x: (x[0], x[1] or 0)):
        ids = sorted(by_cond[k], key=lambda x: by_id[x]["c"].get("seed", 0))
        print(k, ids)

    # Expected sets
    baseline_ids = sorted(by_cond[("none", None)], key=lambda x: by_id[x]["c"]["seed"])
    d1 = sorted(by_cond[("DP-SGD", 1)], key=lambda x: by_id[x]["c"]["seed"])
    d5 = sorted(by_cond[("DP-SGD", 5)], key=lambda x: by_id[x]["c"]["seed"])
    d10 = sorted(by_cond[("DP-SGD", 10)], key=lambda x: by_id[x]["c"]["seed"])

    lr_rows = [r for r in rows if is_lr_baseline(r)]
    lr_ids = sorted([r["id"] for r in lr_rows], key=lambda x: by_id[x]["c"]["seed"])

    print("\n=== LR baseline ===", lr_ids)

    # UCI697
    uci = [r for r in rows if is_uci697_mlp_clean(r)]
    ucig = defaultdict(list)
    for r in uci:
        td = r["c"].get("train_defense") or "none"
        eps = r["c"].get("eps")
        ucig[(td, eps)].append(r["id"])
    print("\n=== UCI697 ===")
    for k in sorted(ucig.keys(), key=lambda x: (x[0], x[1] or 0)):
        print(k, sorted(ucig[k], key=lambda x: by_id[x]["c"]["seed"]))

    # Perturb: baseline v2_0005-9 vs v2_0025-9
    pert_base = ["v2_0005", "v2_0006", "v2_0007", "v2_0008", "v2_0009"]
    pert_rel = ["v2_0025", "v2_0026", "v2_0027", "v2_0028", "v2_0029"]
    for pid in pert_base + pert_rel:
        assert pid in by_id, pid
        c = by_id[pid]["c"]
        assert c.get("publish_defense") in (None, "output_perturbation")
    # verify v2_0025 has publish
    assert by_id["v2_0025"]["c"].get("publish_defense") == "output_perturbation"

    # DP-SGD + perturb (publish + DP-SGD) — find in repo
    dpert = []
    for r in rows:
        c = r["c"]
        if c.get("dataset") != "OULAD":
            continue
        if c.get("model") != "MLP" or c.get("model_variant") != "small":
            continue
        if c.get("publish_defense") != "output_perturbation":
            continue
        if c.get("train_defense") != "DP-SGD":
            continue
        if c.get("eps") != 5:
            continue
        dpert.append(r["id"])
    dpert = sorted(dpert, key=lambda x: by_id[x]["c"]["seed"])
    print("\n=== DP-SGD e=5 + output perturb (publish_defense) ===", dpert)

    keys = ["test_auc", "mia_auc", "worst_group_tpr_gap", "worst_group_fpr_gap"]

    def line(label, ids):
        ag = agg(ids, by_id, keys)
        sep_vals = [separation_from_run(by_id[i]) for i in ids]
        sep_vals = [s for s in sep_vals if s is not None]
        sep_agg = (np.mean(sep_vals), np.std(sep_vals), len(sep_vals)) if sep_vals else None
        ttg = agg(ids, by_id, ["train_test_gap"])
        return label, ids, ag, sep_agg, ttg.get("train_test_gap")

    # Print markdown tables
    print("\n\n# TABLE A")
    print("| Row | Run IDs (seeds 1–5) | Test AUC | MIA AUC | TPR Gap | FPR Gap |")
    print("|-----|---------------------|----------|---------|---------|---------|")
    for lab, ids in [
        ("MLP-small Baseline", baseline_ids),
        ("MLP-small DP-SGD ε=1", d1),
        ("MLP-small DP-SGD ε=5", d5),
        ("MLP-small DP-SGD ε=10", d10),
    ]:
        ag = agg(ids, by_id, keys)
        idstr = ", ".join(ids)
        print(
            f"| {lab} | `{idstr}` | {fmt_cell(ag['test_auc'])} | {fmt_cell(ag['mia_auc'])} | "
            f"{fmt_cell(ag['worst_group_tpr_gap'])} | {fmt_cell(ag['worst_group_fpr_gap'])} |"
        )
    if lr_ids:
        ag = agg(lr_ids, by_id, keys)
        idstr = ", ".join(lr_ids)
        print(
            f"| LR Baseline | `{idstr}` | {fmt_cell(ag['test_auc'])} | {fmt_cell(ag['mia_auc'])} | "
            f"{fmt_cell(ag['worst_group_tpr_gap'])} | {fmt_cell(ag['worst_group_fpr_gap'])} |"
        )

    print("\n# TABLE B")
    ub = sorted(ucig[("none", None)], key=lambda x: by_id[x]["c"]["seed"])
    u1 = sorted(ucig[("DP-SGD", 1)], key=lambda x: by_id[x]["c"]["seed"])
    u5 = sorted(ucig[("DP-SGD", 5)], key=lambda x: by_id[x]["c"]["seed"])
    u10 = sorted(ucig[("DP-SGD", 10)], key=lambda x: by_id[x]["c"]["seed"])
    print("| Row | Run IDs | Test AUC | MIA AUC |")
    print("|-----|---------|----------|---------|")
    for lab, ids in [
        ("Baseline", ub),
        ("DP-SGD ε=1", u1),
        ("DP-SGD ε=5", u5),
        ("DP-SGD ε=10", u10),
    ]:
        ag = agg(ids, by_id, ["test_auc", "mia_auc"])
        print(f"| {lab} | `{', '.join(ids)}` | {fmt_cell(ag['test_auc'])} | {fmt_cell(ag['mia_auc'])} |")

    print("\n# TABLE C")
    print("| Condition | Run IDs | Test AUC | MIA AUC | Mem−NonMem sep | |train−test| gap |")
    print("|-----------|---------|----------|---------|----------------|------------------|")
    for lab, ids in [
        ("Baseline", baseline_ids),
        ("DP-SGD ε=1", d1),
        ("DP-SGD ε=5", d5),
        ("DP-SGD ε=10", d10),
    ]:
        ag = agg(ids, by_id, ["test_auc", "mia_auc"])
        sep_vals = [separation_from_run(by_id[i]) for i in ids]
        sep_vals = [s for s in sep_vals if s is not None]
        sp = (np.mean(sep_vals), np.std(sep_vals), len(sep_vals)) if sep_vals else None
        ttg = agg(ids, by_id, ["train_test_gap"])
        sp_str = fmt_cell(sp, 4) if sp else "---"
        tt_str = fmt_cell(ttg["train_test_gap"], 4) if ttg.get("train_test_gap") else "---"
        print(
            f"| {lab} | `{', '.join(ids)}` | {fmt_cell(ag['test_auc'])} | {fmt_cell(ag['mia_auc'])} | "
            f"{sp_str} | {tt_str} |"
        )

    print("\n# PERTURBATION SUPPLEMENT")
    print("| Condition | Run IDs | Test AUC (released) | MIA AUC | TPR Gap | FPR Gap |")
    print("|-----------|---------|---------------------|---------|---------|---------|")
    for lab, ids in [
        ("Baseline (no release perturb)", pert_base),
        ("Baseline + output perturb", pert_rel),
    ]:
        ag = agg(ids, by_id, keys)
        print(
            f"| {lab} | `{', '.join(ids)}` | {fmt_cell(ag['test_auc'])} | {fmt_cell(ag['mia_auc'])} | "
            f"{fmt_cell(ag['worst_group_tpr_gap'])} | {fmt_cell(ag['worst_group_fpr_gap'])} |"
        )
    if len(dpert) >= 5:
        ag = agg(dpert, by_id, keys)
        print(
            f"| DP-SGD ε=5 + output perturb | `{', '.join(dpert)}` | {fmt_cell(ag['test_auc'])} | "
            f"{fmt_cell(ag['mia_auc'])} | {fmt_cell(ag['worst_group_tpr_gap'])} | {fmt_cell(ag['worst_group_fpr_gap'])} |"
        )
    else:
        print("| DP-SGD ε=5 + output perturb | — | *excluded: fewer than 5 matched seeds or not found* | | | |")


if __name__ == "__main__":
    main()
