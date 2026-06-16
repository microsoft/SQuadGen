#!/usr/bin/env python3
"""Compare current inference quad-quality results to paper_score.json.

For each entry under results2/infer_sqdiffuse/test_data_8192_smooth/data/<name>/gen_000/
extracted_quad_quadquality.json, compute min(Fratio, Eratio) across all submeshes
and compare to the value recorded in debug_script/paper_score.json.

Rows are printed sorted by diff = cur - paper (descending), so the entries
where the current run beats the paper the most appear first.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_JSON = os.path.join(REPO_ROOT, "debug_script", "paper_score.json")
DATA_DIR = os.path.join(
    REPO_ROOT, "results2", "infer_sqdiffuse", "test_data_8192_smooth", "data"
)


def load_current(data_dir):
    results = {}
    if not os.path.isdir(data_dir):
        print(f"[error] data dir not found: {data_dir}", file=sys.stderr)
        return results
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name, "gen_000", "extracted_quad_quadquality.json")
        if not os.path.exists(p):
            results[name] = None
            continue
        try:
            arr = json.load(open(p))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}", file=sys.stderr)
            results[name] = None
            continue
        vals = [min(e["Fratio"], e["Eratio"]) for e in arr]
        results[name] = min(vals) if vals else None
    return results


def main():
    paper = json.load(open(PAPER_JSON))
    current = load_current(DATA_DIR)

    all_keys = set(paper.keys()) | set(current.keys())
    rows = []
    for k in all_keys:
        cur = current.get(k)
        pap_entry = paper.get(k)
        pap = pap_entry.get("min_Fratio_Eratio") if isinstance(pap_entry, dict) else None
        if cur is not None and pap is not None:
            diff = cur - pap
        else:
            diff = None
        rows.append((k, cur, pap, diff))

    # Sort by diff descending; entries with missing diff go to the bottom.
    rows.sort(key=lambda r: (r[3] is None, -(r[3] if r[3] is not None else 0)))

    name_w = max(len(r[0]) for r in rows) if rows else 4
    header = f"{'name':<{name_w}} {'cur':>8} {'paper':>8} {'diff':>9} {'better':>7}"
    print(header)
    print("-" * len(header))

    better = 0
    worse = 0
    for k, cur, pap, diff in rows:
        cur_s = f"{cur:.4f}" if cur is not None else "-"
        pap_s = f"{pap:.4f}" if pap is not None else "-"
        diff_s = f"{diff:+.4f}" if diff is not None else "-"
        if diff is None:
            flag = "-"
        elif diff > 0:
            flag = "YES"
            better += 1
        elif diff < 0:
            flag = "no"
            worse += 1
        else:
            flag = "eq"
        print(f"{k:<{name_w}} {cur_s:>8} {pap_s:>8} {diff_s:>9} {flag:>7}")

    print("-" * len(header))
    total = sum(1 for _, _, _, d in rows if d is not None)
    print(f"better than paper: {better}/{total}    worse: {worse}/{total}")


if __name__ == "__main__":
    main()
