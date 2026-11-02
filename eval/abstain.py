"""
Usage: python -m eval.abstain <val_results.json> <test_results.json> [--target 0.95]

Abstaining: answer "unknown" when the retrieval evidence is weak, instead of
guessing. Confidence of the nearest-neighbour vote is

    confidence = best similarity x share of the (similarity-weighted) vote

The threshold is chosen on the VALIDATION results only: the lowest threshold
whose precision (accuracy on the lines it does answer) reaches --target.
It is then applied once to the TEST results. Reports a precision-vs-coverage
table for both sets, what happened to each category on test (answered right,
answered wrong, abstained), and writes an SVG of both curves next to the test
results.

The target (0.95) was fixed before looking at the test set.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    for r in rows:
        r["conf"] = round(r["top_score"] * r["vote_share"], 4)
    return rows


def at(rows: list[dict], t: float) -> tuple[float, float, int]:
    answered = [r for r in rows if r["conf"] >= t]
    if not answered:
        return 1.0, 0.0, 0
    correct = sum(r["pred"]["knn_vote"] == r["true_label"] for r in answered)
    return correct / len(answered), len(answered) / len(rows), len(answered)


def choose(rows: list[dict], target: float) -> float:
    for t in sorted({r["conf"] for r in rows}):
        if at(rows, t)[0] >= target:
            return t
    return 1.0


def curve(rows: list[dict]) -> list[tuple[float, float, float]]:
    return [(t, *at(rows, t)[:2]) for t in sorted({0.0} | {r["conf"] for r in rows})]


def svg(val_curve, test_curve, chosen_point, out: Path) -> None:
    W, H, L, B, T, R = 560, 360, 60, 50, 20, 20
    x = lambda cov: L + cov * (W - L - R)
    y = lambda prec, lo=0.5: H - B - (max(prec, lo) - lo) / (1 - lo) * (H - B - T)

    def line(points, style):
        pts = " ".join(f"{x(c):.1f},{y(p):.1f}" for _, p, c in points)
        return f'<polyline fill="none" stroke-width="2" {style} points="{pts}"/>'

    grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ddd"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11">{v:.1f}</text>'
                   for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    xt = "".join(f'<text x="{x(v):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11">{v:.0%}</text>'
                 for v in (0, 0.25, 0.5, 0.75, 1.0))
    cx, cy = x(chosen_point[1]), y(chosen_point[0])
    body = (f'<rect width="{W}" height="{H}" fill="#fff"/>{grid}{xt}'
            f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H - B}" stroke="#333"/>'
            f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="#333"/>'
            + line(val_curve, 'stroke="#888" stroke-dasharray="5,4"') + line(test_curve, 'stroke="#c0392b"')
            + f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#c0392b"/>'
            f'<text x="{cx - 8:.1f}" y="{cy - 10:.1f}" text-anchor="end" font-size="11">chosen on validation</text>'
            f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle" font-size="12">coverage (share of lines answered)</text>'
            f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-size="12">precision on answered lines</text>'
            f'<line x1="{W - 190}" x2="{W - 165}" y1="{T + 14}" y2="{T + 14}" stroke="#888" stroke-dasharray="5,4" stroke-width="2"/>'
            f'<text x="{W - 160}" y="{T + 18}" font-size="11">validation</text>'
            f'<line x1="{W - 190}" x2="{W - 165}" y1="{T + 32}" y2="{T + 32}" stroke="#c0392b" stroke-width="2"/>'
            f'<text x="{W - 160}" y="{T + 36}" font-size="11">test (time-ordered)</text>')
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                   f'font-family="Segoe UI, Arial, sans-serif">{body}</svg>\n', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val")
    ap.add_argument("test")
    ap.add_argument("--target", type=float, default=0.95)
    args = ap.parse_args()
    val, test = load(args.val), load(args.test)

    t = choose(val, args.target)
    vp, vc, vn = at(val, t)
    tp, tc, tn = at(test, t)
    bp, _, _ = at(test, 0.0)
    md = ["# Abstaining on weak evidence", "",
          f"Confidence = best similarity x vote share. Target precision {args.target}, fixed before looking at test.", "",
          f"Threshold chosen on validation ({len(val)} lines): **{t:.4f}**", "",
          "| set | precision on answered | coverage | answered | precision if always answering |",
          "|---|---|---|---|---|",
          f"| validation | {vp:.3f} | {vc:.3f} | {vn}/{len(val)} | {at(val, 0.0)[0]:.3f} |",
          f"| **test** | **{tp:.3f}** | **{tc:.3f}** | {tn}/{len(test)} | {bp:.3f} |", ""]

    md += ["## Test, per category", "",
           "| category | lines | answered right | answered wrong | abstained |", "|---|---|---|---|---|"]
    outcome = Counter()
    for r in test:
        o = "abstained" if r["conf"] < t else "right" if r["pred"]["knn_vote"] == r["true_label"] else "wrong"
        outcome[(r["true_label"], o)] += 1
    for cat in sorted({r["true_label"] for r in test}):
        n = sum(outcome[(cat, o)] for o in ("right", "wrong", "abstained"))
        md.append(f"| {cat} | {n} | {outcome[(cat, 'right')]} | {outcome[(cat, 'wrong')]} | {outcome[(cat, 'abstained')]} |")

    md += ["", "## Precision vs coverage", "", "| threshold | validation precision | validation coverage | test precision | test coverage |",
           "|---|---|---|---|---|"]
    for th in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0):
        a, b = at(val, th), at(test, th)
        md.append(f"| {th:.2f} | {a[0]:.3f} | {a[1]:.3f} | {b[0]:.3f} | {b[1]:.3f} |")

    out = Path(args.test).with_name(Path(args.test).stem + "-abstain")
    svg(curve(val), curve(test), (tp, tc), out.with_suffix(".svg"))
    md += ["", f"Curve: `{out.with_suffix('.svg').name}`"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
