r"""Can Locks go down to 70%? Kalshi favourites priced 70-85% with the Lock filters (model agrees, no risk flags,
   non-ITF) and variants, two halves + 30 luck runs. Needs work/locks_df.pkl (pickle of tools.upset_risk.kalshi()).

Result (2026-11-07, 8,097 matches): 1 cell held both halves vs 1.0 per luck run. 70-85% favourites win what they are
priced (75.0/79.5% vs 76.3/76.5% with filters; ROI -4.1% / +1.6%). Only veteran favourites (30+) passed: already the
staked veteran_fav strategy. Locks stay at 85%+.

  .venv\Scripts\python.exe -m tools.locks70
"""
import math, numpy as np, pandas as pd
try:
    D = pd.read_pickle("work/locks_df.pkl")
except FileNotFoundError:
    import contextlib, io
    from tools.upset_risk import kalshi
    with contextlib.redirect_stdout(io.StringIO()):
        D = kalshi()
    D.to_pickle("work/locks_df.pkl")
fa = D.px >= 0.5
D["ask_f"] = np.where(fa, D.ask_a, D.ask_b)
clean = (D.retired_last_f.fillna(0) == 0) & (D.retired_last_d.fillna(0) == 0) & (D.m7_f.fillna(0) < 4) & (D.days_f.fillna(0) < 45) & (D.days_d.fillna(0) < 31)
agree = (D.pm_f - D.pf).abs() < 0.10
tour = ~D.series.str.contains("ITF")
def run(D, y, quiet=False):
    passed = 0
    for lo, hi in ((0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.70, 0.85)):
        band = D.pf.between(lo, hi, inclusive="left") & D.ask_f.between(0.05, 0.95)
        for name, m in (("any", band), ("model agrees, no flags, non-ITF", band & agree & clean & tour),
                        ("model 3+ pts HIGHER, no flags, non-ITF", band & (D.pm_f - D.pf >= 0.03) & (D.pm_f - D.pf < 0.15) & clean & tour),
                        ("model 85%+ (ours says lock), no flags, non-ITF", band & (D.pm_f >= 0.85) & clean & tour),
                        ("veteran fav 30+, no flags, non-ITF", band & (D.age_f >= 30) & clean & tour)):
            out, ok = [], True
            for h in (0, 1):
                g = m & (D.half == h)
                yy = y[g.values]; a = D.ask_f[g].values; pf = D.pf[g].values
                if len(yy) < 40:
                    out.append(f"half {h}: n={len(yy):>4} (few)"); ok = False; continue
                c = a + 0.07 * a * (1 - a); r = np.where(yy == 1, (1 - c) / c, -1.0)
                ok &= r.mean() > 0.01
                out.append(f"half {h}: n={len(yy):>4} won {yy.mean():5.1%} (priced {pf.mean():5.1%}) ROI {r.mean():+5.1%} ± {r.std(ddof=1)/math.sqrt(len(r)):3.1%}")
            passed += ok
            if not quiet:
                print(f"  {int(lo*100)}-{int(hi*100)}%  {name:<46} " + "   ".join(out) + ("  <-- both halves" if ok else ""))
    return passed
print(len(D), "Kalshi matches")
p = run(D, D.fw.values)
rng = np.random.default_rng(0)
luck = [run(D, (rng.random(len(D)) < D.pf.values).astype(int), quiet=True) for _ in range(30)]
print(f"\n{p} cells held both halves; luck runs average {np.mean(luck):.1f} (max {max(luck)})")
