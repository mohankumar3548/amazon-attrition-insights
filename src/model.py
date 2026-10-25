"""Attrition drivers, causal overtime test, manager hotspots and retention scenarios for Amazon."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import bar, kpi, money, num, pct, pp, rec, table

META = json.loads(Path(__file__).with_name("meta.json").read_text())
PARAMS = [
    {"name": "overtime_cap", "label": "Weekly overtime cap (hrs)", "min": 0, "max": 20, "step": 1, "default": 8},
    {"name": "raise_pct", "label": "Raise for staff >5% under market (%)", "min": 0, "max": 10, "step": 1, "default": 2},
    {"name": "replacement_cost", "label": "Cost to replace one associate ($)", "min": 3000, "max": 15000, "step": 500, "default": 10000},
]
SCALE = 125  # 2,400 sample -> 300,000 associates
FEAT = {"weekly_overtime_hrs": "Weekly overtime", "pay_vs_market": "Pay vs market", "engagement_score": "Engagement score", "absence_days": "Absence days",
        "months_since_promotion": "Months since promotion", "training_hours": "Training hours", "performance_rating": "Performance rating",
        "tenure_months": "Tenure (months)", "night_shift": "Night shift"}
THEMES = {"Overtime / burnout": ["hours", "overtime", "extra time", "burned"], "Pay": ["pay", "paying", "dollar"], "Manager": ["manager", "supervisor", "lead"],
          "Growth": ["promotion", "same role"], "Personal": ["moved", "school", "family"]}


def run(data, overtime_cap=8, raise_pct=2, replacement_cost=10000):
    E = data["employees"].copy()
    E["exit_comment"] = E["exit_comment"].fillna("")
    cols = list(FEAT); X, y = E[cols], E["left_company"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    pi = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
    coef = model[-1].coef_[0]
    imp = pd.DataFrame({"factor": [FEAT[c] for c in cols], "importance": pi.importances_mean.round(4), "coefficient": coef.round(2)}).sort_values("importance", ascending=False)
    rate = y.mean()

    # propensity score matching: treatment = heavy overtime (>10 hrs/week)
    E["heavy_ot"] = (E.weekly_overtime_hrs > 10).astype(int)
    conf = ["engagement_score", "pay_vs_market", "months_since_promotion", "training_hours", "tenure_months", "night_shift", "performance_rating"]
    Ec = E.assign(cs=(E.job_family == "Customer Service").astype(int))
    ps = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Ec[conf + ["cs"]], E.heavy_ot).predict_proba(Ec[conf + ["cs"]])[:, 1]
    t, c = E.heavy_ot == 1, E.heavy_ot == 0
    nn = NearestNeighbors(n_neighbors=1).fit(ps[c.values].reshape(-1, 1))
    _, idx = nn.kneighbors(ps[t.values].reshape(-1, 1))
    att = E.left_company[t].mean() - E.left_company[c].values[idx[:, 0]].mean()
    naive = E.left_company[t].mean() - E.left_company[c].mean()

    mg = E.groupby(["manager_id", "job_family"]).agg(team_size=("employee_id", "count"), attrition=("left_company", "mean"), engagement=("engagement_score", "mean")).reset_index().sort_values("attrition", ascending=False)
    cut = rate + 1.5 * mg.attrition.std(); hot = mg[mg.attrition > cut].round(3)
    leavers = E[E.left_company == 1]
    themes = pd.DataFrame([{"theme": k, "share": round(leavers.exit_comment.apply(lambda s: any(w in s for w in ws)).mean(), 3)} for k, ws in THEMES.items()]).sort_values("share", ascending=False)
    fam = E.groupby("job_family").left_company.mean().round(3).rename("attrition").reset_index().sort_values("attrition", ascending=False)

    sim = X.copy()
    excess = np.maximum(0, sim.weekly_overtime_hrs - overtime_cap)
    sim["engagement_score"] = sim.engagement_score + 0.05 * excess
    sim["weekly_overtime_hrs"] = np.minimum(sim.weekly_overtime_hrs, overtime_cap)
    under = sim.pay_vs_market < -0.05
    sim.loc[under, "pay_vs_market"] += raise_pct / 100
    base_p, new_p = model.predict_proba(X)[:, 1].sum(), model.predict_proba(sim)[:, 1].sum()
    avoided = (base_p - new_p) * SCALE
    saving = avoided * replacement_cost
    pay_cost = under.sum() * raise_pct / 100 * 38000 * SCALE
    hours_cut = excess.sum() * 52 * SCALE
    return {
        "steps": [f"{len(E):,} associates, {pct(rate)} left in 12 months", f"{len(cols)} features per associate", f"Test AUC {auc:.3f}",
                  f"Top driver: {imp.iloc[0].factor}, then {imp.iloc[1].factor}", f"Heavy overtime adds {pp(att)} to exit risk after matching (naive gap {pp(naive)})",
                  f"{len(hot)} hotspot managers above {pct(cut, 0)}", f"Top exit theme: {themes.iloc[0].theme}", f"Projected exits avoided: {num(avoided)} per year"],
        "kpis": [kpi("Annual attrition", pct(rate), "Scaled to 300K associates"), kpi("Cost of turnover", money(rate * len(E) * SCALE * replacement_cost), f"{num(rate * len(E) * SCALE)} exits"),
                 kpi("Overtime effect", pp(att), "Causal estimate from matching"), kpi("Exits avoided", num(avoided), "With the overtime cap and raise"),
                 kpi("Net saving", money(saving - pay_cost), f"{money(saving)} saved - {money(pay_cost)} pay cost")],
        "charts": [bar("What drives exits", imp, "factor", "importance", "Drop in model AUC when each factor is shuffled."),
                   bar("Correlation vs cause (attrition gap, heavy overtime)", pd.DataFrame({"comparison": ["Naive comparison", "After matching (causal)"], "gap": [round(naive, 3), round(att, 3)]}), "comparison", "gap"),
                   bar("Top 10 manager teams by attrition", mg.head(10).assign(team=lambda d: d.manager_id + " " + d.job_family), "team", "attrition"),
                   bar("What leavers said", themes, "theme", "share"), bar("Attrition by job family", fam, "job_family", "attrition")],
        "tables": [table("Hotspot managers", hot), table("Driver model", imp)],
        "recs": [rec(f"Cap routine overtime at {overtime_cap} hours a week", f"Matching shows heavy overtime itself adds {pp(att)} to exit risk. The cap removes {num(hours_cut)} overtime hours a year; cover peaks with flex and seasonal staff.", f"{num(avoided)} exits"),
                 rec(f"Give a {raise_pct}% raise to staff paid over 5% under market", f"Pay vs market is a key driver and 'pay' appears in {pct(themes.set_index('theme').loc['Pay', 'share'], 0)} of exit comments.", money(pay_cost) + " cost"),
                 rec(f"Coach the {len(hot)} hotspot managers", f"Their teams lose people well above the {pct(rate)} average. Run skip-level interviews and pair each with a high-retention peer.", "Fastest win"),
                 rec("Create a visible promotion path", "Risk rises with months since promotion. Publish skill-based steps to Process Assistant after 12 months.", "Long-term"),
                 rec("Watch engagement monthly", "Engagement is the strongest early signal. Flag teams whose score drops 0.3 points.", "Early warning")],
    }


if __name__ == "__main__":
    from src.data_generator import load_or_generate
    from src.report import write_reports

    out = run(load_or_generate(), **{p["name"]: p["default"] for p in PARAMS})
    for k in out["kpis"]:
        print(f"{k['label']}: {k['value']}  {k['note']}")
    print("Reports written to", write_reports(out, META, Path(__file__).resolve().parent.parent / "reports"))
