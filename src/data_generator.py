"""Sample data for Amazon fulfillment: 2,400 associates, 48 managers, 12 months of HR records."""
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ["Fulfillment Center", "Sort Center", "Delivery Station", "Customer Service"]
EXIT_TEXT = {"ot": ["mandatory extra time every week", "too many hours, no life outside work", "burned out from peak overtime"],
             "pay": ["found a job paying more nearby", "pay did not keep up", "competitor offered a dollar more"],
             "mgr": ["manager never listened", "felt disrespected by my supervisor", "unfair treatment by lead"],
             "grow": ["no path to promotion", "same role for two years"], "other": ["moved away", "went back to school", "family reasons"]}


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    mgr = pd.DataFrame({"manager_id": [f"M{100 + i}" for i in range(48)], "job_family": [FAMILIES[i % 4] for i in range(48)],
                        "weak_manager": (rng.random(48) < 0.15).astype(int)})
    rows = []
    for i in range(2400):
        m = mgr.iloc[rng.integers(48)]; night = int(rng.random() < 0.35)
        ot = max(0.0, rng.normal(3 if m.job_family == "Customer Service" else 7 + 4 * night, 4.5))
        gap = rng.normal(-0.03, 0.06)
        eng = float(np.clip(rng.normal(3.6 - 0.05 * ot - 0.6 * m.weak_manager, 0.7), 1, 5))
        absd = max(0, round(rng.normal(4 + 0.25 * ot, 3))); promo = int(rng.integers(2, 37)); train = max(0.0, rng.normal(18, 9))
        perf = float(np.clip(rng.normal(3.2, 0.7), 1, 5)); tenure = int(rng.integers(1, 61))
        z = -2.2 + 0.11 * ot - 9 * gap - 0.65 * (eng - 3.4) + 0.06 * absd + 0.02 * promo - 0.025 * train + 0.9 * m.weak_manager - 0.02 * tenure + 0.35 * night
        left = int(rng.random() < 1 / (1 + np.exp(-z)))
        why = ""
        if left:
            key = "ot" if ot > 10 else "pay" if gap < -0.06 else "mgr" if m.weak_manager else "grow" if promo > 24 else ("other" if rng.random() < 0.5 else "ot")
            why = rng.choice(EXIT_TEXT[key])
        rows.append({"employee_id": f"E{20000 + i}", "job_family": m.job_family, "manager_id": m.manager_id, "night_shift": night,
                     "weekly_overtime_hrs": round(ot, 1), "pay_vs_market": round(gap, 3), "engagement_score": round(eng, 2), "absence_days": absd,
                     "months_since_promotion": promo, "training_hours": round(train, 1), "performance_rating": round(perf, 1), "tenure_months": tenure,
                     "left_company": left, "exit_comment": why})
    return {"employees": pd.DataFrame(rows), "managers": mgr}


# ---------------------------------------------------------------------------
# Save and load helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save(data: dict, folder: Path = DATA_DIR) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def load_or_generate(folder: Path = DATA_DIR, seed: int = 42) -> dict:
    """Read the CSVs in data/ if they exist, otherwise generate fresh sample data."""
    folder = Path(folder)
    names = list(generate(seed).keys())
    if all((folder / f"{n}.csv").exists() for n in names):
        return {n: pd.read_csv(folder / f"{n}.csv") for n in names}
    return generate(seed)


if __name__ == "__main__":
    d = generate(42)
    save(d)
    for n, df in d.items():
        print(f"data/{n}.csv  {len(df):,} rows x {df.shape[1]} columns")
