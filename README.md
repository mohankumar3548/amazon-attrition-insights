# Amazon Attrition Lens

AI solution 05 of 14 · HR · built for **Amazon (Fulfillment Operations)**

Prepared by **K Mohan Kumar**.

**Live website:** https://mohankumar3548.github.io/amazon-attrition-insights/

## Problem statement

Employee turnover has increased significantly, creating recruitment costs and operational disruption. Develop an AI system that integrates employee performance, workload, compensation, absenteeism, overtime, engagement and training data to identify organizational factors associated with attrition and recommend evidence-based retention interventions.

## Why Amazon (Fulfillment Operations)

Amazon's fulfillment and delivery workforce has well-documented high turnover, which drives up hiring and training cost and disrupts sites. This tool finds which conditions cause exits and what to change first.

## Project structure

```
amazon-attrition-insights/
├── index.html                 Website entry point (served by GitHub Pages)
├── frontend/
│   ├── css/style.css          Layout, typography, light and dark themes
│   └── js/
│       ├── engine.js          Maths, models, charts, page layout and AI analyst panel
│       └── model.js           Sample data and the algorithm for this problem (browser version)
├── app.py                     Streamlit dashboard (Python version of the website)
├── src/
│   ├── data_generator.py      Builds the sample dataset and saves it to data/
│   ├── model.py               The algorithm in Python (pandas, NumPy, scikit-learn)
│   ├── report.py              Writes results to reports/
│   ├── utils.py               Formatting and forecasting helpers
│   └── meta.json              Problem statement, company and algorithm steps
├── data/                      Sample data as CSV
├── reports/                   Results: results.json, summary.md and CSV tables
├── tests/test_model.py        pytest tests for the data and the model
├── requirements.txt
└── pytest.ini
```

## Algorithm

1. **Integrate HR data**: Performance, workload, pay, absence, overtime, engagement, training
2. **Engineer features**: Overtime level, pay gap to market, months since promotion
3. **Train attrition model**: Logistic regression, 70/30 split
4. **Rank drivers**: Permutation importance on held-out data
5. **Causal check**: Propensity score matching: does heavy overtime cause exits?
6. **Manager hotspots**: Attrition by manager vs company rate
7. **Exit interview themes**: Keyword grouping of exit comments
8. **Interventions**: Simulate policy changes with the model

The same algorithm is implemented twice: in JavaScript (`frontend/js/model.js`) so the website runs in any browser with no server, and in Python (`src/model.py`) for analysis, the Streamlit dashboard and the tests.

## How to run

**Website:** open the live link above, or open `index.html` in a browser.

**Python:**

```bash
pip install -r requirements.txt
python -m src.data_generator     # create the sample data in data/
python -m src.model              # run the algorithm and write reports/
streamlit run app.py             # open the interactive dashboard
pytest                           # run the tests
```

## Data

2,400 associates across 4 job families and 48 managers, 12 months of HR records (sample data). Files: `data/employees.csv`, `data/managers.csv`. All figures are illustrative and are not Amazon (Fulfillment Operations)'s actual numbers.

## Results (default settings, Python run)

| Measure | Value | Note |
|---|---|---|
| Annual attrition | 30.0% | Scaled to 300K associates |
| Cost of turnover | $898.8M | 89,875 exits |
| Overtime effect | +14.6 pp | Causal estimate from matching |
| Exits avoided | 18,050 | With the overtime cap and raise |
| Net saving | $95.2M | $180.5M saved - $85.3M pay cost |

### Recommended actions

1. **Cap routine overtime at 8 hours a week.** Matching shows heavy overtime itself adds +14.6 pp to exit risk. The cap removes 28,044,250 overtime hours a year; cover peaks with flex and seasonal staff.
2. **Give a 2% raise to staff paid over 5% under market.** Pay vs market is a key driver and 'pay' appears in 23% of exit comments.
3. **Coach the 5 hotspot managers.** Their teams lose people well above the 30.0% average. Run skip-level interviews and pair each with a high-retention peer.
4. **Create a visible promotion path.** Risk rises with months since promotion. Publish skill-based steps to Process Assistant after 12 months.
5. **Watch engagement monthly.** Engagement is the strongest early signal. Flag teams whose score drops 0.3 points.

Full output: [`reports/summary.md`](reports/summary.md) and [`reports/results.json`](reports/results.json).

## AI analyst panel

The website includes an "Ask the AI analyst" panel that answers questions about the results through Claude. It works when the page is opened as a Claude artifact on claude.ai. On GitHub Pages the models, charts and controls run in full and the panel shows that it is unavailable.
