# Amazon Attrition Lens: results

Built for Amazon (Fulfillment Operations). Prepared by K Mohan Kumar.

## Key figures

- **Annual attrition:** 30.0% (Scaled to 300K associates)
- **Cost of turnover:** $898.8M (89,875 exits)
- **Overtime effect:** +14.6 pp (Causal estimate from matching)
- **Exits avoided:** 18,050 (With the overtime cap and raise)
- **Net saving:** $95.2M ($180.5M saved - $85.3M pay cost)

## Algorithm, as it ran

1. **Integrate HR data**: 2,400 associates, 30.0% left in 12 months
2. **Engineer features**: 9 features per associate
3. **Train attrition model**: Test AUC 0.758
4. **Rank drivers**: Top driver: Engagement score, then Weekly overtime
5. **Causal check**: Heavy overtime adds +14.6 pp to exit risk after matching (naive gap +25.6 pp)
6. **Manager hotspots**: 5 hotspot managers above 48%
7. **Exit interview themes**: Top exit theme: Overtime / burnout
8. **Interventions**: Projected exits avoided: 18,050 per year

## Recommended actions

1. **Cap routine overtime at 8 hours a week.** Matching shows heavy overtime itself adds +14.6 pp to exit risk. The cap removes 28,044,250 overtime hours a year; cover peaks with flex and seasonal staff. *Impact: 18,050 exits*
2. **Give a 2% raise to staff paid over 5% under market.** Pay vs market is a key driver and 'pay' appears in 23% of exit comments. *Impact: $85.3M cost*
3. **Coach the 5 hotspot managers.** Their teams lose people well above the 30.0% average. Run skip-level interviews and pair each with a high-retention peer. *Impact: Fastest win*
4. **Create a visible promotion path.** Risk rises with months since promotion. Publish skill-based steps to Process Assistant after 12 months. *Impact: Long-term*
5. **Watch engagement monthly.** Engagement is the strongest early signal. Flag teams whose score drops 0.3 points. *Impact: Early warning*

_Generated from sample data. Figures are illustrative._
