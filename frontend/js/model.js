/* Amazon Attrition Lens: sample data generator and algorithm for Amazon (Fulfillment Operations). */
const TOOL = {
  num: 5, area: "HR", product: "Attrition Root-Cause Intelligence", seed: 75,
  title: "Amazon Attrition Lens", company: "Amazon (Fulfillment Operations)",
  why: "Amazon's fulfillment and delivery workforce has well-documented high turnover, which drives up hiring and training cost and disrupts sites. This tool finds which conditions cause exits and what to change first.",
  statement: "Employee turnover has increased significantly, creating recruitment costs and operational disruption. Develop an AI system that integrates employee performance, workload, compensation, absenteeism, overtime, engagement and training data to identify organizational factors associated with attrition and recommend evidence-based retention interventions.",
  dataNote: "2,400 associates across 4 job families and 48 managers, 12 months of HR records (sample data).",
  steps: [
    ["Integrate HR data", "Performance, workload, pay, absence, overtime, engagement, training"],
    ["Engineer features", "Overtime level, pay gap to market, months since promotion"],
    ["Train attrition model", "Logistic regression, 70/30 split"],
    ["Rank drivers", "Permutation importance on held-out data"],
    ["Causal check", "Propensity score matching: does heavy overtime cause exits?"],
    ["Manager hotspots", "Attrition by manager vs company rate"],
    ["Exit interview themes", "Keyword grouping of exit comments"],
    ["Interventions", "Simulate policy changes with the model"]
  ],
  params: [
    { id: "ot", label: "Weekly overtime cap", min: 0, max: 20, step: 1, value: 8, fmt: v => v + " hrs" },
    { id: "pay", label: "Raise for staff >5% under market", min: 0, max: 10, step: 1, value: 2, fmt: v => "+" + v + "%" },
    { id: "cost", label: "Cost to replace one associate", min: 3000, max: 15000, step: 500, value: 10000, fmt: v => F.money(v) }
  ],
  questions: ["Is overtime really causing people to quit?", "Which managers need coaching first?", "Is the pay correction worth its cost?", "Write a retention plan for the night shift"],
  generate(r) {
    const fam = ["Fulfillment Center", "Sort Center", "Delivery Station", "Customer Service"];
    const mgrs = S.range(48).map(i => ({ id: "M" + (100 + i), fam: fam[i % 4], bad: r.chance(0.15) ? 1 : 0 }));
    const exitT = { ot: ["mandatory extra time every week", "too many hours, no life outside work", "burned out from peak overtime"], pay: ["found a job paying more nearby", "pay did not keep up", "competitor offered a dollar more"], mgr: ["manager never listened", "felt disrespected by my supervisor", "unfair treatment by lead"], grow: ["no path to promotion", "same role for two years"], other: ["moved away", "went back to school", "family reasons"] };
    const E = [];
    for (let i = 0; i < 2400; i++) {
      const m = r.pick(mgrs), night = r.chance(0.35) ? 1 : 0;
      const ot = Math.max(0, r.n(m.fam === "Customer Service" ? 3 : 7 + 4 * night, 4.5)), gap = r.n(-0.03, 0.06), eng = S.clamp(r.n(3.6 - 0.05 * ot - 0.6 * m.bad, 0.7), 1, 5);
      const abs = Math.max(0, Math.round(r.n(4 + 0.25 * ot, 3))), promo = r.int(2, 36), train = Math.max(0, r.n(18, 9)), perf = S.clamp(r.n(3.2, 0.7), 1, 5), tenure = r.int(1, 60);
      const z = -2.2 + 0.11 * ot - 9 * gap - 0.65 * (eng - 3.4) + 0.06 * abs + 0.02 * promo - 0.025 * train + 0.9 * m.bad - 0.02 * tenure + 0.35 * night;
      const left = r() < S.sigmoid(z) ? 1 : 0;
      const why = !left ? "" : r.pick(ot > 10 ? exitT.ot : gap < -0.06 ? exitT.pay : m.bad ? exitT.mgr : promo > 24 ? exitT.grow : r.chance(0.5) ? exitT.other : exitT.ot);
      E.push({ id: "E" + (20000 + i), fam: m.fam, mgr: m.id, night, ot, gap, eng, abs, promo, train, perf, tenure, left, why });
    }
    return { E, mgrs };
  },
  analyze(D, P, r) {
    const E = D.E, fn = ["Weekly overtime", "Pay vs market", "Engagement score", "Absence days", "Months since promotion", "Training hours", "Performance rating", "Tenure (months)", "Night shift"];
    const x = e => [e.ot, e.gap, e.eng, e.abs, e.promo, e.train, e.perf, e.tenure, e.night];
    const idx = S.range(E.length); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1));[idx[i], idx[j]] = [idx[j], idx[i]]; }
    const tr = idx.slice(0, 1680), te = idx.slice(1680);
    const m = M.logit(tr.map(i => x(E[i])), tr.map(i => E[i].left), { iters: 350, lr: 0.5 });
    const Xte = te.map(i => x(E[i])), yte = te.map(i => E[i].left);
    const auc = M.auc(Xte.map(m.predict), yte);
    const imp = M.permImp(m.predict, Xte, yte, (p, y) => M.auc(p, y), r, fn).map((v, j) => ({ ...v, w: m.w[j] })).sort((a, b) => b.drop - a.drop);
    const rate = S.mean(E.map(e => e.left));
    // propensity score matching: treatment heavy OT (>10)
    const treat = e => e.ot > 10 ? 1 : 0; const conf = e => [e.eng, e.gap, e.promo, e.train, e.tenure, e.night, e.perf, e.fam === "Customer Service" ? 1 : 0];
    const ps = M.logit(E.map(conf), E.map(treat), { iters: 200, lr: 0.5 }); const sc = E.map(e => ps.predict(conf(e)));
    const T1 = E.map((e, i) => i).filter(i => treat(E[i])), T0 = E.map((e, i) => i).filter(i => !treat(E[i])).sort((a, b) => sc[a] - sc[b]);
    const s0 = T0.map(i => sc[i]); const nearest = v => { let lo = 0, hi = s0.length - 1; while (lo < hi) { const md = (lo + hi) >> 1; if (s0[md] < v) lo = md + 1; else hi = md; } return T0[lo]; };
    const att = S.mean(T1.map(i => E[i].left - E[nearest(sc[i])].left));
    const naive = S.mean(T1.map(i => E[i].left)) - S.mean(T0.map(i => E[i].left));
    // managers
    const mg = [...S.groupBy(E, e => e.mgr)].map(([id, v]) => ({ id, fam: v[0].fam, n: v.length, r: S.mean(v.map(e => e.left)), eng: S.mean(v.map(e => e.eng)) })).sort((a, b) => b.r - a.r);
    const sdM = S.sd(mg.map(g => g.r)), hot = mg.filter(g => g.r > rate + 1.5 * sdM);
    // exit themes
    const th = { "Overtime / burnout": ["hours", "overtime", "extra time", "burned"], "Pay": ["pay", "paying", "dollar"], "Manager": ["manager", "supervisor", "lead"], "Growth": ["promotion", "same role"], "Personal": ["moved", "school", "family"] };
    const lefts = E.filter(e => e.left); const themes = Object.entries(th).map(([k, ws]) => ({ label: k, value: lefts.filter(e => ws.some(w => e.why.includes(w))).length / lefts.length })).sort((a, b) => b.value - a.value);
    // simulation
    const simP = e => m.predict(x({ ...e, ot: Math.min(e.ot, P.ot), gap: e.gap < -0.05 ? e.gap + P.pay / 100 : e.gap, eng: e.eng + 0.05 * Math.max(0, e.ot - P.ot) }));
    const base = S.sum(E.map(e => m.predict(x(e)))), after = S.sum(E.map(simP));
    const SC = 125; // sample → 300k associates
    const savedExits = (base - after) * SC, savedCost = savedExits * P.cost;
    const payCost = S.sum(E.filter(e => e.gap < -0.05).map(e => P.pay / 100 * 38000)) * SC;
    const otCost = 0; // overtime cap may need extra hires; reported as hours
    const hrsCut = S.sum(E.map(e => Math.max(0, e.ot - P.ot))) * 52 * SC;
    const fams = [...S.groupBy(E, e => e.fam)].map(([k, v]) => ({ label: k, value: S.mean(v.map(e => e.left)) })).sort((a, b) => b.value - a.value);
    return {
      steps: [`${F.num(E.length)} associates, ${F.pct(rate)} left in 12 months`, `${fn.length} features per associate`, `Test AUC ${auc.toFixed(3)}`, `Top driver: ${imp[0].name}, then ${imp[1].name}`, `Heavy overtime raises exit risk by ${F.pp(att)} after matching (naive gap ${F.pp(naive)})`, `${hot.length} hotspot managers above ${F.pct(rate + 1.5 * sdM, 0)}`, `Top exit theme: ${themes[0].label}`, `Projected exits avoided: ${F.num(savedExits)} per year`],
      kpis: [
        { label: "Annual attrition", value: F.pct(rate), note: "Sample of 2,400; scaled to 300K associates", tone: "bad" },
        { label: "Cost of turnover", value: F.money(rate * E.length * SC * P.cost), note: `${F.num(rate * E.length * SC)} exits × ${F.money(P.cost)}` },
        { label: "Overtime effect", value: F.pp(att), note: "Causal estimate from matching", tone: "warn" },
        { label: "Exits avoided", value: F.num(savedExits), note: "With the overtime cap and pay fix", tone: "good" },
        { label: "Net saving", value: F.money(savedCost - payCost), note: `${F.money(savedCost)} saved − ${F.money(payCost)} pay cost`, tone: savedCost > payCost ? "good" : "bad" }
      ],
      sections: [
        { pair: [
          { title: "What drives exits", note: "Drop in model accuracy when each factor is scrambled. Red factors raise attrition when they go up.", chart: C.bar(imp.map(v => ({ label: v.name, value: Math.max(0, v.drop), tone: (v.w > 0) !== (v.name === "Pay vs market" || v.name === "Engagement score" || v.name === "Training hours" || v.name === "Tenure (months)" || v.name === "Performance rating") ? "bad" : "" })), { fmt: v => v.toFixed(3) }) },
          { title: "Correlation vs cause", note: "Attrition gap between heavy-overtime staff (>10 hrs/week) and everyone else, before and after matching on engagement, pay, tenure and shift.", chart: C.bar([{ label: "Naive comparison", value: naive, tone: "mut" }, { label: "After matching (causal)", value: att, tone: "bad" }], { fmt: v => F.pp(v), labelW: 170 }) + `<p class="small">Matched ${F.num(T1.length)} heavy-overtime associates to their closest look-alike on the propensity score.</p>` }
        ] },
        { pair: [
          { title: "Manager hotspots", note: "Top 10 managers by team attrition. Red bars are more than 1.5 standard deviations above the company rate (dashed).", chart: C.bar(mg.slice(0, 10).map(g => ({ label: `${g.id} · ${g.fam}`, value: g.r, tone: hot.includes(g) ? "bad" : "" })), { fmt: v => F.pct(v, 0), ref: rate, labelW: 190 }) },
          { title: "What leavers said", note: "Share of exit comments that mention each theme.", chart: C.bar(themes.map(t => ({ label: t.label, value: t.value, tone: "c3" })), { fmt: v => F.pct(v, 0) }) + `<div style="margin-top:14px">${C.bar(fams.map(f => ({ label: f.label, value: f.value })), { fmt: v => F.pct(v, 0) })}</div><p class="small">Attrition by job family.</p>` }
        ] },
        { title: "Hotspot managers", html: T.table(["Manager", "Job family", "Team size", "Attrition", "Team engagement", "Action"], hot.map(g => [g.id, g.fam, g.n, T.pill(F.pct(g.r, 0), "bad"), g.eng.toFixed(2) + " / 5", "Coaching + skip-level interviews"]), { right: [2, 3, 4] }) }
      ],
      recs: [
        { title: `Cap routine overtime at ${P.ot} hours a week`, body: `Matching shows heavy overtime itself adds ${F.pp(att)} to exit risk. The cap removes ${F.num(hrsCut)} overtime hours a year; cover peaks with flex and seasonal staff instead.`, impact: F.num(savedExits) + " exits", tone: "good" },
        { title: `Give a ${P.pay}% raise to staff paid over 5% under market`, body: `Pay vs market is a top-three driver and "pay" shows up in ${F.pct(themes.find(t => t.label === "Pay").value, 0)} of exit comments. Target sites where local competitors pay more.`, impact: F.money(payCost) + " cost", tone: "" },
        { title: `Coach the ${hot.length} hotspot managers`, body: `Their teams lose people well above the ${F.pct(rate)} average with lower engagement. Run skip-level interviews and pair each with a high-retention peer.`, impact: "Fastest win", tone: "good" },
        { title: "Create a visible promotion path", body: "Months since promotion raises risk steadily. Publish skill-based steps to Process Assistant and Area Manager roles after 12 months.", impact: "Long-term", tone: "" },
        { title: "Watch engagement scores monthly", body: "Engagement is the strongest early signal. Flag any team whose average drops by 0.3 points so HR can act before exits start.", impact: "Early warning", tone: "" }
      ],
      context: { attrition_rate: rate, auc, drivers: imp.map(v => ({ factor: v.name, importance: +v.drop.toFixed(3), coefficient: +v.w.toFixed(2) })), overtime_causal_effect_pp: +(att * 100).toFixed(1), overtime_naive_gap_pp: +(naive * 100).toFixed(1), hotspot_managers: hot.map(g => ({ id: g.id, family: g.fam, attrition: +g.r.toFixed(2), engagement: +g.eng.toFixed(2) })), exit_themes: themes.map(t => ({ theme: t.label, share: +t.value.toFixed(2) })), attrition_by_family: fams.map(f => ({ family: f.label, rate: +f.value.toFixed(3) })), scenario: { overtime_cap_hours: P.ot, pay_correction_pct: P.pay, replacement_cost: P.cost, exits_avoided: Math.round(savedExits), savings: Math.round(savedCost), pay_cost: Math.round(payCost) } }
    };
  }
};
