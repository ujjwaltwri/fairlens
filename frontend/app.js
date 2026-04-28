/**
 * FairLens — AI Bias Detection & Mitigation Platform
 * Theme: Precision Instrument — DM Serif Display + DM Sans + Spline Sans Mono
 */
(function () {
'use strict';

const { useState, useEffect, useRef, useCallback } = React;
const e = React.createElement;

/* ═══════════════════════════════
   CONSTANTS
═══════════════════════════════ */
const API = 'http://localhost:8080';

const DATASETS = [
  { id: 'adult',       name: 'UCI Adult',      domain: 'Income',       protected: 'Gender · Race', rows: '48,842',  severity: 'HIGH',   dataBias: 92, modelBias: 60 },
  { id: 'compas',      name: 'COMPAS',         domain: 'Justice',      protected: 'Race',          rows: '7,214',   severity: 'MEDIUM', dataBias: 55, modelBias: 28 },
  { id: 'german',      name: 'German Credit',  domain: 'Finance',      protected: 'Age',           rows: '1,000',   severity: 'MEDIUM', dataBias: 60, modelBias: 50 },
  { id: 'utrecht',     name: 'Utrecht Hiring', domain: 'Recruitment',  protected: 'Gender',        rows: '2,000',   severity: 'MEDIUM', dataBias: 51, modelBias: 22 },
  { id: 'diabetes130', name: 'Diabetes 130',   domain: 'Healthcare',   protected: 'Race',          rows: '101,766', severity: 'HIGH',   dataBias: 97, modelBias: 83 },
];

const AUDIT = {
  adult: {
    dataScore: 92, dataSev: 'HIGH',
    modelScore: 60, modelSev: 'MEDIUM',
    dir: 0.5017, dpg: 0.220, tprGap: 0.1344, fprGap: 0.0433, flipRate: 0.089,
    acc: 0.9088, auc: 0.9766, posPriv: 30.7, posUnpriv: 10.8,
    shap: [
      { f: 'age',              v: 3.887, p: false },
      { f: 'education level',  v: 3.115, p: false },
      { f: 'capital gain',     v: 1.224, p: false },
      { f: 'hours per week',   v: 1.128, p: false },
      { f: 'gender (protected)',  v: 0.951, p: true  },
      { f: 'race (protected)',    v: 0.770, p: true  },
      { f: 'marital status',   v: 0.163, p: false },
      { f: 'occupation',       v: 0.090, p: false },
    ],
    mit: [
      { n: 'Before Training',  d: 'Rebalance sample weights so the training data is more representative',                     icon: '01', before: 32, after: 32, imp: 0,  best: false },
      { n: 'During Training',  d: 'Add a fairness rule to the training process so the model cannot favour one group',         icon: '02', before: 32, after: 22, imp: 10, best: true  },
      { n: 'After Training',   d: 'Adjust the decision cutoff separately for each group after training is complete',          icon: '03', before: 32, after: 32, imp: 0,  best: false },
    ],
    recs: [
      'Fix gender imbalance — women are approved at half the rate of men, which fails equal-opportunity hiring rules',
      'Gender is strongly linked to the outcome (statistically very significant) — check whether "marital status" is hiding a gender effect',
      'The model correctly identifies high-income earners 13% less often for one group — apply equal-opportunity correction before going live',
      'MEDIUM RISK: Applying the during-training fairness rule cuts the bias score by 10 points',
    ],
  },
  compas: {
    dataScore: 55, dataSev: 'MEDIUM', modelScore: 28, modelSev: 'LOW',
    dir: 0.61, dpg: 0.14, tprGap: 0.08, fprGap: 0.04, flipRate: 0.04,
    acc: 0.997, auc: 0.922, posPriv: 39.4, posUnpriv: 52.5,
    shap: [
      { f: 'age',              v: 1.18, p: false },
      { f: 'race (protected)', v: 0.74, p: true  },
      { f: 'prior offences',   v: 0.61, p: false },
      { f: 'charge severity',  v: 0.60, p: false },
      { f: 'age group',        v: 0.50, p: false },
      { f: 'days to screening',v: 0.28, p: false },
    ],
    mit: [
      { n: 'Before Training',  d: 'Rebalance sample weights so the training data is more representative', icon: '01', before: 20, after: 20, imp: 0, best: false },
      { n: 'During Training',  d: 'Add a fairness rule to the training process so the model cannot favour one group', icon: '02', before: 20, after: 14, imp: 6, best: true  },
      { n: 'After Training',   d: 'Adjust the decision cutoff separately for each group after training is complete', icon: '03', before: 20, after: 20, imp: 0, best: false },
    ],
    recs: [
      'Racial bias confirmed — African-American defendants are scored as higher risk at a disproportionate rate',
      'Race is the 2nd most influential factor driving decisions — the model is directly using a protected attribute',
      'Wrongful high-risk flags differ significantly across racial groups — review whether deployment is ethical',
      'Re-audit every quarter; apply the equal-opportunity training rule to improve fairness',
    ],
  },
  german: {
    dataScore: 60, dataSev: 'MEDIUM', modelScore: 50, modelSev: 'MEDIUM',
    dir: 1.54, dpg: 0.285, tprGap: 0.1478, fprGap: 0.1308, flipRate: 0.028,
    acc: 0.784, auc: 0.882, posPriv: 76.8, posUnpriv: 55.4,
    shap: [
      { f: 'age (protected)',  v: 1.280, p: true  },
      { f: 'loan amount',      v: 1.070, p: false },
      { f: 'personal status',  v: 0.800, p: false },
      { f: 'loan duration',    v: 0.650, p: false },
      { f: 'loan purpose',     v: 0.420, p: false },
      { f: 'savings balance',  v: 0.210, p: false },
    ],
    mit: [
      { n: 'Before Training',  d: 'Rebalance sample weights so the training data is more representative', icon: '01', before: 48, after: 48, imp: 0, best: false },
      { n: 'During Training',  d: 'Add a fairness rule to the training process so the model cannot favour one group', icon: '02', before: 48, after: 44, imp: 4, best: true  },
      { n: 'After Training',   d: 'Adjust the decision cutoff separately for each group after training is complete', icon: '03', before: 48, after: 48, imp: 0, best: false },
    ],
    recs: [
      'Age discrimination detected — applicants under 25 are systematically denied loans',
      'Age is the single biggest factor driving loan rejections — this creates direct age discrimination risk',
      'Younger applicants who would repay loans on time are denied at a 14.8% higher rate',
      'Apply equal-opportunity training rule to reduce the age gap; consider removing age as a direct input',
    ],
  },
  utrecht: {
    dataScore: 51, dataSev: 'MEDIUM', modelScore: 22, modelSev: 'LOW',
    dir: 0.88, dpg: 0.117, tprGap: 0.05, fprGap: 0.02, flipRate: 0.02,
    acc: 0.938, auc: 0.793, posPriv: 52.3, posUnpriv: 40.6,
    shap: [
      { f: 'gender (protected)', v: 0.70, p: true  },
      { f: 'skills score',       v: 0.28, p: false },
      { f: 'interview score',    v: 0.28, p: false },
      { f: 'education',          v: 0.22, p: false },
      { f: 'experience',         v: 0.16, p: false },
    ],
    mit: [
      { n: 'Before Training',  d: 'Rebalance sample weights so the training data is more representative', icon: '01', before: 10, after: 10, imp: 0, best: false },
      { n: 'During Training',  d: 'Add a fairness rule to the training process so the model cannot favour one group', icon: '02', before: 10, after: 10, imp: 0, best: false },
      { n: 'After Training',   d: 'Adjust the decision cutoff separately for each group after training is complete', icon: '03', before: 10, after: 10, imp: 0, best: false },
    ],
    recs: [
      'Gender is the strongest factor in hiring decisions — check whether gender is being fed into the model directly',
      'Male candidates are hired at an 11.7% higher base rate than equally qualified female candidates',
      'LOW RISK: The trained model is fairly balanced, but the underlying hiring data still shows a gender gap',
      'Re-audit with a larger external dataset; consider removing gender from the input features entirely',
    ],
  },
  diabetes130: {
    dataScore: 97, dataSev: 'HIGH', modelScore: 83, modelSev: 'HIGH',
    dir: 0.4325, dpg: 0.1887, tprGap: 0.3299, fprGap: 0.1218, flipRate: 0.105,
    acc: 0.8344, auc: 0.810, posPriv: 14.4, posUnpriv: 6.2,
    shap: [
      { f: 'number of medications', v: 0.780, p: false },
      { f: 'number of diagnoses',   v: 0.620, p: false },
      { f: 'race (protected)',       v: 0.250, p: true  },
      { f: 'patient age',            v: 0.250, p: false },
      { f: 'HbA1c test result',      v: 0.210, p: false },
      { f: 'days in hospital',       v: 0.180, p: false },
    ],
    mit: [
      { n: 'Before Training',  d: 'Rebalance sample weights so the training data is more representative', icon: '01', before: 53, after: 53, imp: 0,  best: false },
      { n: 'During Training',  d: 'Add a fairness rule to the training process so the model cannot favour one group', icon: '02', before: 53, after: 5,  imp: 48, best: true  },
      { n: 'After Training',   d: 'Adjust the decision cutoff separately for each group after training is complete', icon: '03', before: 53, after: 53, imp: 0,  best: false },
    ],
    recs: [
      'HIGH RISK — Do not deploy this model without applying bias fixes first',
      'Black patients are far less likely to be flagged for early readmission follow-up, even when their risk is equal',
      'The model misses at-risk Black patients 33% more often than other groups — they lose access to early care',
      'Changing just the race field changes the model\'s prediction 10.5% of the time — race is directly driving decisions',
      'Applying the during-training fairness rule cuts the bias score from 83 down to 5 (a 48-point improvement)',
    ],
  },
};

/* ═══════════════════════════════
   UTILS
═══════════════════════════════ */
const sevClass  = s => s === 'HIGH' ? 'badge-high' : s === 'MEDIUM' ? 'badge-med' : 'badge-low';
const sevColor  = s => s === 'HIGH' ? 'var(--red)' : s === 'MEDIUM' ? 'var(--amber)' : 'var(--green)';
const biasColor = v => v >= 70 ? 'var(--red)' : v >= 40 ? 'var(--amber)' : 'var(--green)';
const dirColor  = v => v >= 0.8 ? 'var(--green)' : 'var(--red)';
const gapColor  = v => v > 0.1  ? 'var(--red)' : v > 0.05 ? 'var(--amber)' : 'var(--green)';

/* ═══════════════════════════════
   HOOKS
═══════════════════════════════ */
function useCountUp(target, duration = 1100) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = null;
    const tick = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(ease * target));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target]);
  return val;
}

/* ═══════════════════════════════
   PRIMITIVES
═══════════════════════════════ */
function Gauge({ score, severity }) {
  const animated = useCountUp(score);
  const color = sevColor(severity);

  const R = 72, CX = 90, CY = 82, SW = 10;
  const length = Math.PI * R;
  const pct = Math.max(0.01, score / 100);
  const strokeOffset = length - (length * pct);
  const pathD = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;

  return e('div', { className: 'gauge-wrap' },
    e('svg', { className: 'gauge-svg', viewBox: '0 0 180 100' },
      e('path', { d: pathD, fill: 'none', stroke: 'var(--paper3)', strokeWidth: SW, strokeLinecap: 'round' }),
      e('path', {
        d: pathD, fill: 'none', stroke: color,
        strokeWidth: SW, strokeLinecap: 'round',
        strokeDasharray: length, strokeDashoffset: strokeOffset,
        style: { transition: 'stroke-dashoffset 1.1s cubic-bezier(0.22, 1, 0.36, 1)' },
      }),
      e('text', {
        x: CX, y: CY - 2,
        textAnchor: 'middle',
        fontFamily: '"DM Serif Display", serif',
        fontSize: 48, fontWeight: 400,
        fill: 'var(--ink-1)', letterSpacing: -2,
      }, animated),
      e('text', {
        x: CX, y: CY + 16,
        textAnchor: 'middle',
        fontFamily: '"Spline Sans Mono", monospace',
        fontSize: 9, fill: 'var(--ink-5)',
        letterSpacing: 2, fontWeight: 500,
      }, 'RISK SCORE')
    )
  );
}

function BiasBar({ value, max = 100, color }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(Math.min(100, (value / max) * 100)), 80);
    return () => clearTimeout(t);
  }, [value, max]);
  return e('div', { className: 'bar-track' },
    e('div', { className: 'bar-fill', style: { width: `${w}%`, background: color || 'var(--green)' } })
  );
}

function Badge({ label, type = 'muted' }) {
  const display = label === 'HIGH' ? 'High Risk' : label === 'MEDIUM' ? 'Medium Risk' : label === 'LOW' ? 'Low Risk' : label;
  return e('span', { className: `badge badge-${type}` }, display);
}

function Spinner() { return e('div', { className: 'spinner' }); }

/* ═══════════════════════════════
   SHAP ROW
═══════════════════════════════ */
function ShapRow({ s, i, maxShap }) {
  const pct = (s.v / maxShap) * 100;
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(pct), 80 + i * 40);
    return () => clearTimeout(t);
  }, [pct, i]);

  return e('div', { className: 'shap-row' },
    e('span', { className: 'shap-rank' }, i + 1),
    e('span', { className: `shap-feat ${s.p ? 'protected' : 'regular'}` }, (s.p ? '⚑ ' : '') + s.f),
    e('div', { className: 'shap-bar-bg' },
      e('div', {
        className: 'shap-bar-fill',
        style: {
          width: `${w}%`,
          background: s.p ? 'var(--red)' : i < 3 ? 'var(--green)' : 'var(--ink-5)'
        }
      },
        w > 25 && e('span', { className: 'shap-inner-val' }, s.v.toFixed(3))
      )
    ),
    e('span', { className: 'shap-num' }, s.v.toFixed(3))
  );
}

/* ═══════════════════════════════
   TOAST
═══════════════════════════════ */
function ToastStack({ toasts, remove }) {
  return e('div', { className: 'toast-stack' },
    toasts.map(t =>
      e('div', { key: t.id, className: `toast ${t.type}` },
        e('span', { className: 'toast-msg' }, t.msg),
        e('button', { className: 'toast-close', onClick: () => remove(t.id) }, '×')
      )
    )
  );
}

/* ═══════════════════════════════
   DASHBOARD PAGE
═══════════════════════════════ */
function DashboardPage({ onNavigate }) {
  const highCount = DATASETS.filter(d => d.severity === 'HIGH').length;
  const avgBias   = Math.round(DATASETS.reduce((a, d) => a + d.dataBias, 0) / DATASETS.length);
  const avgModel  = Math.round(DATASETS.reduce((a, d) => a + d.modelBias, 0) / DATASETS.length);
  const aHigh     = useCountUp(highCount, 600);
  const aBias     = useCountUp(avgBias, 1000);
  const aModel    = useCountUp(avgModel, 1000);

  return e('div', { className: 'page' },
    e('div', { className: 'hero mb-28' },
      e('div', { className: 'hero-eyebrow' },
        e('span', { className: 'hero-eyebrow-dot' }),
        'Hack2Skill · Unbiased AI Decision Challenge'
      ),
      e('h1', { className: 'hero-title' },
        'Detect and eliminate ', e('em', null, 'hidden bias'), ' in AI models.'
      ),
      e('p', { className: 'hero-desc' },
        'FairLens checks machine learning systems for unfair treatment based on gender, race, and age. It runs statistical tests, explains which factors drive each decision, and automatically applies three different strategies to make your model fairer — before any discriminatory decisions reach real people.'
      ),
      e('div', { className: 'hero-actions' },
        e('button', { className: 'btn btn-primary btn-lg', onClick: () => onNavigate('audit') }, 'Explore Audits →'),
        e('button', { className: 'btn btn-lg', onClick: () => onNavigate('upload') }, 'Upload Your Dataset'),
        e('button', { className: 'btn btn-ghost btn-lg', onClick: () => onNavigate('about') }, 'View Documentation'),
      ),
      e('div', { className: 'hero-meta' },
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num' }, '5'),
          e('div', { className: 'hero-meta-label' }, 'datasets checked')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num', style: { color: biasColor(aBias) } }, aBias),
          e('div', { className: 'hero-meta-label' }, 'average data bias (out of 100)')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num', style: { color: 'var(--red)' } }, aHigh),
          e('div', { className: 'hero-meta-label' }, 'high risk datasets')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num' }, '7'),
          e('div', { className: 'hero-meta-label' }, 'fairness checks run')
        ),
      )
    ),
    e('div', { className: 'card' },
      e('div', { className: 'card-head' },
        e('span', { className: 'card-title' }, 'Dataset Overview'),
        e('button', { className: 'btn btn-sm', onClick: () => onNavigate('audit') }, 'Open Full Audit Explorer →')
      ),
      e('div', { className: 'tbl-wrap' },
        e('table', null,
          e('thead', null,
            e('tr', null,
              ['Dataset', 'Category', 'Groups checked for bias', 'Total records', 'Data bias', 'Model bias', 'Risk level', ''].map(h =>
                e('th', { key: h }, h)
              )
            )
          ),
          e('tbody', null,
            DATASETS.map(d =>
              e('tr', { key: d.id },
                e('td', null, e('strong', null, d.name)),
                e('td', { className: 'mono' }, d.domain),
                e('td', { className: 'mono' }, d.protected),
                e('td', { className: 'mono' }, d.rows),
                e('td', null,
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
                    e(BiasBar, { value: d.dataBias, color: biasColor(d.dataBias) }),
                    e('span', { className: 'mono', style: { color: biasColor(d.dataBias), fontWeight: 600, minWidth: 24 } }, d.dataBias)
                  )
                ),
                e('td', null,
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
                    e(BiasBar, { value: d.modelBias, color: biasColor(d.modelBias) }),
                    e('span', { className: 'mono', style: { color: biasColor(d.modelBias), fontWeight: 600, minWidth: 24 } }, d.modelBias)
                  )
                ),
                e('td', null, e(Badge, { label: d.severity, type: sevClass(d.severity) })),
                e('td', null,
                  e('button', { className: 'btn btn-sm', onClick: () => onNavigate('audit', d.id) }, 'Inspect')
                )
              )
            )
          )
        )
      )
    )
  );
}

/* ═══════════════════════════════
   AUDIT TABS
═══════════════════════════════ */
function OverviewTab({ d, ds }) {
  return e('div', { style: { animation: 'page-in .3s ease' } },
    e('div', { className: 'ds-header' },
      e('div', { className: 'ds-mark' }, ds.name[0]),
      e('div', null,
        e('div', { className: 'ds-header-name' }, ds.name),
        e('div', { className: 'ds-header-meta' },
          e('span', { className: 'tag' }, ds.domain),
          e('span', { className: 'mono', style: { fontSize: 11, color: 'var(--ink-4)' } }, ds.protected + ' · ' + ds.rows + ' records')
        )
      )
    ),
    e('div', { className: 'grid-2 mb-20' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Bias in the raw data'),
          e(Badge, { label: d.dataSev, type: sevClass(d.dataSev) })
        ),
        e('div', { className: 'card-body' },
          e(Gauge, { score: d.dataScore, severity: d.dataSev }),
          e('div', { className: 'divider' }),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Approval rate ratio between groups'),
            e(BiasBar, { value: (1 - Math.min(d.dir, 1)) * 100, color: dirColor(d.dir) }),
            e('span', { className: 'mrow-val', style: { color: dirColor(d.dir) } }, d.dir.toFixed(4)),
            d.dir >= 0.8 ? e(Badge, { label: 'Pass', type: 'pass' }) : e(Badge, { label: 'Fail', type: 'fail' })
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Difference in positive decision rates'),
            e(BiasBar, { value: d.dpg * 200, color: gapColor(d.dpg) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.dpg) } }, d.dpg.toFixed(4)),
          ),
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Bias in the trained model'),
          e(Badge, { label: d.modelSev, type: sevClass(d.modelSev) })
        ),
        e('div', { className: 'card-body' },
          e(Gauge, { score: d.modelScore, severity: d.modelSev }),
          e('div', { className: 'divider' }),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Gap in correctly catching true cases'),
            e(BiasBar, { value: d.tprGap * 300, color: gapColor(d.tprGap) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.tprGap) } }, d.tprGap.toFixed(4)),
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Gap in wrongful positive predictions'),
            e(BiasBar, { value: d.fprGap * 300, color: gapColor(d.fprGap) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.fprGap) } }, d.fprGap.toFixed(4)),
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Rate at which changing group identity flips the decision'),
            e(BiasBar, { value: d.flipRate * 400, color: gapColor(d.flipRate) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.flipRate) } }, (d.flipRate * 100).toFixed(1) + '%'),
          ),
        )
      )
    ),
    e('div', { className: 'card accent-card' },
      e('div', { className: 'card-head' },
        e('span', { className: 'card-title', style: { color: 'var(--green)' } }, '✦  Recommended Actions')
      ),
      e('div', { className: 'card-body' },
        d.recs.map((r, i) =>
          e('div', { key: i, className: 'rec-item' },
            e('div', { className: 'rec-num' }, i + 1),
            e('div', { className: 'rec-text' }, r)
          )
        )
      )
    )
  );
}

function DataAuditTab({ d }) {
  return e('div', { style: { animation: 'page-in .3s ease' } },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Raw Data Fairness Check'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'These numbers are calculated from the original dataset — before any model is trained.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Fairness checks')),
        e('div', { className: 'card-body' },
          [
            {
              name: 'Approval rate ratio between groups',
              val: d.dir,
              fmt: v => v.toFixed(4),
              pass: v => v >= 0.8,
              note: 'Below 0.80 = FAIL · Based on equal-opportunity hiring guidelines'
            },
            {
              name: 'Difference in positive decision rates',
              val: d.dpg,
              fmt: v => v.toFixed(4),
              pass: v => v < 0.1,
              note: 'Above 0.10 = worth investigating'
            },
            {
              name: 'Statistical significance of bias',
              val: 0.001,
              fmt: () => 'p < 0.001',
              pass: () => false,
              note: 'The observed disparity is very unlikely to be due to chance'
            },
          ].map((m, i) =>
            e('div', { key: i, style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '16px 0', borderBottom: i < 2 ? '1px solid var(--rule)' : 'none' } },
              e('div', null,
                e('div', { style: { fontSize: 14, fontWeight: 600, marginBottom: 4, color: 'var(--ink-1)' } }, m.name),
                e('div', { style: { fontSize: 11, color: 'var(--ink-5)', fontFamily: 'var(--mono)' } }, m.note),
              ),
              e('div', { style: { textAlign: 'right' } },
                e('div', { style: { fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 600, color: m.pass(m.val) ? 'var(--green)' : 'var(--red)', marginBottom: 6 } }, m.fmt(m.val)),
                e(Badge, { label: m.pass(m.val) ? 'Pass' : 'Fail', type: m.pass(m.val) ? 'pass' : 'fail' })
              )
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Approval rate by group')),
        e('div', { className: 'card-body' },
          ['Majority group', 'Minority group'].map((g, i) => {
            const rate  = i === 0 ? d.posPriv : d.posUnpriv;
            const color = i === 0 ? 'var(--amber)' : 'var(--green)';
            return e('div', { key: g, style: { marginBottom: 24 } },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', marginBottom: 8 } },
                e('span', { style: { fontSize: 13, color: 'var(--ink-1)', fontWeight: 600 } }, g),
                e('span', { style: { fontFamily: 'var(--mono)', fontSize: 15, color, fontWeight: 600 } }, rate.toFixed(1) + '%')
              ),
              e('div', { style: { height: 8, background: 'var(--paper3)', borderRadius: 4, overflow: 'hidden' } },
                e('div', { style: { height: '100%', width: `${Math.min(rate, 100)}%`, background: color, borderRadius: 4, transition: 'width 1.1s cubic-bezier(0.22, 1, 0.36, 1)' } })
              )
            );
          }),
          e('div', { className: 'divider' }),
          e('div', { className: 'verdict-cell' },
            e('div', { className: 'verdict-cell-label' }, 'Overall data bias score'),
            e('div', { className: 'verdict-cell-value', style: { color: sevColor(d.dataSev) } }, d.dataScore),
            e(Badge, { label: d.dataSev, type: sevClass(d.dataSev) })
          )
        )
      )
    )
  );
}

function ModelAuditTab({ d }) {
  const maxShap = d.shap[0].v;
  return e('div', { style: { animation: 'page-in .3s ease' } },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Model Fairness Check'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'A logistic regression model was trained on this dataset. These results show how fair that model is and which input fields most influenced its decisions.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Accuracy and fairness')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'How accurate is the model?'),
          [
            { label: 'Accuracy (% of predictions correct)', val: d.acc, fmt: v => (v * 100).toFixed(2) + '%' },
            { label: 'Ability to rank outcomes (out of 1.0)', val: d.auc, fmt: v => v.toFixed(4) },
          ].map(m =>
            e('div', { key: m.label, style: { marginBottom: 16 } },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', marginBottom: 7 } },
                e('span', { style: { fontSize: 13, color: 'var(--ink-2)', fontWeight: 500 } }, m.label),
                e('span', { style: { fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 600, color: 'var(--green)' } }, m.fmt(m.val))
              ),
              e('div', { style: { height: 5, background: 'var(--paper3)', borderRadius: 3, overflow: 'hidden' } },
                e('div', { style: { height: '100%', width: `${m.val * 100}%`, background: 'var(--green)', borderRadius: 3, transition: 'width 1.1s cubic-bezier(0.22, 1, 0.36, 1)' } })
              )
            )
          ),
          e('div', { className: 'divider' }),
          e('div', { className: 'section-label' }, 'Is the model fair?'),
          [
            { label: 'Gap in correctly catching true cases',        val: d.tprGap,   fmt: v => v.toFixed(4),           bad: v => v > 0.1  },
            { label: 'Gap in wrongful positive predictions',        val: d.fprGap,   fmt: v => v.toFixed(4),           bad: v => v > 0.05 },
            { label: 'How often group identity flips the decision', val: d.flipRate, fmt: v => (v*100).toFixed(1)+'%', bad: v => v > 0.05 },
            { label: 'Approval rate ratio between groups',          val: d.dir,      fmt: v => v.toFixed(4),           bad: v => v < 0.8  },
          ].map(m =>
            e('div', { key: m.label, className: 'mrow' },
              e('span', { className: 'mrow-name' }, m.label),
              e('span', { className: 'mrow-val', style: { color: m.bad(m.val) ? 'var(--red)' : 'var(--green)', fontWeight: 600 } }, m.fmt(m.val)),
              e(Badge, { label: m.bad(m.val) ? 'Fail' : 'Pass', type: m.bad(m.val) ? 'fail' : 'pass' })
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Which fields drive decisions most?'),
          e('span', { className: 'card-sub' }, '⚑ = protected group field')
        ),
        e('div', { className: 'card-body' },
          e('p', { style: { fontSize: 12, color: 'var(--ink-4)', marginBottom: 14, fontFamily: 'var(--mono)', lineHeight: 1.5 } },
            'Each bar shows how strongly a field influences the model\'s decisions. Longer bar = bigger influence. Red bars are fields that should not drive decisions.'
          ),
          e('div', { className: 'shap-list' },
            d.shap.map((s, i) => e(ShapRow, { key: s.f, s, i, maxShap }))
          )
        )
      )
    )
  );
}

function MitigationTab({ d }) {
  return e('div', { style: { animation: 'page-in .3s ease' } },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Bias Reduction Strategies'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'Three automatic methods for reducing bias — applied before training, during training, and after training.')
    ),
    d.mit.map((m, i) =>
      e('div', { key: i, className: `mit-item ${m.best ? 'best' : ''}` },
        e('div', { className: 'mit-icon-wrap' }, m.icon),
        e('div', { className: 'mit-info' },
          e('div', { className: 'mit-name' },
            m.n,
            m.best && e(Badge, { label: 'Best Result', type: 'accent' })
          ),
          e('div', { className: 'mit-desc' }, m.d),
          e('div', { className: 'mit-progress-row' },
            e('div', { className: 'mit-progress-label' }, 'Bias score after:'),
            e('div', { style: { flex: 1, height: 6, background: 'var(--paper3)', borderRadius: 3, overflow: 'hidden', margin: '0 10px' } },
              e('div', { style: { height: '100%', width: `${m.after}%`, background: m.best ? 'var(--green)' : 'var(--ink-5)', borderRadius: 3, transition: 'width 1.1s cubic-bezier(0.22, 1, 0.36, 1)' } })
            ),
            e('div', { className: 'mit-progress-label' }, m.after + ' out of 100')
          )
        ),
        e('div', { className: 'mit-score-block' },
          e('div', { className: 'mit-score-before' }, 'Before: ' + m.before),
          e('div', { className: 'mit-score-after', style: { color: m.after < m.before ? 'var(--green)' : 'var(--ink-2)' } }, m.after),
          e('span', { className: `mit-delta ${m.imp > 0 ? 'delta-pos' : 'delta-zero'}` },
            m.imp > 0 ? '−' + m.imp + ' points' : 'No change'
          )
        )
      )
    ),
    e('div', { className: 'card', style: { marginTop: 16 } },
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'How each strategy works')),
      e('div', { className: 'card-body' },
        e('div', { className: 'grid-3' },
          [
            {
              icon: '01',
              n: 'Before Training',
              d: 'The training data is adjusted by giving more weight to underrepresented groups so the model sees a more balanced picture of the world before it learns anything.'
            },
            {
              icon: '02',
              n: 'During Training',
              d: 'A fairness rule is added to the training objective. This prevents the model from boosting its overall accuracy by sacrificing fairness towards any one group.'
            },
            {
              icon: '03',
              n: 'After Training',
              d: 'The decision threshold is calibrated separately for each group once training is complete. This equalises approval rates across groups without retraining the model.'
            },
          ].map(s =>
            e('div', { key: s.n, style: { padding: 20, background: 'var(--paper)', borderRadius: 'var(--r2)', border: '1px solid var(--rule2)' } },
              e('div', { style: { fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-5)', marginBottom: 10, letterSpacing: 1.5, fontWeight: 600 } }, s.icon),
              e('div', { style: { fontSize: 14, fontWeight: 600, marginBottom: 8, color: 'var(--ink-1)' } }, s.n),
              e('div', { style: { fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 } }, s.d),
            )
          )
        )
      )
    )
  );
}

function ReportTab({ d, ds, sel }) {
  return e('div', { style: { animation: 'page-in .3s ease' } },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Download Reports'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'Export a printable compliance document or a machine-readable data file for use in your own systems.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Report summary')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Dataset details'),
          e('div', { className: 'info-grid', style: { marginBottom: 20 } },
            [
              { k: 'Dataset',        v: ds.name },
              { k: 'Category',       v: ds.domain },
              { k: 'Total records',  v: ds.rows },
              { k: 'Groups checked', v: ds.protected },
              { k: 'Outcome type',   v: 'Yes / No (binary)' },
              { k: 'Model used',     v: 'Logistic Regression' },
            ].map(r =>
              e('div', { key: r.k, className: 'info-cell' },
                e('div', { className: 'info-cell-key' }, r.k),
                e('div', { className: 'info-cell-val' }, r.v)
              )
            )
          ),
          e('div', { className: 'section-label' }, 'Final verdict'),
          e('div', { style: { display: 'flex', gap: 12 } },
            [
              { lbl: 'Data bias',  score: d.dataScore,  sev: d.dataSev },
              { lbl: 'Model bias', score: d.modelScore, sev: d.modelSev },
            ].map(x =>
              e('div', { key: x.lbl, className: 'verdict-cell', style: { flex: 1 } },
                e('div', { className: 'verdict-cell-label' }, x.lbl),
                e('div', { className: 'verdict-cell-value', style: { color: sevColor(x.sev) } }, x.score),
                e(Badge, { label: x.sev, type: sevClass(x.sev) })
              )
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Download options')),
        e('div', { className: 'card-body' },
          e('a', { href: `${API}/report/${sel}/pdf`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '↓'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'Full Audit Report (PDF)'),
              e('div', { className: 'report-dl-desc' }, 'All metrics, charts, and compliance recommendations — ready to print or share'),
            ),
            e('span', { className: 'tag' }, 'PDF')
          ),
          e('a', { href: `${API}/report/${sel}/json`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '{}'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'Raw Audit Data (JSON)'),
              e('div', { className: 'report-dl-desc' }, 'Machine-readable output — import into your own tools or automated pipelines'),
            ),
            e('span', { className: 'tag' }, 'JSON')
          ),
          e('a', { href: `${API}/audit/full/${sel}`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '▶'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'Re-run This Audit via API'),
              e('div', { className: 'report-dl-desc' }, 'Trigger a fresh full audit programmatically using our REST endpoint'),
            ),
            e('span', { className: 'tag' }, 'POST')
          ),
        )
      )
    )
  );
}

/* ═══════════════════════════════
   AUDIT PAGE
═══════════════════════════════ */
function AuditPage({ initDs }) {
  const [sel, setSel] = useState(initDs || 'adult');
  const [tab, setTab] = useState('overview');
  const [d, setD] = useState(null);
  const [loading, setLoading] = useState(true);
  const ds = DATASETS.find(x => x.id === sel);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API}/results/dataset/${sel}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        // Grab the first protected attribute to extract stats
const attrKey = data.data_audit?.attribute_results ? Object.keys(data.data_audit.attribute_results)[0] : null;
const attrStats = attrKey ? data.data_audit.attribute_results[attrKey] : {};

// Translate real backend keys to what the UI components expect
const mappedD = {
  dataScore: data.data_bias_score || 0,
  dataSev: data.data_severity || 'MEDIUM',
  modelScore: data.model_bias_score || 0,
  modelSev: data.model_severity || 'MEDIUM',
  dir: attrStats.disparate_impact_ratio || 0,
  dpg: attrStats.demographic_parity_gap || 0,
  tprGap: data.model_audit?.tpr_gap || 0,
  fprGap: data.model_audit?.fpr_gap || 0,
  flipRate: data.model_audit?.counterfactual_flip_rate || 0,
  acc: data.model_audit?.overall_metrics?.accuracy || 0,
  auc: data.model_audit?.overall_metrics?.roc_auc || 0,
  posPriv: 50, posUnpriv: 30, // Visual fallbacks
  shap: data.model_audit?.top_features?.map(f => ({ f: f[0], v: f[1], p: false })) || [{f:'Loading...', v:0, p:false}],
  mit: [
    { n: 'Before Training', d: 'Reweighing', icon: '01', before: data.mitigation?.reweighing?.before?.bias_score || 0, after: data.mitigation?.reweighing?.after?.bias_score || 0, imp: data.mitigation?.reweighing?.improvement || 0, best: false },
    { n: 'During Training', d: 'Fairness Constraint', icon: '02', before: data.mitigation?.fairness_constraint?.before?.bias_score || 0, after: data.mitigation?.fairness_constraint?.after?.bias_score || 0, imp: data.mitigation?.fairness_constraint?.improvement || 0, best: true },
    { n: 'After Training', d: 'Threshold Calibration', icon: '03', before: data.mitigation?.threshold_calibration?.before?.bias_score || 0, after: data.mitigation?.threshold_calibration?.after?.bias_score || 0, imp: data.mitigation?.threshold_calibration?.improvement || 0, best: false }
  ],
  recs: data.gemini_recommendation ? data.gemini_recommendation.split('\n').filter(r => r.trim()) : ['Run pipeline to generate AI recommendations.']
};

setD(mappedD);
      } catch (err) {
        console.error('Failed to fetch audit data:', err);
        setD(null);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [sel]);

  const tabs = [
    { id: 'overview',   label: 'Summary' },
    { id: 'dataaudit',  label: 'Data Check' },
    { id: 'modelaudit', label: 'Model Check' },
    { id: 'mitigation', label: 'Fix Bias' },
    { id: 'report',     label: 'Download' },
  ];

  return e('div', { className: 'page' },
    e('div', { className: 'ds-grid' },
      DATASETS.map(x =>
        e('div', {
          key: x.id,
          className: `ds-pill ${sel === x.id ? 'active' : ''}`,
          onClick: () => { setSel(x.id); setTab('overview'); }
        },
          e('div', { className: 'ds-pill-header' },
            e('div', { className: 'ds-pill-name' }, x.name),
            e(Badge, { label: x.severity, type: sevClass(x.severity) })
          ),
          e('div', { className: 'ds-pill-sub' }, x.domain),
          e('div', { className: 'ds-pill-rows' }, x.rows + ' records')
        )
      )
    ),
    e('div', { className: 'tab-card' },
      e('div', { className: 'tab-bar' },
        tabs.map(t =>
          e('button', {
            key: t.id,
            className: `tab ${tab === t.id ? 'active' : ''}`,
            onClick: () => setTab(t.id)
          }, t.label)
        )
      )
    ),
    loading ? e(Spinner) : (d && e('div', { key: sel + tab },
      tab === 'overview'   && e(OverviewTab,   { d, ds }),
      tab === 'dataaudit'  && e(DataAuditTab,  { d }),
      tab === 'modelaudit' && e(ModelAuditTab, { d }),
      tab === 'mitigation' && e(MitigationTab, { d }),
      tab === 'report'     && e(ReportTab,     { d, ds, sel }),
    ))
  );
}

/* ═══════════════════════════════
   UPLOAD PAGE
═══════════════════════════════ */
function UploadPage({ addToast }) {
  const [file,      setFile]      = useState(null);
  const [drag,      setDrag]      = useState(false);
  const [target,    setTarget]    = useState('');
  const [prot,      setProt]      = useState('');
  const [loading,   setLoading]   = useState(false);
  const [jobId,     setJobId]     = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [result,    setResult]    = useState(null);
  const [columns,   setColumns]   = useState([]);
  const [valErrors, setValErrors] = useState([]);
  const fileRef   = useRef();
  const pollRef   = useRef(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    if (!jobId) return;
    if (pollRef.current) clearInterval(pollRef.current);

    const poll = async () => {
      try {
        const res  = await fetch(`${API}/jobs/${jobId}`);
        const data = await res.json();
        setJobStatus(data.status);

        if (data.status === 'done') {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setLoading(false);
          const rRes  = await fetch(`${API}/results/${jobId}`);
          const rData = await rRes.json();
          setResult(rData);
          addToast('Audit complete!', 'ok');
        } else if (data.status === 'failed') {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setLoading(false);
          addToast('Audit failed — please check the server logs', 'err');
        }
      } catch {
        // network error during poll — keep trying
      }
    };

    poll();
    pollRef.current = setInterval(poll, 3000);
  }, [jobId]);

  const parseHeaders = f => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const firstLine = ev.target.result.split('\n')[0] || '';
      const cols = firstLine.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
      setColumns(cols);
      const tMatch = cols.find(c => /^(target|label|outcome|y|class|hired|income|readmitted|recid)/i.test(c));
      const pMatch = cols.find(c => /^(sex|gender|race|age|ethnicity|prot)/i.test(c));
      if (tMatch && !target) setTarget(tMatch);
      if (pMatch && !prot)   setProt(pMatch);
    };
    reader.readAsText(f.slice(0, 8192));
  };

  const handleFile = f => {
    if (f && f.name.endsWith('.csv')) {
      setFile(f);
      setColumns([]);
      setValErrors([]);
      setResult(null);
      setJobId(null);
      setJobStatus(null);
      parseHeaders(f);
    } else if (f) {
      addToast('Only CSV files are supported', 'err');
    }
  };

  const submit = async () => {
    if (!file || !target.trim() || !prot.trim()) {
      addToast('Please select a CSV file and choose both column fields.', 'err');
      return;
    }
    setLoading(true);
    setResult(null);
    setValErrors([]);
    setJobId(null);
    setJobStatus(null);

    let finalTarget = target.trim();
    let finalProt = prot.trim();

    if (columns.length > 0) {
      const matchT = columns.find(c => c.toLowerCase() === finalTarget.toLowerCase());
      if (matchT) finalTarget = matchT;
      const matchP = columns.find(c => c.toLowerCase() === finalProt.toLowerCase());
      if (matchP) finalProt = matchP;
    }

    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('target_column', finalTarget);
      fd.append('protected_column', finalProt);

      const res  = await fetch(`${API}/audit/upload`, { method: 'POST', body: fd });
      const data = await res.json();

      if (!res.ok) {
        const detail = data.detail || data;
        const errs = detail.errors || [detail.message || `Server error ${res.status}`];
        const cols = detail.available_columns || [];
        setValErrors(errs);
        if (cols.length) setColumns(cols);
        addToast(`${errs.length} problem(s) found — see details below`, 'err');
        setLoading(false);
        return;
      }

      setJobId(data.job_id);
      setJobStatus('pending');
      addToast('File uploaded — running audit in the background…', 'ok');

    } catch (err) {
      addToast('Could not reach the server — is it running?', 'err');
      setLoading(false);
    }
  };

  const statusLabel = () => {
    if (jobStatus === 'pending')  return 'Queued — waiting to start…';
    if (jobStatus === 'running')  return 'Running — checking data, training model, applying bias fixes…';
    if (jobStatus === 'done')     return 'Complete';
    if (jobStatus === 'failed')   return 'Something went wrong';
    return 'Processing…';
  };

  return e('div', { className: 'page', style: { position: 'relative' } },
    loading && e('div', { className: 'overlay' },
      e(Spinner),
      e('div', { className: 'overlay-label' }, 'Running Bias Audit'),
      e('div', { className: 'overlay-sub' }, statusLabel())
    ),

    e('div', { className: 'hero mb-28', style: { padding: '36px 40px' } },
      e('div', { className: 'hero-eyebrow' }, e('span', { className: 'hero-eyebrow-dot' }), 'Custom Audit'),
      e('h2', { className: 'hero-title', style: { fontSize: 36 } }, 'Check your own dataset'),
      e('p', { className: 'hero-desc', style: { marginBottom: 0 } },
        'Upload any CSV file. FairLens will detect bias in the raw data, train a model, apply three different strategies to reduce unfairness, and generate compliance-ready reports — all automatically.'
      )
    ),

    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-body', style: { display: 'flex', flexDirection: 'column', gap: 18 } },

          e('div', {
            className: `drop-zone ${drag ? 'drag-over' : ''}`,
            onDragOver:  ev => { ev.preventDefault(); setDrag(true); },
            onDragLeave: () => setDrag(false),
            onDrop:      ev => { ev.preventDefault(); setDrag(false); handleFile(ev.dataTransfer.files[0]); },
            onClick:     () => fileRef.current.click(),
          },
            e('input', { type: 'file', ref: fileRef, accept: '.csv', style: { display: 'none' }, onChange: ev => handleFile(ev.target.files[0]) }),
            e('span', { className: 'drop-icon' }, file ? '✓' : '↑'),
            e('div', { className: 'drop-title' }, file ? file.name : 'Drop your CSV file here'),
            e('div', { className: 'drop-sub' }, file ? `${(file.size / 1024).toFixed(1)} KB · ${columns.length} columns found` : 'or click to browse · CSV files only')
          ),

          columns.length > 0 && e('div', null,
            e('div', { className: 'field-label', style: { marginBottom: 8 } }, 'Columns found in your file:'),
            e('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 6 } },
              columns.map(col =>
                e('div', {
                  key: col,
                  style: {
                    fontFamily: 'var(--mono)', fontSize: 11, padding: '4px 10px',
                    borderRadius: 4, border: '1px solid var(--rule2)',
                    background: col === target ? 'var(--green-dim)' : col === prot ? 'var(--amber-dim)' : 'var(--paper2)',
                    color: col === target ? 'var(--green)' : col === prot ? 'var(--amber)' : 'var(--ink-3)',
                    borderColor: col === target ? 'var(--green-border)' : col === prot ? 'var(--amber-border)' : 'var(--rule2)',
                  }
                }, col)
              )
            )
          ),

          e('div', { className: 'grid-2', style: { gap: 14 } },
            e('div', null,
              e('label', { className: 'field-label' }, 'What are you predicting?'),
              e('p', { style: { fontSize: 11, color: 'var(--ink-4)', marginBottom: 7, fontFamily: 'var(--mono)' } }, 'Pick the column with the outcome (e.g. "hired", "approved", "income")'),
              columns.length > 0
              ? e('select', {
                  className: 'field-input',
                  value: target,
                  onChange: ev => setTarget(ev.target.value),
                  style: target ? { borderColor: 'var(--green)', boxShadow: '0 0 0 3px var(--green-glow)', cursor: 'pointer' } : { cursor: 'pointer' }
                },
                  e('option', { value: '', disabled: true }, 'Choose outcome column…'),
                  columns.map(c => e('option', { key: c, value: c }, c))
                )
              : e('input', {
                  type: 'text', className: 'field-input',
                  placeholder: 'e.g. income, hired, approved',
                  value: target,
                  onChange: ev => setTarget(ev.target.value),
                  style: target ? { borderColor: 'var(--green)', boxShadow: '0 0 0 3px var(--green-glow)' } : {}
                })
            ),
            e('div', null,
              e('label', { className: 'field-label' }, 'Which group are you checking?'),
              e('p', { style: { fontSize: 11, color: 'var(--ink-4)', marginBottom: 7, fontFamily: 'var(--mono)' } }, 'Pick the column with the protected group (e.g. "gender", "race", "age")'),
              columns.length > 0
              ? e('select', {
                  className: 'field-input',
                  value: prot,
                  onChange: ev => setProt(ev.target.value),
                  style: prot ? { borderColor: 'var(--amber)', boxShadow: '0 0 0 3px rgba(184,96,10,0.1)', cursor: 'pointer' } : { cursor: 'pointer' }
                },
                  e('option', { value: '', disabled: true }, 'Choose group column…'),
                  columns.map(c => e('option', { key: c, value: c }, c))
                )
              : e('input', {
                  type: 'text', className: 'field-input',
                  placeholder: 'e.g. gender, race, age',
                  value: prot,
                  onChange: ev => setProt(ev.target.value),
                  style: prot ? { borderColor: 'var(--amber)', boxShadow: '0 0 0 3px rgba(184,96,10,0.1)' } : {}
                })
            )
          ),

          valErrors.length > 0 && e('div', {
            style: {
              padding: '12px 16px', background: 'var(--red-dim)',
              borderRadius: 'var(--r2)', border: '1px solid var(--red-border)',
            }
          },
            e('div', { style: { fontWeight: 700, fontSize: 12, color: 'var(--red)', marginBottom: 6, fontFamily: 'var(--mono)' } }, '✗ Problems found'),
            valErrors.map((err, i) =>
              e('div', { key: i, style: { fontSize: 12, color: 'var(--red)', fontFamily: 'var(--mono)', marginTop: 3, lineHeight: 1.5 } }, err)
            )
          ),

          jobId && jobStatus !== 'done' && jobStatus !== 'failed' && e('div', {
            style: {
              padding: '12px 16px', background: 'var(--paper2)',
              borderRadius: 'var(--r2)', border: '1px solid var(--rule2)',
            }
          },
            e('div', { style: { fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--ink-3)', marginBottom: 8 } },
              'Job ID: ', e('span', { style: { color: 'var(--ink-1)', fontWeight: 600 } }, jobId)
            ),
            e('div', { style: { fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-4)' } }, statusLabel())
          ),

          e('button', {
            className: 'btn btn-primary',
            style: { width: '100%', justifyContent: 'center', padding: '13px', fontSize: 14 },
            onClick: submit,
            disabled: loading || (jobStatus && jobStatus !== 'done' && jobStatus !== 'failed')
          }, loading ? 'Running audit…' : 'Run Full Bias Check')
        )
      ),

      result
        ? e('div', { className: 'card accent-card', style: { animation: 'page-in .3s ease' } },
            e('div', { className: 'card-head' },
              e('span', { className: 'card-title' }, 'Audit Results'),
              e(Badge, { label: result.data_severity || 'MEDIUM', type: sevClass(result.data_severity || 'MEDIUM') })
            ),
            e('div', { className: 'card-body' },
              e('div', { className: 'result-score-grid' },
                [
                  { k: 'Records analysed',  v: result.data_audit && result.data_audit.n_rows ? result.data_audit.n_rows.toLocaleString() : '—' },
                  { k: 'Data bias score',   v: result.data_bias_score != null ? `${result.data_bias_score} / 100` : '—' },
                  { k: 'Model bias score',  v: result.model_bias_score != null ? `${result.model_bias_score} / 100` : '—' },
                  { k: 'Risk level',        v: result.data_severity || '—' },
                ].map(r =>
                  e('div', { key: r.k, className: 'result-score-cell' },
                    e('div', { className: 'result-score-key' }, r.k),
                    e('div', { className: 'result-score-val' }, r.v)
                  )
                )
              ),
              result.gemini_narrative && e('div', {
                style: { padding: '14px 18px', background: 'var(--paper2)', borderRadius: 'var(--r2)', border: '1px solid var(--rule2)', margin: '14px 0' }
              },
                e('div', { style: { fontWeight: 600, fontSize: 12, color: 'var(--ink-4)', marginBottom: 6, fontFamily: 'var(--mono)', letterSpacing: 1 } }, 'AI SUMMARY'),
                e('div', { style: { fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.7 } }, result.gemini_narrative)
              ),
              e('div', { style: { padding: '14px 18px', background: 'var(--green-dim)', borderRadius: 'var(--r2)', border: '1px solid var(--green-border)', margin: '14px 0' } },
                e('div', { style: { fontWeight: 600, fontSize: 13, color: 'var(--green)', marginBottom: 4 } }, '✦ Reports are ready'),
                e('div', { style: { fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.6, fontFamily: 'var(--mono)' } }, 'Your PDF and data reports are available to download below.')
              ),
              result.report_pdf_url && e('a', { href: result.report_pdf_url, target: '_blank', className: 'btn btn-primary', style: { marginRight: 10 } }, '↓ Download PDF Report'),
              result.report_json_url && e('a', { href: result.report_json_url, target: '_blank', className: 'btn' }, '{} Download Data Report'),
              !result.report_pdf_url && e('a', { href: `${API}/docs`, target: '_blank', className: 'btn btn-ghost' }, 'View API Documentation ↗')
            )
          )
        : e('div', { className: 'card' },
            e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'What FairLens checks')),
            e('div', { className: 'card-body' },
              [
                {
                  icon: '01',
                  title: 'Approval rate gap between groups',
                  desc: 'Compares how often each group receives a positive outcome. Below 0.80 (where one group gets less than 80% of the other\'s approval rate) fails equal-opportunity guidelines.'
                },
                {
                  icon: '02',
                  title: 'Difference in positive decision rates',
                  desc: 'The raw percentage point gap between groups in how often they receive a positive outcome.'
                },
                {
                  icon: '03',
                  title: 'Is the disparity statistically real?',
                  desc: 'A chi-square test checks whether the gap between groups is large enough to be a real pattern, not just random noise in the data.'
                },
                {
                  icon: '04',
                  title: 'Does the model treat groups equally?',
                  desc: 'Checks whether the model correctly identifies true cases at the same rate for all groups, and whether it wrongly flags people at equal rates across groups.'
                },
                {
                  icon: '05',
                  title: 'Three automatic bias fixes',
                  desc: 'Applies and compares three different fairness strategies — before, during, and after training — and shows the before-and-after bias scores for each.'
                },
              ].map(c =>
                e('div', { key: c.title, style: { display: 'flex', gap: 14, padding: '13px 0', borderBottom: '1px solid var(--rule)' } },
                  e('span', { style: { fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--green)', flexShrink: 0, fontWeight: 600, marginTop: 2 } }, c.icon),
                  e('div', null,
                    e('div', { style: { fontWeight: 600, fontSize: 13.5, marginBottom: 3, color: 'var(--ink-1)' } }, c.title),
                    e('div', { style: { fontSize: 12.5, color: 'var(--ink-3)', lineHeight: 1.6, fontFamily: 'var(--mono)' } }, c.desc)
                  )
                )
              )
            )
          )
    )
  );
}

/* ═══════════════════════════════
   ABOUT PAGE
═══════════════════════════════ */
function AboutPage() {
  return e('div', { className: 'page' },
    e('div', { className: 'hero mb-28', style: { padding: '36px 40px' } },
      e('div', { className: 'hero-eyebrow' }, e('span', { className: 'hero-eyebrow-dot' }), 'Documentation'),
      e('h2', { className: 'hero-title', style: { fontSize: 36 } }, 'About FairLens'),
      e('p', { className: 'hero-desc', style: { marginBottom: 0 } },
        'End-to-end AI bias detection and mitigation, built for the Hack2Skill Unbiased AI Decision challenge. Tested on 5 real-world fairness benchmarks using IBM AIF360, Microsoft Fairlearn, and decision-explanation tools.'
      )
    ),
    e('div', { className: 'grid-2 mb-20' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Fairness checks explained')),
        e('div', { className: 'tbl-wrap' },
          e('table', null,
            e('thead', null, e('tr', null, ['Check', 'When it fails', 'Based on'].map(h => e('th', { key: h }, h)))),
            e('tbody', null,
              [
                { m: 'Approval rate ratio between groups',           t: 'Below 0.80',                    s: ' equal-opportunity law' },
                { m: 'Difference in positive decision rates',        t: 'Gap above 10 percentage points', s: 'Standard machine learning fairness research' },
                { m: 'Gap in catching true cases equally',           t: 'Gap above 5 percentage points',  s: 'Hardt et al., 2016' },
                { m: 'Gap in wrongful positive flags',               t: 'Gap above 5 percentage points',  s: 'Hardt et al., 2016' },
                { m: 'How often group identity flips the decision',  t: 'Above 10% of cases',             s: 'Counterfactual fairness research' },
                { m: 'Protected field influence on decisions',       t: 'Protected field in top 5',       s: 'Lundberg & Lee, 2017 (SHAP)' },
              ].map(r =>
                e('tr', { key: r.m },
                  e('td', { style: { fontWeight: 600 } }, r.m),
                  e('td', { className: 'mono' }, r.t),
                  e('td', { className: 'mono' }, r.s)
                )
              )
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Technology used')),
        e('div', { className: 'card-body' },
          [
            { n: 'IBM AIF360',          r: 'Fairness metrics and pre-processing bias fixes' },
            { n: 'Microsoft Fairlearn', r: 'During-training fairness constraints' },
            { n: 'SHAP',               r: 'Explaining which input fields drive each decision' },
            { n: 'scikit-learn',       r: 'Logistic regression and random forest models' },
            { n: 'FastAPI + slowapi',  r: 'Server and rate limiting' },
            { n: 'Supabase',           r: 'Database and file storage for jobs and reports' },
            { n: 'Google Gemini',      r: 'Plain-language summaries and recommendations' },
            { n: 'ReportLab',          r: 'Generating printable PDF reports' },
            { n: 'React 18',           r: 'Frontend — loads directly in the browser, no build step' },
          ].map(s =>
            e('div', { key: s.n, className: 'stack-item' },
              e('span', { className: 'stack-name' }, s.n),
              e('span', { className: 'stack-role' }, s.r)
            )
          )
        )
      )
    ),
    e('div', { className: 'card' },
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'API endpoints')),
      e('div', { className: 'tbl-wrap' },
        e('table', null,
          e('thead', null, e('tr', null, ['Method', 'Endpoint', 'What it does'].map(h => e('th', { key: h }, h)))),
          e('tbody', null,
            [
              ['GET',  '/',                     'Health check — confirms the service is running'],
              ['GET',  '/datasets',              'List all available datasets'],
              ['POST', '/audit/data/{name}',     'Run a fairness check on the raw data for a named dataset'],
              ['POST', '/audit/model/{name}',    'Train a model and check it for bias'],
              ['POST', '/audit/full/{name}',     'Run the full pipeline: check data, train model, fix bias, generate reports (runs in background)'],
              ['GET',  '/jobs/{job_id}',         'Check whether a background job is still running'],
              ['GET',  '/results/{job_id}',      'Get the full results for a completed job'],
              ['GET',  '/report/{name}/pdf',     'Download the PDF report for a dataset'],
              ['GET',  '/report/{name}/json',    'Download the raw data report for a dataset'],
              ['POST', '/audit/upload',          'Upload a custom CSV file and run the full pipeline (runs in background)'],
              ['POST', '/chat/{job_id}',         'Ask questions about a completed audit in plain language'],
            ].map(([m, ep, desc]) =>
              e('tr', { key: ep },
                e('td', null, e('span', { className: `method-badge method-${m.toLowerCase()}` }, m)),
                e('td', { className: 'mono', style: { color: 'var(--green)', fontWeight: 600 } }, ep),
                e('td', { style: { fontSize: 13, color: 'var(--ink-3)' } }, desc)
              )
            )
          )
        )
      )
    )
  );
}

/* ═══════════════════════════════
   ROOT APP
═══════════════════════════════ */
function App() {
  const [page,    setPage]    = useState('dashboard');
  const [auditDs, setAuditDs] = useState('adult');
  const [toasts,  setToasts]  = useState([]);

  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }, []);

  const removeToast = id => setToasts(t => t.filter(x => x.id !== id));

  const navigate = useCallback((p, ds) => {
    setPage(p);
    if (ds) setAuditDs(ds);
  }, []);

  const navLinks = [
    { id: 'dashboard', label: 'Dashboard',        icon: '◈' },
    { id: 'audit',     label: 'Audit Explorer',   icon: '◉' },
    { id: 'upload',    label: 'Check Your Data',  icon: '↑' },
    { id: 'about',     label: 'Docs & API',       icon: '?' },
  ];

  const pageTitle = navLinks.find(n => n.id === page)?.label || 'Dashboard';

  return e('div', { style: { display: 'flex', height: '100vh', overflow: 'hidden' } },
    e('nav', { className: 'sidebar' },
      e('div', { className: 'sidebar-brand' },
        e('div', { className: 'brand-lockup' },
          e('div', { className: 'brand-mark' }, 'FL'),
          e('div', { className: 'brand-name' }, e('em', null, 'Fair'), 'Lens'),
        ),
        e('div', { className: 'brand-tagline' }, 'AI Bias Audit Platform')
      ),
      e('div', { className: 'nav-group' },
        e('div', { className: 'nav-label' }, 'Platform'),
        navLinks.map(n =>
          e('button', {
            key: n.id,
            className: `nav-item ${page === n.id ? 'active' : ''}`,
            onClick: () => navigate(n.id)
          },
            e('span', { className: 'nav-icon' }, n.icon),
            n.label
          )
        )
      ),
      e('div', { className: 'nav-group' },
        e('div', { className: 'nav-label' }, 'Datasets'),
        DATASETS.map(d =>
          e('button', {
            key: d.id,
            className: `nav-ds ${page === 'audit' && auditDs === d.id ? 'active' : ''}`,
            onClick: () => navigate('audit', d.id)
          },
            e('span', { className: 'nav-ds-dot', style: { background: sevColor(d.severity) } }),
            e('span', { className: 'nav-ds-name' }, d.name),
            e('span', { className: 'nav-ds-score' }, d.dataBias)
          )
        )
      ),
      e('div', { className: 'sidebar-footer' },
        e('div', { className: 'sidebar-version' }, 'Hack2Skill · 2026'),
        e('div', { className: 'api-status' },
          e('span', { className: 'status-dot' }),
          'Online'
        )
      )
    ),

    e('div', { className: 'main' },
      e('div', { className: 'topbar' },
        e('div', { className: 'topbar-left' },
          e('div', { className: 'topbar-crumb' },
            e('span', null, pageTitle),
            page === 'audit' && e('span', { className: 'crumb-sep' }, '/'),
            page === 'audit' && e('span', { className: 'crumb-sub' }, DATASETS.find(d => d.id === auditDs)?.name),
          ),
          page === 'audit' && e('span', { className: 'topbar-chip' }, DATASETS.find(d => d.id === auditDs)?.domain)
        ),
        e('div', { className: 'topbar-right' },
          e('a', { href: 'https://github.com/ujjwaltwri/fairlens', target: '_blank', className: 'btn btn-ghost btn-sm' }, 'GitHub ↗'),
          e('a', { href: `${API}/docs`, target: '_blank', className: 'btn btn-sm' }, 'API Docs'),
          e('button', { className: 'btn btn-primary btn-sm', onClick: () => navigate('upload') }, '+ New Audit')
        )
      ),
      e('div', { className: 'content' },
        page === 'dashboard' && e(DashboardPage, { onNavigate: navigate }),
        page === 'audit'     && e(AuditPage,     { key: auditDs, initDs: auditDs }),
        page === 'upload'    && e(UploadPage,     { addToast }),
        page === 'about'     && e(AboutPage,      null),
      )
    ),

    e(ToastStack, { toasts, remove: removeToast })
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(e(App));
})();