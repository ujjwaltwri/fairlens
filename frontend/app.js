/**
 * FairLens — AI Bias Detection & Mitigation Platform
 * app.js — React 18 (createElement, no build step required)
 * Theme: Precision Instrument — DM Serif Display + DM Sans + Spline Sans Mono
 */
(function () {
'use strict';

const { useState, useEffect, useRef, useCallback } = React;
const e = React.createElement;

/* ═══════════════════════════════
   CONSTANTS
═══════════════════════════════ */
const API = 'http://localhost:8000';

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
      { f: 'education_num',    v: 3.115, p: false },
      { f: 'capital_gain',     v: 1.224, p: false },
      { f: 'hours_per_week',   v: 1.128, p: false },
      { f: 'sex (protected)',  v: 0.951, p: true  },
      { f: 'race (protected)', v: 0.770, p: true  },
      { f: 'marital_status',   v: 0.163, p: false },
      { f: 'occupation',       v: 0.090, p: false },
    ],
    mit: [
      { n: 'Pre-processing',  d: 'Reweighing sample weights',                     icon: '01', before: 32, after: 32, imp: 0,  best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds',         icon: '02', before: 32, after: 22, imp: 10, best: true  },
      { n: 'Post-processing', d: 'Per-group threshold calibration',               icon: '03', before: 32, after: 32, imp: 0,  best: false },
    ],
    recs: [
      'Apply reweighing to fix disparate impact on gender — DIR = 0.50 fails the US EEOC four-fifths rule (< 0.80)',
      'Gender is statistically associated with outcome at p < 0.001 — investigate proxy features like marital_status',
      'TPR gap of 0.134 detected — apply equalized odds post-processing before deployment',
      'MEDIUM RISK: Apply in-processing fairness constraint; bias score reduces by 10 pts',
    ],
  },
  compas: {
    dataScore: 55, dataSev: 'MEDIUM', modelScore: 28, modelSev: 'LOW',
    dir: 0.61, dpg: 0.14, tprGap: 0.08, fprGap: 0.04, flipRate: 0.04,
    acc: 0.997, auc: 0.922, posPriv: 39.4, posUnpriv: 52.5,
    shap: [
      { f: 'age',              v: 1.18, p: false },
      { f: 'race (protected)', v: 0.74, p: true  },
      { f: 'priors_count',     v: 0.61, p: false },
      { f: 'charge_degree',    v: 0.60, p: false },
      { f: 'age_cat_enc',      v: 0.50, p: false },
      { f: 'days_b_screening', v: 0.28, p: false },
    ],
    mit: [
      { n: 'Pre-processing',  d: 'Reweighing sample weights',             icon: '01', before: 20, after: 20, imp: 0, best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds', icon: '02', before: 20, after: 14, imp: 6, best: true  },
      { n: 'Post-processing', d: 'Per-group threshold calibration',       icon: '03', before: 20, after: 20, imp: 0, best: false },
    ],
    recs: [
      'Racial bias confirmed in COMPAS dataset (DIR = 0.61) — African-American defendants score higher risk',
      'Race ranks #2 by SHAP importance — model directly uses protected attribute for decisions',
      'False positive rates differ significantly across racial groups — review deployment ethics',
      'Monitor re-audit quarterly; apply EqualizedOdds constraint for improved fairness',
    ],
  },
  german: {
    dataScore: 60, dataSev: 'MEDIUM', modelScore: 50, modelSev: 'MEDIUM',
    dir: 1.54, dpg: 0.285, tprGap: 0.1478, fprGap: 0.1308, flipRate: 0.028,
    acc: 0.784, auc: 0.882, posPriv: 76.8, posUnpriv: 55.4,
    shap: [
      { f: 'age (protected)', v: 1.280, p: true  },
      { f: 'credit_amount',   v: 1.070, p: false },
      { f: 'personal_status', v: 0.800, p: false },
      { f: 'duration',        v: 0.650, p: false },
      { f: 'purpose',         v: 0.420, p: false },
      { f: 'savings_status',  v: 0.210, p: false },
    ],
    mit: [
      { n: 'Pre-processing',  d: 'Reweighing sample weights',             icon: '01', before: 48, after: 48, imp: 0, best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds', icon: '02', before: 48, after: 44, imp: 4, best: true  },
      { n: 'Post-processing', d: 'Per-group threshold calibration',       icon: '03', before: 48, after: 48, imp: 0, best: false },
    ],
    recs: [
      'Age-based discrimination detected — younger applicants under 25 systematically disadvantaged',
      'Age is the top SHAP feature driving loan rejections — direct age discrimination risk',
      'TPR gap of 0.148 — younger applicants miss out on loans they would repay at higher rate',
      'Apply equalized odds constraint to reduce age disparity; consider removing age as direct feature',
    ],
  },
  utrecht: {
    dataScore: 51, dataSev: 'MEDIUM', modelScore: 22, modelSev: 'LOW',
    dir: 0.88, dpg: 0.117, tprGap: 0.05, fprGap: 0.02, flipRate: 0.02,
    acc: 0.938, auc: 0.793, posPriv: 52.3, posUnpriv: 40.6,
    shap: [
      { f: 'gender (protected)', v: 0.70, p: true  },
      { f: 'skill_score',        v: 0.28, p: false },
      { f: 'interview_score',    v: 0.28, p: false },
      { f: 'education',          v: 0.22, p: false },
      { f: 'experience',         v: 0.16, p: false },
    ],
    mit: [
      { n: 'Pre-processing',  d: 'Reweighing sample weights',             icon: '01', before: 10, after: 10, imp: 0, best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds', icon: '02', before: 10, after: 10, imp: 0, best: false },
      { n: 'Post-processing', d: 'Per-group threshold calibration',       icon: '03', before: 10, after: 10, imp: 0, best: false },
    ],
    recs: [
      'Gender is top SHAP feature — investigate whether gender is leaking into the model directly',
      'Demographic parity gap of 11.7% indicates male candidates hired at higher base rate',
      'LOW RISK: Model bias is low but data bias warrants monitoring',
      'Re-audit with a larger external dataset; consider removing gender from feature set entirely',
    ],
  },
  diabetes130: {
    dataScore: 97, dataSev: 'HIGH', modelScore: 83, modelSev: 'HIGH',
    dir: 0.4325, dpg: 0.1887, tprGap: 0.3299, fprGap: 0.1218, flipRate: 0.105,
    acc: 0.8344, auc: 0.810, posPriv: 14.4, posUnpriv: 6.2,
    shap: [
      { f: 'num_medications',  v: 0.780, p: false },
      { f: 'number_diagnoses', v: 0.620, p: false },
      { f: 'race (protected)', v: 0.250, p: true  },
      { f: 'age_num',          v: 0.250, p: false },
      { f: 'A1Cresult',        v: 0.210, p: false },
      { f: 'time_in_hospital', v: 0.180, p: false },
    ],
    mit: [
      { n: 'Pre-processing',  d: 'Reweighing sample weights',             icon: '01', before: 53, after: 53, imp: 0,  best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds', icon: '02', before: 53, after: 5,  imp: 48, best: true  },
      { n: 'Post-processing', d: 'Per-group threshold calibration',       icon: '03', before: 53, after: 53, imp: 0,  best: false },
    ],
    recs: [
      'HIGH RISK — Do not deploy without applying bias mitigation first',
      'DIR = 0.43 (far below 0.80 threshold) — Black patients systematically under-predicted for readmission',
      'TPR gap 0.33 — African-American patients miss early intervention at 33% higher rate',
      'Counterfactual flip rate 10.5% — model directly relies on race attribute for predictions',
      'In-processing fairness constraint reduces bias score from 83 to 5 (improvement: −48 pts)',
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
  return e('span', { className: `badge badge-${type}` }, label);
}

function Spinner() { return e('div', { className: 'spinner' }); }

/* ═══════════════════════════════
   SHAP ROW — extracted to fix Hook-in-map bug
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
        'FairLens audits machine learning systems across gender, race, and age using statistical bias analysis, SHAP explainability, and three automated mitigation strategies — before discriminatory decisions reach production.'
      ),
      e('div', { className: 'hero-actions' },
        e('button', { className: 'btn btn-primary btn-lg', onClick: () => onNavigate('audit') }, 'Explore Audits →'),
        e('button', { className: 'btn btn-lg', onClick: () => onNavigate('upload') }, 'Upload Dataset'),
        e('button', { className: 'btn btn-ghost btn-lg', onClick: () => onNavigate('about') }, 'API Docs'),
      ),
      e('div', { className: 'hero-meta' },
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num' }, '5'),
          e('div', { className: 'hero-meta-label' }, 'datasets audited')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num', style: { color: biasColor(aBias) } }, aBias),
          e('div', { className: 'hero-meta-label' }, 'avg data bias /100')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num', style: { color: 'var(--red)' } }, aHigh),
          e('div', { className: 'hero-meta-label' }, 'high-risk datasets')
        ),
        e('div', { className: 'hero-meta-divider' }),
        e('div', { className: 'hero-meta-item' },
          e('div', { className: 'hero-meta-num' }, '7'),
          e('div', { className: 'hero-meta-label' }, 'fairness metrics tracked')
        ),
      )
    ),
    e('div', { className: 'card' },
      e('div', { className: 'card-head' },
        e('span', { className: 'card-title' }, 'Dataset Registry'),
        e('button', { className: 'btn btn-sm', onClick: () => onNavigate('audit') }, 'Open Audit Explorer →')
      ),
      e('div', { className: 'tbl-wrap' },
        e('table', null,
          e('thead', null,
            e('tr', null,
              ['Dataset', 'Domain', 'Protected attributes', 'Rows', 'Data bias', 'Model bias', 'Severity', ''].map(h =>
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
          e('span', { className: 'mono', style: { fontSize: 11, color: 'var(--ink-4)' } }, ds.protected + ' · ' + ds.rows + ' rows')
        )
      )
    ),
    e('div', { className: 'grid-2 mb-20' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Data bias score'),
          e(Badge, { label: d.dataSev, type: sevClass(d.dataSev) })
        ),
        e('div', { className: 'card-body' },
          e(Gauge, { score: d.dataScore, severity: d.dataSev }),
          e('div', { className: 'divider' }),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Disparate impact ratio'),
            e(BiasBar, { value: (1 - Math.min(d.dir, 1)) * 100, color: dirColor(d.dir) }),
            e('span', { className: 'mrow-val', style: { color: dirColor(d.dir) } }, d.dir.toFixed(4)),
            d.dir >= 0.8 ? e(Badge, { label: 'Pass', type: 'pass' }) : e(Badge, { label: 'Fail', type: 'fail' })
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Demographic parity gap'),
            e(BiasBar, { value: d.dpg * 200, color: gapColor(d.dpg) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.dpg) } }, d.dpg.toFixed(4)),
          ),
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Model bias score'),
          e(Badge, { label: d.modelSev, type: sevClass(d.modelSev) })
        ),
        e('div', { className: 'card-body' },
          e(Gauge, { score: d.modelScore, severity: d.modelSev }),
          e('div', { className: 'divider' }),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'TPR gap (equalized odds)'),
            e(BiasBar, { value: d.tprGap * 300, color: gapColor(d.tprGap) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.tprGap) } }, d.tprGap.toFixed(4)),
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'FPR gap'),
            e(BiasBar, { value: d.fprGap * 300, color: gapColor(d.fprGap) }),
            e('span', { className: 'mrow-val', style: { color: gapColor(d.fprGap) } }, d.fprGap.toFixed(4)),
          ),
          e('div', { className: 'mrow' },
            e('span', { className: 'mrow-name' }, 'Counterfactual flip rate'),
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
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Statistical Data Audit'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'Bias analysis on raw dataset distributions — measured before any model is trained.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Fairness metrics')),
        e('div', { className: 'card-body' },
          [
            { name: 'Disparate impact ratio',  val: d.dir,   fmt: v => v.toFixed(4), pass: v => v >= 0.8, note: '< 0.80 = FAIL · EEOC four-fifths rule' },
            { name: 'Demographic parity gap',  val: d.dpg,   fmt: v => v.toFixed(4), pass: v => v < 0.1,  note: '> 0.10 = concerning' },
            { name: 'Chi-square significance', val: 0.001,   fmt: () => 'p < 0.001', pass: () => false,   note: 'Statistically significant association' },
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
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Outcome rate by group')),
        e('div', { className: 'card-body' },
          ['Privileged', 'Unprivileged'].map((g, i) => {
            const rate  = i === 0 ? d.posPriv : d.posUnpriv;
            const color = i === 0 ? 'var(--amber)' : 'var(--green)';
            return e('div', { key: g, style: { marginBottom: 24 } },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', marginBottom: 8 } },
                e('span', { style: { fontSize: 13, color: 'var(--ink-1)', fontWeight: 600 } }, g + ' group'),
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
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Model Fairness Audit'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'LogisticRegression trained on dataset — SHAP feature attribution + post-training parity metrics.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Performance & fairness')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Model performance'),
          [
            { label: 'Accuracy', val: d.acc, fmt: v => (v * 100).toFixed(2) + '%' },
            { label: 'ROC-AUC',  val: d.auc, fmt: v => v.toFixed(4) },
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
          e('div', { className: 'section-label' }, 'Fairness metrics'),
          [
            { label: 'TPR gap (equalized odds)', val: d.tprGap,   fmt: v => v.toFixed(4),           bad: v => v > 0.1 },
            { label: 'FPR gap',                  val: d.fprGap,   fmt: v => v.toFixed(4),           bad: v => v > 0.05 },
            { label: 'Counterfactual flip rate',  val: d.flipRate, fmt: v => (v*100).toFixed(1)+'%', bad: v => v > 0.05 },
            { label: 'Model DIR',                 val: d.dir,      fmt: v => v.toFixed(4),           bad: v => v < 0.8 },
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
          e('span', { className: 'card-title' }, 'SHAP feature importance'),
          e('span', { className: 'card-sub' }, '⚑ = protected attribute')
        ),
        e('div', { className: 'card-body' },
          e('div', { className: 'shap-list' },
            /* FIX: use ShapRow component instead of inlining hooks inside .map() */
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
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Mitigation Engine'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'Three algorithmic fairness strategies — before, during, and after training.')
    ),
    d.mit.map((m, i) =>
      e('div', { key: i, className: `mit-item ${m.best ? 'best' : ''}` },
        e('div', { className: 'mit-icon-wrap' }, m.icon),
        e('div', { className: 'mit-info' },
          e('div', { className: 'mit-name' },
            m.n,
            m.best && e(Badge, { label: 'Optimal', type: 'accent' })
          ),
          e('div', { className: 'mit-desc' }, m.d),
          e('div', { className: 'mit-progress-row' },
            e('div', { className: 'mit-progress-label' }, 'After:'),
            e('div', { style: { flex: 1, height: 6, background: 'var(--paper3)', borderRadius: 3, overflow: 'hidden', margin: '0 10px' } },
              e('div', { style: { height: '100%', width: `${m.after}%`, background: m.best ? 'var(--green)' : 'var(--ink-5)', borderRadius: 3, transition: 'width 1.1s cubic-bezier(0.22, 1, 0.36, 1)' } })
            ),
            e('div', { className: 'mit-progress-label' }, m.after + ' / 100')
          )
        ),
        e('div', { className: 'mit-score-block' },
          e('div', { className: 'mit-score-before' }, 'Before: ' + m.before),
          e('div', { className: 'mit-score-after', style: { color: m.after < m.before ? 'var(--green)' : 'var(--ink-2)' } }, m.after),
          e('span', { className: `mit-delta ${m.imp > 0 ? 'delta-pos' : 'delta-zero'}` },
            m.imp > 0 ? '−' + m.imp + ' pts' : 'No change'
          )
        )
      )
    ),
    e('div', { className: 'card', style: { marginTop: 16 } },
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Algorithm reference')),
      e('div', { className: 'card-body' },
        e('div', { className: 'grid-3' },
          [
            { icon: '01', n: 'Pre-processing',  d: 'Assigns sample weights to underrepresented group/label combinations so the model trains on a more balanced view of the data. Uses IBM AIF360 Reweighing.' },
            { icon: '02', n: 'In-processing',   d: 'ExponentiatedGradient (Fairlearn) adds an EqualizedOdds constraint to the training objective, preventing the optimizer from trading group parity for accuracy.' },
            { icon: '03', n: 'Post-processing', d: 'Per-group decision thresholds are calibrated to equalize true positive rates across groups after training. No model retraining required.' },
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
      e('h2', { style: { fontFamily: 'var(--serif)', fontSize: 24, fontWeight: 400, marginBottom: 4, color: 'var(--ink-1)', letterSpacing: '-0.5px' } }, 'Generate Reports'),
      e('p', { style: { fontSize: 13, color: 'var(--ink-4)', fontFamily: 'var(--mono)' } }, 'Export compliance-ready PDF documentation and machine-readable JSON.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Report summary')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Dataset metadata'),
          e('div', { className: 'info-grid', style: { marginBottom: 20 } },
            [
              { k: 'Dataset',  v: ds.name },       { k: 'Domain',    v: ds.domain },
              { k: 'Rows',     v: ds.rows },        { k: 'Protected', v: ds.protected },
              { k: 'Target',   v: 'Binary outcome'}, { k: 'Model',    v: 'LogisticRegression' },
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
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Download center')),
        e('div', { className: 'card-body' },
          e('a', { href: `${API}/report/${sel}/pdf`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '↓'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'PDF Audit Report'),
              e('div', { className: 'report-dl-desc' }, 'Full metrics, SHAP charts & compliance recommendations'),
            ),
            e('span', { className: 'tag' }, 'PDF')
          ),
          e('a', { href: `${API}/report/${sel}/json`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '{}'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'JSON Audit Data'),
              e('div', { className: 'report-dl-desc' }, 'Machine-readable output for CI/CD integration'),
            ),
            e('span', { className: 'tag' }, 'JSON')
          ),
          e('a', { href: `${API}/audit/full/${sel}`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '▶'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'Re-run Pipeline via API'),
              e('div', { className: 'report-dl-desc' }, 'Trigger full audit and mitigation via REST endpoint'),
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
  const d  = AUDIT[sel];
  const ds = DATASETS.find(x => x.id === sel);

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'dataaudit',  label: 'Data Audit' },
    { id: 'modelaudit', label: 'Model Audit' },
    { id: 'mitigation', label: 'Mitigation' },
    { id: 'report',     label: 'Export' },
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
          e('div', { className: 'ds-pill-rows' }, x.rows + ' rows')
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
    e('div', { key: sel + tab },
      tab === 'overview'   && e(OverviewTab,   { d, ds }),
      tab === 'dataaudit'  && e(DataAuditTab,  { d }),
      tab === 'modelaudit' && e(ModelAuditTab, { d }),
      tab === 'mitigation' && e(MitigationTab, { d }),
      tab === 'report'     && e(ReportTab,     { d, ds, sel }),
    )
  );
}

/* ═══════════════════════════════
   UPLOAD PAGE — Fully fixed dropdown mapping
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

  // Stop polling on unmount
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  // Poll job status when jobId is set
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
          addToast('Audit pipeline failed — check server logs', 'err');
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
      addToast('Only .csv files are supported', 'err');
    }
  };

  const submit = async () => {
    if (!file || !target.trim() || !prot.trim()) {
      addToast('Please select a CSV and fill both column fields.', 'err');
      return;
    }
    setLoading(true);
    setResult(null);
    setValErrors([]);
    setJobId(null);
    setJobStatus(null);

    let finalTarget = target.trim();
    let finalProt = prot.trim();

    // Exact string match from columns array to prevent 422 errors due to case sensitivity
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
        const errs = detail.errors || [detail.message || `HTTP ${res.status}`];
        const cols = detail.available_columns || [];
        setValErrors(errs);
        if (cols.length) setColumns(cols);
        addToast(`${errs.length} validation error(s) — see details below`, 'err');
        setLoading(false);
        return;
      }

      setJobId(data.job_id);
      setJobStatus('pending');
      addToast('Upload accepted — auditing in background…', 'ok');

    } catch (err) {
      addToast('Could not reach API — is the server running?', 'err');
      setLoading(false);
    }
  };

  const statusLabel = () => {
    if (jobStatus === 'pending')  return 'Queued — waiting to start…';
    if (jobStatus === 'running')  return 'Running — data audit · model training · mitigation…';
    if (jobStatus === 'done')     return 'Complete';
    if (jobStatus === 'failed')   return 'Pipeline failed';
    return 'Processing…';
  };

  return e('div', { className: 'page', style: { position: 'relative' } },
    loading && e('div', { className: 'overlay' },
      e(Spinner),
      e('div', { className: 'overlay-label' }, 'Auditing Dataset'),
      e('div', { className: 'overlay-sub' }, statusLabel())
    ),

    e('div', { className: 'hero mb-28', style: { padding: '36px 40px' } },
      e('div', { className: 'hero-eyebrow' }, e('span', { className: 'hero-eyebrow-dot' }), 'Custom Audit'),
      e('h2', { className: 'hero-title', style: { fontSize: 36 } }, 'Upload your own dataset'),
      e('p', { className: 'hero-desc', style: { marginBottom: 0 } },
        'FairLens will detect biases, train a baseline model, apply three mitigation strategies, and generate compliance reports for any binary classification dataset.'
      )
    ),

    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-body', style: { display: 'flex', flexDirection: 'column', gap: 18 } },

          // Drop zone
          e('div', {
            className: `drop-zone ${drag ? 'drag-over' : ''}`,
            onDragOver:  ev => { ev.preventDefault(); setDrag(true); },
            onDragLeave: () => setDrag(false),
            onDrop:      ev => { ev.preventDefault(); setDrag(false); handleFile(ev.dataTransfer.files[0]); },
            onClick:     () => fileRef.current.click(),
          },
            e('input', { type: 'file', ref: fileRef, accept: '.csv', style: { display: 'none' }, onChange: ev => handleFile(ev.target.files[0]) }),
            e('span', { className: 'drop-icon' }, file ? '✓' : '↑'),
            e('div', { className: 'drop-title' }, file ? file.name : 'Drop CSV file here'),
            e('div', { className: 'drop-sub' }, file ? `${(file.size / 1024).toFixed(1)} KB · ${columns.length} columns detected` : 'or click to browse · CSV only')
          ),

          // Visual column readout (Read Only chips)
          columns.length > 0 && e('div', null,
            e('div', { className: 'field-label', style: { marginBottom: 8 } }, 'Detected columns:'),
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

          // Dropdown selects for Target and Protected columns
          e('div', { className: 'grid-2', style: { gap: 14 } },
            e('div', null,
              e('label', { className: 'field-label' }, 'Target Column'),
              columns.length > 0 
              ? e('select', {
                  className: 'field-input',
                  value: target,
                  onChange: ev => setTarget(ev.target.value),
                  style: target ? { borderColor: 'var(--green)', boxShadow: '0 0 0 3px var(--green-glow)', cursor: 'pointer' } : { cursor: 'pointer' }
                },
                  e('option', { value: '', disabled: true }, 'Select target column...'),
                  columns.map(c => e('option', { key: c, value: c }, c))
                )
              : e('input', {
                  type: 'text', className: 'field-input',
                  placeholder: 'e.g. income, hired, label',
                  value: target,
                  onChange: ev => setTarget(ev.target.value),
                  style: target ? { borderColor: 'var(--green)', boxShadow: '0 0 0 3px var(--green-glow)' } : {}
                })
            ),
            e('div', null,
              e('label', { className: 'field-label' }, 'Protected Column'),
              columns.length > 0
              ? e('select', {
                  className: 'field-input',
                  value: prot,
                  onChange: ev => setProt(ev.target.value),
                  style: prot ? { borderColor: 'var(--amber)', boxShadow: '0 0 0 3px rgba(184,96,10,0.1)', cursor: 'pointer' } : { cursor: 'pointer' }
                },
                  e('option', { value: '', disabled: true }, 'Select protected column...'),
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

          // Validation errors
          valErrors.length > 0 && e('div', {
            style: {
              padding: '12px 16px', background: 'var(--red-dim)',
              borderRadius: 'var(--r2)', border: '1px solid var(--red-border)',
            }
          },
            e('div', { style: { fontWeight: 700, fontSize: 12, color: 'var(--red)', marginBottom: 6, fontFamily: 'var(--mono)' } }, '✗ Validation failed'),
            valErrors.map((err, i) =>
              e('div', { key: i, style: { fontSize: 12, color: 'var(--red)', fontFamily: 'var(--mono)', marginTop: 3, lineHeight: 1.5 } }, err)
            )
          ),

          // Job status bar (shown while polling)
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
          }, loading ? 'Auditing…' : 'Run Full Bias Audit')
        )
      ),

      // Results panel
      result
        ? e('div', { className: 'card accent-card', style: { animation: 'page-in .3s ease' } },
            e('div', { className: 'card-head' },
              e('span', { className: 'card-title' }, 'Audit Results'),
              e(Badge, { label: result.data_severity || 'MEDIUM', type: sevClass(result.data_severity || 'MEDIUM') })
            ),
            e('div', { className: 'card-body' },
              e('div', { className: 'result-score-grid' },
                [
                  { k: 'Rows analysed',    v: result.data_audit && result.data_audit.n_rows ? result.data_audit.n_rows.toLocaleString() : '—' },
                  { k: 'Data bias score',  v: result.data_bias_score != null ? `${result.data_bias_score}/100` : '—' },
                  { k: 'Model bias score', v: result.model_bias_score != null ? `${result.model_bias_score}/100` : '—' },
                  { k: 'Data severity',    v: result.data_severity || '—' },
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
                e('div', { style: { fontWeight: 600, fontSize: 12, color: 'var(--ink-4)', marginBottom: 6, fontFamily: 'var(--mono)', letterSpacing: 1 } }, 'AI NARRATIVE'),
                e('div', { style: { fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.7 } }, result.gemini_narrative)
              ),
              e('div', { style: { padding: '14px 18px', background: 'var(--green-dim)', borderRadius: 'var(--r2)', border: '1px solid var(--green-border)', margin: '14px 0' } },
                e('div', { style: { fontWeight: 600, fontSize: 13, color: 'var(--green)', marginBottom: 4 } }, '✦ Reports generated'),
                e('div', { style: { fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.6, fontFamily: 'var(--mono)' } }, 'PDF and JSON reports are available via the links below.')
              ),
              result.report_pdf_url && e('a', { href: result.report_pdf_url, target: '_blank', className: 'btn btn-primary', style: { marginRight: 10 } }, '↓ PDF Report'),
              result.report_json_url && e('a', { href: result.report_json_url, target: '_blank', className: 'btn' }, '{} JSON Report'),
              !result.report_pdf_url && e('a', { href: `${API}/docs`, target: '_blank', className: 'btn btn-ghost' }, 'View API Documentation ↗')
            )
          )
        : e('div', { className: 'card' },
            e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'What FairLens checks')),
            e('div', { className: 'card-body' },
              [
                { icon: '01', title: 'Disparate impact ratio',      desc: 'P(positive|unprivileged) / P(positive|privileged) — flagged if below 0.80 (EEOC four-fifths rule)' },
                { icon: '02', title: 'Demographic parity gap',       desc: 'Absolute difference in positive prediction rates across protected groups' },
                { icon: '03', title: 'Chi-square significance test', desc: 'Statistical significance of association between protected attribute and outcome' },
                { icon: '04', title: 'Equalized odds',               desc: 'Equal TPR and FPR across groups — computed via SHAP and model audit' },
                { icon: '05', title: 'Three mitigation strategies',  desc: 'Pre-, in-, and post-processing fairness fixes with before/after comparison' },
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
        'End-to-end AI bias detection and mitigation built for the Hack2Skill Unbiased AI Decision challenge. Validated across 5 real-world fairness benchmarks using IBM AIF360, Microsoft Fairlearn, and SHAP explainability.'
      )
    ),
    e('div', { className: 'grid-2 mb-20' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Fairness metrics reference')),
        e('div', { className: 'tbl-wrap' },
          e('table', null,
            e('thead', null, e('tr', null, ['Metric', 'Threshold', 'Standard'].map(h => e('th', { key: h }, h)))),
            e('tbody', null,
              [
                { m: 'Disparate impact ratio',   t: '< 0.80 = FAIL',         s: 'EEOC four-fifths rule' },
                { m: 'Demographic parity gap',   t: '> 0.10 = concerning',    s: 'ML fairness literature' },
                { m: 'Equalized odds (TPR gap)', t: '> 0.05 = concerning',    s: 'Hardt et al., 2016' },
                { m: 'Equal opportunity (FPR)',  t: '> 0.05 = concerning',    s: 'Hardt et al., 2016' },
                { m: 'Counterfactual flip rate', t: '> 10% = high reliance',  s: 'Counterfactual fairness' },
                { m: 'SHAP feature importance',  t: 'Protected rank ≤ 5 = flag', s: 'Lundberg & Lee, 2017' },
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
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Technology stack')),
        e('div', { className: 'card-body' },
          [
            { n: 'IBM AIF360',          r: 'Fairness metrics + Reweighing mitigation' },
            { n: 'Microsoft Fairlearn', r: 'ExponentiatedGradient with EqualizedOdds' },
            { n: 'SHAP',               r: 'Model explainability and feature attribution' },
            { n: 'scikit-learn',       r: 'LogisticRegression, RandomForestClassifier' },
            { n: 'FastAPI + slowapi',  r: 'REST API backend + rate limiting' },
            { n: 'Supabase',           r: 'Postgres DB + Storage for jobs & reports' },
            { n: 'Google Gemini',      r: 'AI narrative + deployment recommendation' },
            { n: 'ReportLab',          r: 'PDF audit report generation' },
            { n: 'React 18',           r: 'Frontend — CDN, zero build step' },
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
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'REST API reference')),
      e('div', { className: 'tbl-wrap' },
        e('table', null,
          e('thead', null, e('tr', null, ['Method', 'Endpoint', 'Description'].map(h => e('th', { key: h }, h)))),
          e('tbody', null,
            [
              ['GET',  '/',                     'Health check + service info'],
              ['GET',  '/datasets',              'List all available datasets'],
              ['POST', '/audit/data/{name}',     'Statistical data audit on named dataset'],
              ['POST', '/audit/model/{name}',    'Train model + audit for bias'],
              ['POST', '/audit/full/{name}',     'Full pipeline: audit + mitigate + report (async)'],
              ['GET',  '/jobs/{job_id}',         'Poll async job status'],
              ['GET',  '/results/{job_id}',      'Full audit result for a job'],
              ['GET',  '/report/{name}/pdf',     'Redirect to PDF audit report (Supabase Storage)'],
              ['GET',  '/report/{name}/json',    'Redirect to JSON audit report (Supabase Storage)'],
              ['POST', '/audit/upload',          'Upload custom CSV → full pipeline (async)'],
              ['POST', '/chat/{job_id}',         'Gemini Q&A on a completed audit'],
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
    { id: 'dashboard', label: 'Dashboard',      icon: '◈' },
    { id: 'audit',     label: 'Audit Explorer', icon: '◉' },
    { id: 'upload',    label: 'Custom Dataset',  icon: '↑' },
    { id: 'about',     label: 'Docs & API',      icon: '?' },
  ];

  const pageTitle = navLinks.find(n => n.id === page)?.label || 'Dashboard';

  return e('div', { style: { display: 'flex', height: '100vh', overflow: 'hidden' } },
    // SIDEBAR
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

    // MAIN
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