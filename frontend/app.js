/**
 * FairLens — AI Bias Detection & Mitigation Platform
 * app.js — React 18 application (createElement, no build step)
 */
(function () {
'use strict';

const { useState, useEffect, useRef, useCallback } = React;
const e = React.createElement;

/* ═══════════════════════════════════════════
   CONSTANTS
═══════════════════════════════════════════ */
const API = 'http://localhost:8000';

const DATASETS = [
  { id: 'adult',       name: 'UCI Adult',      domain: 'Income Prediction',  protected: 'Gender, Race', rows: '48,842',  severity: 'HIGH',   dataBias: 92, modelBias: 60 },
  { id: 'compas',      name: 'COMPAS',         domain: 'Criminal Justice',   protected: 'Race',         rows: '7,214',   severity: 'MEDIUM', dataBias: 55, modelBias: 28 },
  { id: 'german',      name: 'German Credit',  domain: 'Loan Approval',      protected: 'Age',          rows: '1,000',   severity: 'MEDIUM', dataBias: 60, modelBias: 50 },
  { id: 'utrecht',     name: 'Utrecht Hiring', domain: 'Recruitment',        protected: 'Gender',       rows: '2,000',   severity: 'MEDIUM', dataBias: 51, modelBias: 22 },
  { id: 'diabetes130', name: 'Diabetes 130',   domain: 'Healthcare',         protected: 'Race',         rows: '101,766', severity: 'HIGH',   dataBias: 97, modelBias: 83 },
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
      { n: 'Pre-processing',  d: 'Reweighing sample weights',                   icon: '01', before: 32, after: 32, imp: 0,  best: false },
      { n: 'In-processing',   d: 'ExponentiatedGradient · EqualizedOdds',       icon: '02', before: 32, after: 22, imp: 10, best: true  },
      { n: 'Post-processing', d: 'Per-group threshold calibration',             icon: '03', before: 32, after: 32, imp: 0,  best: false },
    ],
    recs: [
      'Apply reweighing to fix disparate impact on gender — DIR = 0.50 fails the US EEOC four-fifths rule (< 0.80)',
      'Gender is statistically associated with outcome at p < 0.001 — investigate proxy features like marital_status',
      'TPR gap of 0.134 detected — apply equalized odds post-processing before deployment',
      'MEDIUM RISK: Apply in-processing fairness constraint; bias score reduces by 10 pts',
    ],
  },
  compas: {
    dataScore: 55, dataSev: 'MEDIUM',
    modelScore: 28, modelSev: 'LOW',
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
    dataScore: 60, dataSev: 'MEDIUM',
    modelScore: 50, modelSev: 'MEDIUM',
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
    dataScore: 51, dataSev: 'MEDIUM',
    modelScore: 22, modelSev: 'LOW',
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
    dataScore: 97, dataSev: 'HIGH',
    modelScore: 83, modelSev: 'HIGH',
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
      'TPR gap 0.33 — African-American patients miss early intervention at 33% higher rate than Caucasian patients',
      'Counterfactual flip rate 10.5% — model directly relies on race attribute for predictions',
      'In-processing fairness constraint reduces bias score from 83 to 5 (improvement: −48 pts)',
    ],
  },
};

/* ═══════════════════════════════════════════
   UTILS
═══════════════════════════════════════════ */
const sevClass  = s => s === 'HIGH' ? 'badge-high' : s === 'MEDIUM' ? 'badge-med' : 'badge-low';
const sevColor  = s => s === 'HIGH' ? 'var(--danger)' : s === 'MEDIUM' ? 'var(--warn)' : 'var(--ok)';
const biasColor = v => v >= 70 ? 'var(--danger)' : v >= 40 ? 'var(--warn)' : 'var(--ok)';
const dirColor  = v => v >= 0.8 ? 'var(--ok)' : 'var(--danger)';
const gapColor  = v => v > 0.1  ? 'var(--danger)' : v > 0.05 ? 'var(--warn)' : 'var(--ok)';

/* ═══════════════════════════════════════════
   HOOKS
═══════════════════════════════════════════ */
function useCountUp(target, duration = 1200) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = null;
    const tick = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 4); // Quartic ease out for a much smoother slow-down
      setVal(Math.round(ease * target));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target]);
  return val;
}

/* ═══════════════════════════════════════════
   PRIMITIVES
═══════════════════════════════════════════ */
// COMPLETELY REWRITTEN GAUGE: Mathematically perfect dasharray alignment
function Gauge({ score, severity }) {
  const animated = useCountUp(score);
  const color = sevColor(severity);
  
  // Math for a perfect semi-circle
  const R = 80;
  const CX = 100;
  const CY = 90;
  const SW = 12;
  const length = Math.PI * R; // Circumference of half circle = 251.327...
  
  // Calculate dash offset to reveal the path smoothly via CSS
  const pct = Math.max(0.01, score / 100);
  const strokeOffset = length - (length * pct);

  // M = Move to start (left), A = Arc to end (right)
  const pathD = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;

  return e('div', { className: 'gauge-wrap' },
    e('svg', { className: 'gauge-svg', viewBox: '0 0 200 110' },
      // Drop shadow definition for the "wow" glowing gauge effect
      e('defs', null,
        e('filter', { id: 'glow', x: '-20%', y: '-20%', width: '140%', height: '140%' },
          e('feGaussianBlur', { stdDeviation: '4', result: 'blur' }),
          e('feComposite', { in: 'SourceGraphic', in2: 'blur', operator: 'over' })
        )
      ),
      // Background Track
      e('path', { 
        d: pathD, 
        fill: 'none', 
        stroke: 'var(--surface4)', 
        strokeWidth: SW, 
        strokeLinecap: 'round' 
      }),
      // Foreground Colored Track (Animated via CSS strokeDashoffset)
      e('path', { 
        d: pathD, 
        fill: 'none', 
        stroke: color, 
        strokeWidth: SW, 
        strokeLinecap: 'round',
        strokeDasharray: length,
        strokeDashoffset: strokeOffset,
        style: { transition: 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)' },
        filter: 'url(#glow)'
      }),
      // Centered Text
      e('text', { 
        x: CX, 
        y: CY - 6, 
        textAnchor: 'middle', 
        fontFamily: 'Instrument Serif, serif', 
        fontSize: 54, 
        fontWeight: 400, 
        fill: 'var(--t1)', 
        letterSpacing: -2 
      }, animated),
      e('text', { 
        x: CX, 
        y: CY + 18, 
        textAnchor: 'middle', 
        fontFamily: 'JetBrains Mono, monospace', 
        fontSize: 10, 
        fill: 'var(--t3)', 
        letterSpacing: 2,
        fontWeight: 600
      }, 'RISK SCORE')
    )
  );
}

function BiasBar({ value, max = 100, color }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    // Delay animation slightly for dramatic reveal
    const t = setTimeout(() => setW(Math.min(100, (value / max) * 100)), 100);
    return () => clearTimeout(t);
  }, [value, max]);

  return e('div', { className: 'bar-track' },
    e('div', { className: 'bar-fill', style: { width: `${w}%`, background: color || 'var(--accent)' } })
  );
}

function Badge({ label, type = 'muted' }) {
  return e('span', { className: `badge badge-${type}` }, label);
}

function Spinner() {
  return e('div', { className: 'spinner' });
}

/* ═══════════════════════════════════════════
   TOAST
═══════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════
   PAGE: DASHBOARD
═══════════════════════════════════════════ */
function DashboardPage({ onNavigate }) {
  const highCount = DATASETS.filter(d => d.severity === 'HIGH').length;
  const avgBias   = Math.round(DATASETS.reduce((a, d) => a + d.dataBias, 0) / DATASETS.length);
  const avgModel  = Math.round(DATASETS.reduce((a, d) => a + d.modelBias, 0) / DATASETS.length);
  const animHigh  = useCountUp(highCount, 600);
  const animAvg   = useCountUp(avgBias, 1000);
  const animModel = useCountUp(avgModel, 1000);

  return e('div', { className: 'page' },

    // Hero (Enhanced with Glassmorphic Orbs)
    e('div', { className: 'hero mb-24' },
      e('div', { className: 'hero-orb orb-1' }),
      e('div', { className: 'hero-orb orb-2' }),
      e('div', { className: 'hero-content' },
        e('div', { className: 'hero-badge-row' },
          e(Badge, { label: 'Hack2Skill · Unbiased AI', type: 'accent' }),
          e(Badge, { label: '5 datasets audited', type: 'glass' }),
        ),
        e('h1', { className: 'hero-title' },
          'Detect, measure, and eliminate ', e('br'), e('em', null, 'hidden bias'), ' in AI models.'
        ),
        e('p', { className: 'hero-desc' },
          'FairLens provides statistical data auditing, model-level fairness metrics, SHAP explainability, and automated mitigation — across gender, race, and age — before discriminatory systems reach production.'
        ),
        e('div', { className: 'hero-actions' },
          e('button', { className: 'btn btn-primary btn-lg', onClick: () => onNavigate('audit') }, 'Run AI Audit'),
          e('button', { className: 'btn btn-glass btn-lg', onClick: () => onNavigate('upload') }, 'Upload Custom CSV'),
        )
      )
    ),

    // KPI row
    e('div', { className: 'grid-4 mb-24' },
      e('div', { className: 'stat-card' },
        e('div', { className: 'stat-icon-bg' }, '◈'),
        e('div', { className: 'stat-label' }, 'Datasets audited'),
        e('div', { className: 'stat-value', style: { color: 'var(--accent)' } }, '5'),
        e('div', { className: 'stat-detail' }, 'UCI · COMPAS · German · Utrecht · Diabetes'),
      ),
      e('div', { className: 'stat-card' },
        e('div', { className: 'stat-icon-bg' }, '⚑'),
        e('div', { className: 'stat-label' }, 'High risk datasets'),
        e('div', { className: 'stat-value', style: { color: 'var(--danger)' } }, animHigh),
        e('div', { className: 'stat-detail' }, 'Require immediate mitigation'),
      ),
      e('div', { className: 'stat-card' },
        e('div', { className: 'stat-icon-bg' }, '◉'),
        e('div', { className: 'stat-label' }, 'Avg data bias'),
        e('div', { className: 'stat-value', style: { color: biasColor(animAvg) } }, animAvg),
        e('div', { className: 'stat-detail' }, 'Across all 5 datasets / 100'),
      ),
      e('div', { className: 'stat-card' },
        e('div', { className: 'stat-icon-bg' }, '◎'),
        e('div', { className: 'stat-label' }, 'Avg model bias'),
        e('div', { className: 'stat-value', style: { color: biasColor(animModel) } }, animModel),
        e('div', { className: 'stat-detail' }, '7 fairness metrics tracked'),
      ),
    ),

    // Table
    e('div', { className: 'card' },
      e('div', { className: 'card-head' },
        e('span', { className: 'card-title' }, 'Dataset Registry'),
        e('button', { className: 'btn btn-sm', onClick: () => onNavigate('audit') }, 'Explore Audits →')
      ),
      e('div', { className: 'tbl-wrap' },
        e('table', null,
          e('thead', null,
            e('tr', null,
              ['Dataset', 'Domain', 'Protected', 'Rows', 'Data bias', 'Model bias', 'Severity', ''].map(h =>
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
                    e('span', { className: 'mono', style: { color: biasColor(d.dataBias), fontWeight: 600, minWidth: 26 } }, d.dataBias)
                  )
                ),
                e('td', null,
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
                    e(BiasBar, { value: d.modelBias, color: biasColor(d.modelBias) }),
                    e('span', { className: 'mono', style: { color: biasColor(d.modelBias), fontWeight: 600, minWidth: 26 } }, d.modelBias)
                  )
                ),
                e('td', null, e(Badge, { label: d.severity, type: d.severity.toLowerCase().replace('medium','med') })),
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

/* ═══════════════════════════════════════════
   AUDIT SUB-TABS
═══════════════════════════════════════════ */
function OverviewTab({ d, ds }) {
  return e('div', { className: 'fade-enter' },
    e('div', { style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 } },
      e('div', { className: 'ds-icon-lg' }, '◉'),
      e('div', null,
        e('h2', { style: { fontSize: 22, fontWeight: 600, color: 'var(--t1)', letterSpacing: '-0.5px', marginBottom: 2 } }, ds.name),
        e('div', { style: { display: 'flex', gap: 8, alignItems: 'center' } },
          e('span', { className: 'tag' }, ds.domain),
          e('span', { style: { fontSize: 12, color: 'var(--t3)', fontFamily: 'var(--font-mono)' } },
            ds.protected + ' · ' + ds.rows + ' rows'
          )
        )
      )
    ),
    e('div', { className: 'grid-2 mb-20' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'Data bias score'),
          e(Badge, { label: d.dataSev, type: d.dataSev.toLowerCase().replace('medium','med') })
        ),
        e('div', { className: 'card-body', style: { position: 'relative' } },
          e('div', { className: 'gauge-bg-glow', style: { background: sevColor(d.dataSev) } }),
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
          e(Badge, { label: d.modelSev, type: d.modelSev.toLowerCase().replace('medium','med') })
        ),
        e('div', { className: 'card-body', style: { position: 'relative' } },
          e('div', { className: 'gauge-bg-glow', style: { background: sevColor(d.modelSev) } }),
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
    e('div', { className: 'card glass-card accent-border' },
      e('div', { className: 'card-head' }, e('span', { className: 'card-title', style: { color: 'var(--accent)' } }, '✨ AI Recommended Actions')),
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
  return e('div', { className: 'fade-enter' },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontSize: 20, fontWeight: 600, marginBottom: 4, color: 'var(--t1)', letterSpacing: '-0.4px' } }, 'Statistical Data Audit'),
      e('p', { style: { fontSize: 13, color: 'var(--t3)' } }, 'Deep statistical bias analysis on raw dataset distributions — measured before any model is trained.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Fairness metrics')),
        e('div', { className: 'card-body' },
          [
            { name: 'Disparate impact ratio',  val: d.dir,   fmt: v => v.toFixed(4), pass: v => v >= 0.8, note: '< 0.80 = FAIL · US EEOC four-fifths rule' },
            { name: 'Demographic parity gap',  val: d.dpg,   fmt: v => v.toFixed(4), pass: v => v < 0.1,  note: '> 0.10 = concerning' },
            { name: 'Chi-square significance', val: 0.001,   fmt: () => 'p < 0.001', pass: () => false,   note: 'Statistically significant association confirmed' },
          ].map((m, i) =>
            e('div', { key: i, className: 'mrow' },
              e('div', { style: { flex: 1 } },
                e('div', { style: { fontSize: 14, fontWeight: 600, marginBottom: 4, color: 'var(--t1)' } }, m.name),
                e('div', { style: { fontSize: 11, color: 'var(--t3)', fontFamily: 'var(--font-mono)' } }, m.note),
              ),
              e('div', { style: { textAlign: 'right' } },
                e('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 600, color: m.pass(m.val) ? 'var(--ok)' : 'var(--danger)', marginBottom: 6 } }, m.fmt(m.val)),
                e(Badge, { label: m.pass(m.val) ? 'Pass' : 'Fail', type: m.pass(m.val) ? 'pass' : 'fail' })
              )
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Positive rate by group')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Outcome rate across protected groups'),
          ['Privileged Group', 'Unprivileged Group'].map((g, i) => {
            const rate  = i === 0 ? d.posPriv : d.posUnpriv;
            const color = i === 0 ? 'var(--warn)' : 'var(--accent)';
            return e('div', { key: g, style: { marginBottom: 20 } },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', marginBottom: 8 } },
                e('span', { style: { fontSize: 13, color: 'var(--t1)', fontWeight: 600 } }, g),
                e('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 14, color, fontWeight: 600 } }, rate.toFixed(1) + '%')
              ),
              e('div', { style: { height: 10, background: 'var(--surface3)', borderRadius: 5, overflow: 'hidden' } },
                e('div', { style: { height: '100%', width: `${Math.min(rate, 100)}%`, background: color, borderRadius: 5, transition: 'width 1s cubic-bezier(0.16, 1, 0.3, 1)' } })
              )
            );
          }),
          e('div', { className: 'divider' }),
          e('div', { className: 'verdict-cell' },
            e('div', { className: 'verdict-cell-label' }, 'Overall data bias score'),
            e('div', { className: 'verdict-cell-value', style: { color: sevColor(d.dataSev) } }, d.dataScore),
            e(Badge, { label: d.dataSev, type: d.dataSev.toLowerCase().replace('medium','med') })
          )
        )
      )
    )
  );
}

function ModelAuditTab({ d }) {
  const maxShap = d.shap[0].v;
  return e('div', { className: 'fade-enter' },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontSize: 20, fontWeight: 600, marginBottom: 4, color: 'var(--t1)', letterSpacing: '-0.4px' } }, 'Model Fairness Audit'),
      e('p', { style: { fontSize: 13, color: 'var(--t3)' } }, 'Analysis of the trained LogisticRegression model, including SHAP feature explainability and post-training parity metrics.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Performance & fairness')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Model performance'),
          [
            { label: 'Accuracy', val: d.acc, fmt: v => (v * 100).toFixed(2) + '%', color: 'var(--accent)' },
            { label: 'ROC-AUC',  val: d.auc, fmt: v => v.toFixed(4),              color: 'var(--accent)' },
          ].map(m =>
            e('div', { key: m.label, style: { marginBottom: 14 } },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', marginBottom: 6 } },
                e('span', { style: { fontSize: 13, color: 'var(--t1)', fontWeight: 500 } }, m.label),
                e('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: m.color } }, m.fmt(m.val))
              ),
              e('div', { style: { height: 6, background: 'var(--surface3)', borderRadius: 3, overflow: 'hidden' } },
                e('div', { style: { height: '100%', width: `${m.val * 100}%`, background: m.color, borderRadius: 3, transition: 'width 1s cubic-bezier(0.16, 1, 0.3, 1)' } })
              )
            )
          ),
          e('div', { className: 'divider' }),
          e('div', { className: 'section-label' }, 'Fairness metrics'),
          [
            { label: 'TPR gap (equalized odds)', val: d.tprGap,   fmt: v => v.toFixed(4),          bad: v => v > 0.1 },
            { label: 'FPR gap',                  val: d.fprGap,   fmt: v => v.toFixed(4),          bad: v => v > 0.05 },
            { label: 'Counterfactual flip rate',  val: d.flipRate, fmt: v => (v*100).toFixed(1)+'%', bad: v => v > 0.05 },
            { label: 'Model DIR',                 val: d.dir,      fmt: v => v.toFixed(4),          bad: v => v < 0.8 },
          ].map(m =>
            e('div', { key: m.label, className: 'mrow' },
              e('span', { className: 'mrow-name' }, m.label),
              e('span', { className: 'mrow-val', style: { color: m.bad(m.val) ? 'var(--danger)' : 'var(--ok)', fontWeight: 600 } }, m.fmt(m.val)),
              e(Badge, { label: m.bad(m.val) ? 'Fail' : 'Pass', type: m.bad(m.val) ? 'fail' : 'pass' })
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' },
          e('span', { className: 'card-title' }, 'SHAP feature importance'),
          e('span', { className: 'card-sub' }, 'Mean |SHAP|')
        ),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Features driving predictions — ⚑ protected attribute'),
          e('div', { className: 'shap-list' },
            d.shap.map((s, i) => {
              const [w, setW] = useState(0);
              const targetPct = (s.v / maxShap) * 100;
              useEffect(() => { const t = setTimeout(() => setW(targetPct), 100 + (i * 50)); return () => clearTimeout(t); }, [targetPct, i]);
              
              return e('div', { key: s.f, className: 'shap-row' },
                e('span', { className: 'shap-rank' }, i + 1),
                e('span', { className: `shap-feat ${s.p ? 'protected' : 'regular'}` }, (s.p ? '⚑ ' : '') + s.f),
                e('div', { className: 'shap-bar-bg' },
                  e('div', {
                    className: 'shap-bar-fill',
                    style: {
                      width: `${w}%`,
                      background: s.p
                        ? 'var(--danger)'
                        : i < 3
                        ? 'var(--accent)'
                        : 'var(--surface4)'
                    }
                  },
                    w > 28 && e('span', { className: 'shap-inner-val' }, s.v.toFixed(3))
                  )
                ),
                e('span', { className: 'shap-num' }, s.v.toFixed(3))
              );
            })
          )
        )
      )
    )
  );
}

function MitigationTab({ d }) {
  return e('div', { className: 'fade-enter' },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontSize: 20, fontWeight: 600, marginBottom: 4, color: 'var(--t1)', letterSpacing: '-0.4px' } }, 'Mitigation Engine'),
      e('p', { style: { fontSize: 13, color: 'var(--t3)' } }, 'Automated application of three algorithmic fairness strategies: before, during, and after training.')
    ),
    d.mit.map((m, i) =>
      e('div', { key: i, className: `mit-item ${m.best ? 'best' : ''}` },
        e('div', { className: 'mit-icon-wrap' }, m.icon),
        e('div', { className: 'mit-info' },
          e('div', { className: 'mit-name' },
            m.n,
            m.best && e(Badge, { label: 'Optimal Strategy', type: 'accent' })
          ),
          e('div', { className: 'mit-desc' }, m.d),
          e('div', { className: 'mit-progress-row' },
            e('div', { className: 'mit-progress-label' }, 'Score after mitigation:'),
            e('div', { style: { flex: 1, height: 8, background: 'var(--surface3)', borderRadius: 4, overflow: 'hidden', margin: '0 12px' } },
              e('div', { style: { height: '100%', width: `${m.after}%`, background: m.best ? 'var(--accent)' : 'var(--surface4)', borderRadius: 4, transition: 'width 1.2s cubic-bezier(0.16, 1, 0.3, 1)' } })
            ),
            e('div', { className: 'mit-progress-label' }, m.after + ' / 100')
          )
        ),
        e('div', { className: 'mit-score-block' },
          e('div', { className: 'mit-score-before' }, 'Before: ' + m.before),
          e('div', { className: 'mit-score-after', style: { color: m.after < m.before ? 'var(--ok)' : 'var(--t2)' } }, m.after),
          e('span', { className: `mit-delta ${m.imp > 0 ? 'delta-pos' : 'delta-zero'}` },
            m.imp > 0 ? '−' + m.imp + ' pts' : 'No change'
          )
        )
      )
    ),
    e('div', { className: 'card', style: { marginTop: 16 } },
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'How the algorithms work')),
      e('div', { className: 'card-body' },
        e('div', { className: 'grid-3' },
          [
            { icon: '01', n: 'Pre-processing',  d: 'Assigns sample weights to underrepresented group/label combinations so the model trains on a more balanced view of the data. Uses IBM AIF360 Reweighing when available.' },
            { icon: '02', n: 'In-processing',   d: 'ExponentiatedGradient (Fairlearn) adds an EqualizedOdds constraint to the training objective, preventing the optimizer from trading group parity for accuracy.' },
            { icon: '03', n: 'Post-processing', d: 'After training, per-group decision thresholds are calibrated to equalize true positive rates across groups. No model retraining required.' },
          ].map(s =>
            e('div', { key: s.n, style: { padding: 20, background: 'var(--surface2)', borderRadius: 'var(--r2)', border: '1px solid var(--line)' } },
              e('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--t4)', marginBottom: 10, letterSpacing: 1, fontWeight: 600 } }, s.icon),
              e('div', { style: { fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--t1)', letterSpacing: '-0.2px' } }, s.n),
              e('div', { style: { fontSize: 13, color: 'var(--t3)', lineHeight: 1.6 } }, s.d),
            )
          )
        )
      )
    )
  );
}

function ReportTab({ d, ds, sel }) {
  return e('div', { className: 'fade-enter' },
    e('div', { style: { marginBottom: 20 } },
      e('h2', { style: { fontSize: 20, fontWeight: 600, marginBottom: 4, color: 'var(--t1)', letterSpacing: '-0.4px' } }, 'Generate Reports'),
      e('p', { style: { fontSize: 13, color: 'var(--t3)' } }, 'Export compliance-ready PDF documentation and machine-readable JSON data.')
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Report summary')),
        e('div', { className: 'card-body' },
          e('div', { className: 'section-label' }, 'Dataset metadata'),
          e('div', { className: 'info-grid', style: { marginBottom: 24 } },
            [
              { k: 'Dataset',  v: ds.name },       { k: 'Domain', v: ds.domain },
              { k: 'Rows',     v: ds.rows },        { k: 'Protected', v: ds.protected },
              { k: 'Target',   v: 'Binary outcome'}, { k: 'Model', v: 'LogisticRegression' },
            ].map(r =>
              e('div', { key: r.k, className: 'info-cell' },
                e('div', { className: 'info-cell-key' }, r.k),
                e('div', { className: 'info-cell-val' }, r.v)
              )
            )
          ),
          e('div', { className: 'section-label' }, 'Final Verdict'),
          e('div', { style: { display: 'flex', gap: 12 } },
            [
              { lbl: 'Data bias',  score: d.dataScore,  sev: d.dataSev },
              { lbl: 'Model bias', score: d.modelScore, sev: d.modelSev },
            ].map(x =>
              e('div', { key: x.lbl, className: 'verdict-cell', style: { flex: 1 } },
                e('div', { className: 'verdict-cell-label' }, x.lbl),
                e('div', { className: 'verdict-cell-value', style: { color: sevColor(x.sev) } }, x.score),
                e(Badge, { label: x.sev, type: x.sev.toLowerCase().replace('medium','med') })
              )
            )
          )
        )
      ),
      e('div', { className: 'card' },
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Download Center')),
        e('div', { className: 'card-body' },
          e('a', { href: `${API}/report/${sel}/pdf`, target: '_blank', className: 'report-dl-item' },
            e('div', { className: 'report-dl-icon' }, '↓'),
            e('div', { style: { flex: 1 } },
              e('div', { className: 'report-dl-name' }, 'PDF Audit Report'),
              e('div', { className: 'report-dl-desc' }, 'Full metrics, charts & compliance recommendations'),
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
              e('div', { className: 'report-dl-desc' }, 'Trigger full audit and mitigation via REST API'),
            ),
            e('span', { className: 'tag' }, 'POST')
          ),
        )
      )
    )
  );
}

/* ═══════════════════════════════════════════
   PAGE: AUDIT EXPLORER
═══════════════════════════════════════════ */
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
            e(Badge, { label: x.severity, type: x.severity.toLowerCase().replace('medium','med') })
          ),
          e('div', { className: 'ds-pill-sub' }, x.domain),
          e('div', { className: 'ds-pill-rows' }, x.rows + ' rows')
        )
      )
    ),
    e('div', { className: 'card mb-24', style: { padding: '4px 20px', borderRadius: 'var(--r3)' } },
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
    // Use a wrapper to force re-mount and trigger animations
    e('div', { key: sel + tab },
      tab === 'overview'   && e(OverviewTab,   { d, ds }),
      tab === 'dataaudit'  && e(DataAuditTab,  { d }),
      tab === 'modelaudit' && e(ModelAuditTab, { d }),
      tab === 'mitigation' && e(MitigationTab, { d }),
      tab === 'report'     && e(ReportTab,     { d, ds, sel }),
    )
  );
}

/* ═══════════════════════════════════════════
   PAGE: UPLOAD
═══════════════════════════════════════════ */
function UploadPage({ addToast }) {
  const [file,    setFile]    = useState(null);
  const [drag,    setDrag]    = useState(false);
  const [target,  setTarget]  = useState('');
  const [prot,    setProt]    = useState('');
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const fileRef = useRef();

  const handleFile = f => { if (f && f.name.endsWith('.csv')) setFile(f); };

  const submit = async () => {
    if (!file || !target.trim() || !prot.trim()) {
      addToast('Please select a CSV file and fill both column fields.', 'err');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('target_column', target.trim());
      fd.append('protected_column', prot.trim());
      const res  = await fetch(`${API}/audit/upload`, { method: 'POST', body: fd });
      const data = await res.json();
      setResult(data);
      addToast('Audit complete.', 'ok');
    } catch {
      setTimeout(() => {
        setResult({ status: 'ok', n_rows: 1250, data_bias_score: 68, data_severity: 'MEDIUM', model_bias_score: 45, model_severity: 'MEDIUM', tpr_gap: 0.112, fpr_gap: 0.033 });
        addToast('API offline — showing demo result', 'info');
        setLoading(false);
      }, 1500); // Fake delay for demo
      return;
    }
    setLoading(false);
  };

  return e('div', { className: 'page' },
    loading && e('div', { className: 'overlay' },
      e(Spinner),
      e('div', { className: 'overlay-label' }, 'Processing Dataset'),
      e('div', { className: 'overlay-sub' }, 'Running statistical analysis & model training...')
    ),
    e('div', { className: 'hero mb-24' },
      e('div', { className: 'hero-orb orb-1' }),
      e('div', { className: 'hero-content' },
        e('div', { className: 'hero-eyebrow' }, 'Custom Audit'),
        e('h2', { className: 'hero-title', style: { fontSize: 28 } }, 'Upload your own dataset'),
        e('p', { className: 'hero-desc', style: { marginBottom: 0 } },
          'FairLens will automatically detect biases, train a baseline model, apply three mitigation strategies, and generate compliance reports.'
        )
      )
    ),
    e('div', { className: 'grid-2' },
      e('div', { className: 'card' },
        e('div', { className: 'card-body', style: { display: 'flex', flexDirection: 'column', gap: 20 } },
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
            e('div', { className: 'drop-sub' }, file ? `${(file.size / 1024).toFixed(1)} KB — ready to audit` : 'or click to browse · CSV files only')
          ),
          e('div', { className: 'grid-2', style: { gap: 16 } },
            e('div', null,
              e('label', { className: 'field-label' }, 'Target Column'),
              e('input', { type: 'text', className: 'field-input', placeholder: 'e.g. income', value: target, onChange: ev => setTarget(ev.target.value) })
            ),
            e('div', null,
              e('label', { className: 'field-label' }, 'Protected Column'),
              e('input', { type: 'text', className: 'field-input', placeholder: 'e.g. gender', value: prot, onChange: ev => setProt(ev.target.value) })
            )
          ),
          e('button', {
            className: 'btn btn-primary',
            style: { width: '100%', justifyContent: 'center', padding: '14px', fontSize: 14 },
            onClick: submit,
            disabled: loading
          }, 'Run Full Bias Audit')
        )
      ),
      result
        ? e('div', { className: 'card fade-enter accent-border', style: { background: 'var(--surface2)' } },
            e('div', { className: 'card-head' },
              e('span', { className: 'card-title' }, 'Audit Results'),
              e(Badge, { label: result.data_severity || 'MEDIUM', type: (result.data_severity || 'MEDIUM').toLowerCase().replace('medium','med') })
            ),
            e('div', { className: 'card-body' },
              e('div', { className: 'result-score-grid' },
                [
                  { k: 'Rows analysed',    v: (result.n_rows || '—').toLocaleString() },
                  { k: 'Data bias score',  v: `${result.data_bias_score}/100` },
                  { k: 'Model bias score', v: `${result.model_bias_score}/100` },
                  { k: 'TPR gap',          v: result.tpr_gap ? result.tpr_gap.toFixed(4) : '—' },
                ].map(r =>
                  e('div', { key: r.k, className: 'result-score-cell' },
                    e('div', { className: 'result-score-key' }, r.k),
                    e('div', { className: 'result-score-val' }, r.v)
                  )
                )
              ),
              e('div', { style: { padding: '16px 20px', background: 'var(--ok-dim)', borderRadius: 'var(--r2)', border: '1px solid var(--ok-border)', margin: '16px 0' } },
                e('div', { style: { fontWeight: 700, fontSize: 14, color: 'var(--ok)', marginBottom: 6 } }, 'Reports Generated Successfully'),
                e('div', { style: { fontSize: 13, color: 'var(--t2)', lineHeight: 1.5 } }, 'PDF and JSON reports are now available via the API report endpoints for integration into your CI/CD pipelines.')
              ),
              e('a', { href: `${API}/docs`, target: '_blank', className: 'btn btn-ghost' }, 'View API Documentation ↗')
            )
          )
        : e('div', { className: 'card' },
            e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'What FairLens checks')),
            e('div', { className: 'card-body' },
              [
                { icon: '◈', title: 'Disparate impact ratio',       desc: 'P(positive|unprivileged) / P(positive|privileged) — flagged if below 0.80 (US EEOC four-fifths rule)' },
                { icon: '◉', title: 'Demographic parity gap',        desc: 'Absolute difference in positive prediction rates across protected groups' },
                { icon: '◎', title: 'Chi-square test',               desc: 'Statistical significance of association between the protected attribute and outcome' },
                { icon: '⊞', title: 'Equalized odds',                desc: 'Equal True Positive Rate and False Positive Rate across groups via SHAP and model audit' },
                { icon: '⊠', title: 'Three mitigation strategies',   desc: 'Pre-, in-, and post-processing fixes with before/after bias score comparison' },
              ].map(c =>
                e('div', { key: c.title, style: { display: 'flex', gap: 16, padding: '14px 0', borderBottom: '1px solid var(--line)' } },
                  e('span', { style: { fontSize: 16, color: 'var(--accent)', flexShrink: 0, fontFamily: 'var(--font-mono)' } }, c.icon),
                  e('div', null,
                    e('div', { style: { fontWeight: 700, fontSize: 14, marginBottom: 4, color: 'var(--t1)' } }, c.title),
                    e('div', { style: { fontSize: 13, color: 'var(--t3)', lineHeight: 1.6 } }, c.desc)
                  )
                )
              )
            )
          )
    )
  );
}

/* ═══════════════════════════════════════════
   PAGE: ABOUT
═══════════════════════════════════════════ */
function AboutPage() {
  return e('div', { className: 'page' },
    e('div', { className: 'hero mb-24' },
      e('div', { className: 'hero-orb orb-2' }),
      e('div', { className: 'hero-content' },
        e('div', { className: 'hero-eyebrow' }, 'Documentation'),
        e('h2', { className: 'hero-title', style: { fontSize: 28 } }, 'About FairLens System'),
        e('p', { className: 'hero-desc', style: { marginBottom: 0 } },
          'End-to-end AI bias detection and mitigation built for the Hack2Skill Unbiased AI Decision challenge. Validates across 5 real-world fairness benchmarks using IBM AIF360, Microsoft Fairlearn, and SHAP.'
        )
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
                { m: 'Disparate impact ratio',   t: '< 0.80 = FAIL',         s: 'US EEOC four-fifths rule' },
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
        e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'Technology Stack')),
        e('div', { className: 'card-body' },
          [
            { n: 'IBM AIF360',          r: 'Fairness metrics + Reweighing mitigation' },
            { n: 'Microsoft Fairlearn', r: 'ExponentiatedGradient with EqualizedOdds' },
            { n: 'SHAP',               r: 'Model explainability and feature attribution' },
            { n: 'scikit-learn',       r: 'LogisticRegression, RandomForestClassifier' },
            { n: 'FastAPI',            r: 'REST API backend + static file serving' },
            { n: 'ReportLab',          r: 'PDF audit report generation' },
            { n: 'React 18',           r: 'Frontend UI — CDN, no build step required' },
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
      e('div', { className: 'card-head' }, e('span', { className: 'card-title' }, 'REST API Reference')),
      e('div', { className: 'card-body', style: { padding: 0 } },
        e('div', { className: 'tbl-wrap' },
          e('table', null,
            e('thead', null, e('tr', null, ['Method', 'Endpoint', 'Description'].map(h => e('th', { key: h }, h)))),
            e('tbody', null,
              [
                ['GET',  '/',                  'Health check + dataset list'],
                ['GET',  '/datasets',           'List all available datasets'],
                ['POST', '/audit/data/{name}',  'Statistical data audit on named dataset'],
                ['POST', '/audit/model/{name}', 'Train model + audit for bias'],
                ['POST', '/audit/full/{name}',  'Full pipeline: audit + mitigate + report'],
                ['GET',  '/report/{name}/pdf',  'Download PDF audit report'],
                ['GET',  '/report/{name}/json', 'Download JSON audit report'],
                ['GET',  '/results',            'Get all cached audit results'],
                ['POST', '/audit/upload',       'Upload custom CSV and run full audit'],
              ].map(([m, ep, desc]) =>
                e('tr', { key: ep },
                  e('td', null, e('span', { className: `method-badge method-${m.toLowerCase()}` }, m)),
                  e('td', { className: 'mono', style: { color: 'var(--accent)', fontWeight: 600 } }, ep),
                  e('td', { style: { fontSize: 13, color: 'var(--t3)' } }, desc)
                )
              )
            )
          )
        )
      )
    )
  );
}

/* ═══════════════════════════════════════════
   ROOT APP
═══════════════════════════════════════════ */
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

  return e('div', { className: 'app-container' },
    // Animated Mesh Background
    e('div', { className: 'bg-mesh' },
      e('div', { className: 'mesh-blob blob-1' }),
      e('div', { className: 'mesh-blob blob-2' }),
      e('div', { className: 'mesh-blob blob-3' })
    ),
    
    // Application Shell
    e('div', { className: 'app' },
      // SIDEBAR
      e('nav', { className: 'sidebar' },
        e('div', { className: 'sidebar-brand' },
          e('div', { className: 'brand-mark' }, 'FL'),
          e('div', null,
            e('div', { className: 'brand-name' }, e('em', null, 'Fair'), 'Lens'),
            e('div', { className: 'brand-tagline' }, 'AI Bias Platform'),
          )
        ),
        e('div', { className: 'nav-section' },
          e('div', { className: 'nav-section-label' }, 'Platform'),
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
        e('div', { className: 'nav-section' },
          e('div', { className: 'nav-section-label' }, 'Analyzed Datasets'),
          DATASETS.map(d =>
            e('button', {
              key: d.id,
              className: `nav-ds-item ${page === 'audit' && auditDs === d.id ? 'active' : ''}`,
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
          e('div', { className: 'sidebar-status' },
            e('span', { className: 'status-dot' }),
            'API Online'
          )
        )
      ),

      // MAIN
      e('div', { className: 'main' },
        e('div', { className: 'topbar' },
          e('div', { className: 'topbar-left' },
            e('div', { className: 'topbar-breadcrumb' },
              e('span', null, pageTitle),
              page === 'audit' && e('span', { className: 'breadcrumb-sep' }, '/'),
              page === 'audit' && e('span', { className: 'breadcrumb-sub' }, DATASETS.find(d => d.id === auditDs)?.name),
            ),
            page === 'audit' && e('span', { className: 'topbar-pill' }, DATASETS.find(d => d.id === auditDs)?.domain)
          ),
          e('div', { className: 'topbar-right' },
            e('a', { href: 'https://github.com', target: '_blank', className: 'btn btn-ghost btn-sm' }, 'GitHub ↗'),
            e('a', { href: `${API}/docs`, target: '_blank', className: 'btn btn-sm btn-ghost' }, 'API Documentation'),
            e('button', { className: 'btn btn-primary btn-sm', onClick: () => navigate('upload') }, '+ New Audit')
          )
        ),
        e('div', { className: 'content' },
          page === 'dashboard' && e(DashboardPage, { onNavigate: navigate }),
          page === 'audit'     && e(AuditPage,     { key: auditDs, initDs: auditDs }),
          page === 'upload'    && e(UploadPage,     { addToast }),
          page === 'about'     && e(AboutPage,      null),
        )
      )
    ),

    // TOASTS
    e(ToastStack, { toasts, remove: removeToast })
  );
}

// Mount
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(e(App));

})();