/**
 * HITL Document Pipeline — Frontend Application
 * =============================================
 * Pure vanilla JS. No build step required.
 *
 * The UI is a projection of the backend state machine:
 *   upload → outline → highlights → draft → validation → complete
 * Each generation step is gated on the previous step's status being
 * "approved", mirroring the checks enforced by the API.
 */

const STEPS = ['outline', 'highlights', 'draft', 'validation'];

const STEP_META = {
  outline: {
    n: 1,
    name: 'Outline',
    short: 'Outline',
    title: 'Outline',
    prev: null,
    generateLabel: 'Generate Outline',
    description: 'Generate a structured outline from the input document, then approve or reject it.',
  },
  highlights: {
    n: 2,
    name: 'Themes',
    short: 'Themes',
    title: 'Themes',
    prev: 'outline',
    generateLabel: 'Generate Themes',
    description: 'Add section-level themes and supporting evidence to the approved outline.',
  },
  draft: {
    n: 3,
    name: 'Draft',
    short: 'Draft',
    title: 'Draft',
    prev: 'highlights',
    generateLabel: 'Generate Draft',
    description: 'Expand the approved outline and themes into narrative draft content.',
  },
  validation: {
    n: 4,
    name: 'Faithfulness & Compliance Check',
    short: 'Check',
    title: 'Faithfulness and Compliance Check',
    prev: 'draft',
    generateLabel: 'Run Faithfulness and Compliance Check',
    description: 'Evaluate the approved draft against the extracted requirements. Results are advisory and require human sign-off.',
  },
};

const DECISION_LABELS = {
  accept: 'Approved',
  edit: 'Approved with edits',
  reject: 'Rejected',
  alternative: 'Alternative requested',
  redraft: 'Regeneration requested',
};

const GENERATE_ENDPOINTS = {
  outline:    '/api/step/outline/generate',
  highlights: '/api/step/highlights/generate',
  draft:      '/api/step/draft/generate',
  validation: '/api/step/validation/run',
};

const DETERMINISTIC_RULES = [
  'Page-limit instructions acknowledged by part or page structure',
  'Required parts present in the outline',
  'Required number of case study references present',
  'Mandatory requirements referenced in the content',
  'Font and margin specifications surfaced for manual verification',
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
function initialState() {
  return {
    sessionId: null,
    currentStep: 'upload',
    documentTitle: '',
    currentDocument: null,
    outline: null,    outlineStatus: 'pending',    outlineSources: [],    outlineAttempt: 0,
    highlights: null, highlightsStatus: 'pending', highlightsSources: [], highlightsAttempt: 0,
    draft: null,      draftStatus: 'pending',      draftSources: [],      draftAttempt: 0,
    validation: null, validationStatus: 'pending', validationFlags: [],   validationAttempt: 0,
    extractedRequirements: [],
    reviewHistory: [],
    ui: { busy: null, editing: false, notes: '', editDraft: null },
  };
}

const AppState = initialState();
let reviewerId = sessionStorage.getItem('hitl-reviewer-id') || '';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);

const esc = s => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

const ICON_LOCK = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="2" y="5.25" width="8" height="5.5" rx="1"/><path d="M3.75 5.25V3.75a2.25 2.25 0 0 1 4.5 0v1.5"/></svg>`;
const ICON_CHECK = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2.5 6.25 5 8.75l4.5-5.5"/></svg>`;

function parseTime(ts) {
  if (!ts) return null;
  const hasZone = /[zZ]|[+-]\d{2}:?\d{2}$/.test(ts);
  const d = new Date(hasZone ? ts : `${ts}Z`);
  return isNaN(d) ? null : d;
}

function formatTime(ts) {
  const d = parseTime(ts);
  return d ? d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) : '—';
}

function stepName(step) {
  return STEP_META[step]?.title || step;
}

function lastRecord(step, decisions) {
  const list = AppState.reviewHistory || [];
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i].step === step && (!decisions || decisions.includes(list[i].decision))) return list[i];
  }
  return null;
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function toast(message, type = 'info') {
  const labels = { success: 'Done', error: 'Error', info: 'Note', warning: 'Action needed' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-label">${labels[type] || 'Note'}</span><span>${esc(message)}</span>`;
  $('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
async function api(method, path, body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = Array.isArray(err.detail) ? err.detail.map(d => d.msg).join('; ') : err.detail;
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function syncSession() {
  if (!AppState.sessionId) return;
  const d = await api('GET', `/api/session/${AppState.sessionId}`);
  Object.assign(AppState, {
    currentStep: d.current_step,
    documentTitle: d.document_title,
    currentDocument: d.current_document,
    outline: d.outline,       outlineStatus: d.outline_status,
    outlineSources: d.outline_sources,       outlineAttempt: d.outline_attempt,
    highlights: d.highlights, highlightsStatus: d.highlights_status,
    highlightsSources: d.highlights_sources, highlightsAttempt: d.highlights_attempt,
    draft: d.draft,           draftStatus: d.draft_status,
    draftSources: d.draft_sources,           draftAttempt: d.draft_attempt,
    validation: d.validation_report, validationStatus: d.validation_status,
    validationFlags: d.validation_flags,     validationAttempt: d.validation_attempt,
    extractedRequirements: d.extracted_requirements,
    reviewHistory: d.review_history,
  });
}

function resetStepUi() {
  AppState.ui.editing = false;
  AppState.ui.notes = '';
  AppState.ui.editDraft = null;
}

// ---------------------------------------------------------------------------
// State machine projection
// ---------------------------------------------------------------------------
function stepView(step) {
  const meta = STEP_META[step];
  const status = AppState[`${step}Status`];
  const isActive = AppState.sessionId && AppState.currentStep === step;
  const unlocked = meta.prev
    ? AppState[`${meta.prev}Status`] === 'approved'
    : Boolean(AppState.sessionId);

  if (status === 'approved') {
    const rec = lastRecord(step, ['accept', 'edit']);
    return { cls: 'is-approved', state: 'Approved', reason: rec?.reviewer_id ? `By ${rec.reviewer_id}` : '', marker: ICON_CHECK, isActive };
  }
  if (!unlocked) {
    const reason = meta.prev ? `Requires approved ${STEP_META[meta.prev].title}` : 'Requires an input document';
    return { cls: 'is-locked', state: 'Locked', reason, marker: ICON_LOCK, isActive: false };
  }
  if (AppState.ui.busy === step) {
    return { cls: 'is-available', state: step === 'validation' ? 'Running' : 'Generating', reason: '', marker: meta.n, isActive };
  }
  const map = {
    pending:         { state: 'Available',             reason: step === 'validation' ? 'Ready to run' : 'Ready to generate' },
    generating:      { state: 'Generating',            reason: '' },
    awaiting_review: { state: 'Awaiting human review', reason: 'Decision required' },
    rejected:        { state: 'Rejected',              reason: 'Regenerate to continue' },
  };
  const v = map[status] || map.pending;
  return { cls: 'is-available', state: v.state, reason: v.reason, marker: meta.n, isActive };
}

// ---------------------------------------------------------------------------
// Left panel
// ---------------------------------------------------------------------------
function renderIntakeStatus() {
  const el = $('intake-status');
  if (!AppState.sessionId) {
    el.innerHTML = `
      <div>
        <div class="intake-title">No input document</div>
        <div class="intake-sub">Create a session to begin</div>
      </div>`;
    return;
  }
  const n = AppState.extractedRequirements.length;
  el.innerHTML = `
    <div>
      <div class="intake-title">${esc(AppState.documentTitle)}</div>
      <div class="intake-sub">${n} requirement${n === 1 ? '' : 's'} extracted</div>
    </div>`;
}

function renderStepper() {
  $('stepper').innerHTML = STEPS.map(step => {
    const meta = STEP_META[step];
    const v = stepView(step);
    return `
      <li class="step ${v.cls} ${v.isActive ? 'is-active' : ''}" ${v.isActive ? 'aria-current="step"' : ''}
          aria-label="${esc(`Step ${meta.n}, ${meta.title}: ${v.state}${v.reason ? `. ${v.reason}` : ''}`)}"
          title="${esc(`${meta.title}: ${v.state}${v.reason ? ` (${v.reason})` : ''}`)}">
        <span class="step-marker">${v.marker}</span>
        <div>
          <div class="step-name">
            <span class="step-name-full">${meta.n} — ${esc(meta.name)}</span>
            <span class="step-name-short" aria-hidden="true">${esc(meta.short)}</span>
          </div>
          <div class="step-state">${esc(v.state)}</div>
          ${v.reason ? `<div class="step-reason">${esc(v.reason)}</div>` : ''}
        </div>
      </li>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Right panel
// ---------------------------------------------------------------------------
function renderAudit() {
  const history = AppState.reviewHistory || [];
  $('audit-count').textContent = history.length;

  if (!history.length) {
    $('audit-list').innerHTML = `<li class="audit-empty">No human decisions recorded yet.</li>`;
    return;
  }

  $('audit-list').innerHTML = [...history].reverse().map(r => {
    let diff = '';
    if (r.decision === 'edit') {
      diff = `
        <details class="audit-diff">
          <summary>Compare original and final content</summary>
          <div class="audit-diff-label">Original LLM output</div>
          <pre>${esc(r.original_content)}</pre>
          <div class="audit-diff-label">Final edited content</div>
          <pre>${esc(r.final_content)}</pre>
        </details>`;
    } else if (r.decision === 'accept') {
      diff = `
        <details class="audit-diff">
          <summary>View approved content</summary>
          <pre>${esc(r.final_content)}</pre>
        </details>`;
    } else if (r.original_content) {
      diff = `
        <details class="audit-diff">
          <summary>View reviewed output</summary>
          <pre>${esc(r.original_content)}</pre>
        </details>`;
    }

    return `
      <li class="audit-record">
        <div class="audit-top">
          <span class="audit-step">${esc(stepName(r.step))}</span>
          <span class="audit-decision ${esc(r.decision)}">${esc(DECISION_LABELS[r.decision] || r.decision)}</span>
        </div>
        <dl class="audit-meta">
          <dt>Reviewer</dt><dd>${esc(r.reviewer_id || '—')}</dd>
          <dt>Time</dt><dd>${esc(formatTime(r.timestamp))}</dd>
          <dt>Attempt</dt><dd>${esc(r.generation_attempt)}</dd>
          ${r.feedback ? `<dt>Notes</dt><dd>${esc(r.feedback)}</dd>` : ''}
        </dl>
        ${diff}
      </li>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Workspace — shared fragments
// ---------------------------------------------------------------------------
function statusBadge(step) {
  const status = AppState[`${step}Status`];
  if (AppState.ui.busy === step) return `<span class="status-badge is-running"><span class="spinner"></span>${step === 'validation' ? 'Running' : 'Generating'}</span>`;
  const map = {
    pending:         ['', step === 'validation' ? 'Not yet run' : 'Not yet generated'],
    generating:      ['is-running', 'Generating'],
    awaiting_review: ['is-review', 'Awaiting human review'],
    approved:        ['is-approved', 'Approved'],
    rejected:        ['is-rejected', 'Rejected'],
  };
  const [cls, label] = map[status] || map.pending;
  return `<span class="status-badge ${cls}">${label}</span>`;
}

function workspaceHead(step) {
  const meta = STEP_META[step];
  const attempt = AppState[`${step}Attempt`];
  return `
    <div class="ws-head">
      <div class="ws-kicker">Step ${meta.n} of 4</div>
      <h1 class="ws-title">${esc(meta.title)}</h1>
      <p class="ws-desc">${esc(meta.description)}</p>
      <div class="ws-status-row">
        ${statusBadge(step)}
        ${attempt > 0 ? `<span class="meta-text">Attempt ${attempt}</span>` : ''}
      </div>
    </div>`;
}

function prerequisiteNote(step) {
  const prev = STEP_META[step].prev;
  if (!prev) return '';
  const rec = lastRecord(prev, ['accept', 'edit']);
  const who = rec?.reviewer_id ? ` by <strong>${esc(rec.reviewer_id)}</strong>` : '';
  const when = rec ? ` on ${esc(formatTime(rec.timestamp))}` : '';
  return `<div class="callout subtle">Unlocked because <strong>${esc(STEP_META[prev].title)}</strong> was approved${who}${when}.</div>`;
}

function sourcesDisclosure(step) {
  const sources = AppState[`${step}Sources`] || [];
  if (!sources.length) return '';
  const items = sources.map(s => `
    <li class="source-item">
      <div>
        <div class="source-title">${esc(s.title)}</div>
        <div class="source-meta">${esc((s.source_type || '').replace(/_/g, ' '))}${s.category ? ` · ${esc(s.category)}` : ''}${s.status ? ` · ${esc(s.status)}` : ''}</div>
      </div>
      <span class="source-score" title="Relevance score">${Number(s.relevance_score).toFixed(2)}</span>
    </li>`).join('');
  return `
    <details class="disclosure">
      <summary>Retrieved context <span class="meta-text">${sources.length} source${sources.length === 1 ? '' : 's'}</span></summary>
      <div class="disclosure-body">
        <ul class="source-list">${items}</ul>
        <div class="gate-note">
          <strong>Pre-generation content validation gate.</strong>
          In the RAG pipeline, retrieved content is filtered before reaching the LLM:
          sources older than 1,825 days or below the minimum relevance threshold are excluded.
          This does not guarantee factual correctness.
        </div>
      </div>
    </details>`;
}

function requirementsDisclosure() {
  const reqs = AppState.extractedRequirements || [];
  if (!reqs.length) return '';
  const rows = reqs.map(r => `
    <tr>
      <td>${esc(r.ref)}</td>
      <td>${esc(r.section)}</td>
      <td>${esc(r.text)}</td>
      <td>${r.mandatory ? '<span class="tag-mandatory">Mandatory</span>' : ''}</td>
    </tr>`).join('');
  return `
    <details class="disclosure">
      <summary>Extracted requirements <span class="meta-text">${reqs.length}</span></summary>
      <div class="disclosure-body">
        <div class="table-wrap">
          <table class="req-table">
            <thead><tr><th>Ref</th><th>Section</th><th>Requirement</th><th></th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </details>`;
}

function reviewerFields(notesHint) {
  return `
    <div class="card-body">
      <div class="field-row">
        <div class="field">
          <label for="reviewer-id">Reviewer ID</label>
          <input class="input" id="reviewer-id" data-bind="reviewer" value="${esc(reviewerId)}" placeholder="e.g. reviewer-01" autocomplete="off" />
          <span class="hint">Recorded in the review history.</span>
        </div>
        <div class="field">
          <label for="review-notes">Reviewer notes</label>
          <textarea class="textarea" id="review-notes" data-bind="notes" placeholder="Reason for rejection, or guidance for regeneration">${esc(AppState.ui.notes)}</textarea>
          <span class="hint">${notesHint}</span>
        </div>
      </div>
    </div>`;
}

function notesReady() {
  return AppState.ui.notes.trim().length >= 5;
}

// ---------------------------------------------------------------------------
// Workspace — views
// ---------------------------------------------------------------------------
function renderIntake() {
  return `
    <div class="ws-head">
      <div class="ws-kicker">Session setup</div>
      <h1 class="ws-title">Provide an input document</h1>
      <p class="ws-desc">Requirements are extracted from the document. Step 1 unlocks once the session is created.</p>
    </div>

    <div class="principles">
      <div class="principle">
        <div class="principle-title">Human approval at every stage</div>
        <div class="principle-text">Generated content is not carried forward until a reviewer approves it.</div>
      </div>
      <div class="principle">
        <div class="principle-title">Sequential gating</div>
        <div class="principle-text">Each step unlocks only after the preceding step is approved.</div>
      </div>
      <div class="principle">
        <div class="principle-title">Recorded decisions</div>
        <div class="principle-text">Every approval, rejection and edit is added to the review history.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="card-title">Input document</span>
        <span class="card-sub">Minimum 50 characters</span>
      </div>
      <div class="card-body">
        <div class="field">
          <label for="document-title-input">Document title</label>
          <input class="input" id="document-title-input" placeholder="Untitled Document" maxlength="120" />
        </div>
        <div class="field">
          <label for="input-text-input">Document text</label>
          <textarea class="textarea tall" id="input-text-input" placeholder="Paste the full document text here."></textarea>
        </div>
      </div>
      <div class="action-bar">
        <button class="btn btn-secondary" data-action="load-sample" type="button">Load sample document</button>
        <span class="spacer"></span>
        <button class="btn btn-primary btn-lg" data-action="upload" type="button">Create session and extract requirements</button>
      </div>
    </div>`;
}

function renderGenerateView(step) {
  const meta = STEP_META[step];
  const status = AppState[`${step}Status`];
  const busy = AppState.ui.busy === step;

  let rejected = '';
  if (status === 'rejected') {
    const rec = lastRecord(step, ['reject']);
    rejected = `
      <div class="callout">
        <strong>Previous attempt rejected${rec?.reviewer_id ? ` by ${esc(rec.reviewer_id)}` : ''}.</strong>
        ${rec?.feedback ? `Reason: ${esc(rec.feedback)}.` : ''}
        The rejected output was not carried forward. Generate a new attempt to continue.
      </div>`;
  }

  const text = step === 'validation'
    ? 'Run the hybrid check on the approved draft. Results are returned as advisory flags for your review.'
    : 'Content will be generated and held for human review. Nothing advances until you approve it.';

  return `
    ${workspaceHead(step)}
    ${prerequisiteNote(step)}
    ${rejected}
    <div class="card mt-16">
      <div class="empty-state flush">
        <div class="empty-state-title">${status === 'rejected' ? 'Ready for a new attempt' : `No ${meta.title.toLowerCase()} yet`}</div>
        <p class="empty-state-text">${text}</p>
        <button class="btn btn-primary btn-lg" data-action="generate" data-step="${step}" type="button" ${busy ? 'disabled' : ''}>
          ${busy ? `<span class="spinner"></span>${step === 'validation' ? 'Running' : 'Generating'}` : esc(meta.generateLabel)}
        </button>
      </div>
    </div>
    ${requirementsDisclosure()}`;
}

function renderReviewView(step) {
  const content = AppState.currentDocument || '';
  const attempt = AppState[`${step}Attempt`];
  const cumulative = step !== 'outline'
    ? `<div class="callout subtle">The working document is cumulative. This output contains the previously approved content plus the new ${esc(STEP_META[step].title.toLowerCase())} section.</div>`
    : '';

  let contentBlock;
  let actions;

  if (AppState.ui.editing) {
    const edited = AppState.ui.editDraft ?? content;
    contentBlock = `
      <div class="edit-grid">
        <div class="card">
          <div class="card-head">
            <span class="card-title">Original LLM output</span>
            <span class="origin-tag ai">AI-generated</span>
          </div>
          <pre class="output-text">${esc(content)}</pre>
        </div>
        <div class="card">
          <div class="card-head">
            <span class="card-title">Final edited content</span>
            <span class="origin-tag human">Human-edited</span>
          </div>
          <textarea class="edit-area" id="edit-area" data-bind="edit" spellcheck="true">${esc(edited)}</textarea>
        </div>
      </div>`;
    actions = `
      <button class="btn btn-primary" data-action="approve-edit" data-step="${step}" type="button">Approve edited content</button>
      <button class="btn btn-secondary" data-action="cancel-edit" type="button">Discard edits</button>
      <span class="spacer"></span>
      <span class="disabled-reason">Both versions are saved to the review record.</span>`;
  } else {
    contentBlock = `
      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Generated content</div>
            <div class="card-sub">Attempt ${attempt}. Not yet approved.</div>
          </div>
          <span class="origin-tag ai">AI-generated</span>
        </div>
        <pre class="output-text">${esc(content)}</pre>
      </div>`;
    actions = reviewActions(step, 'Approve');
  }

  return `
    ${workspaceHead(step)}
    ${cumulative}
    <div class="mt-16">${contentBlock}</div>
    ${reviewCard(step, actions)}
    ${sourcesDisclosure(step)}
    ${requirementsDisclosure()}`;
}

function reviewActions(step, approveLabel) {
  const ready = notesReady();
  const regenLabel = step === 'validation' ? 'Re-run with guidance' : 'Regenerate with guidance';
  return `
    <button class="btn btn-primary" data-action="approve" data-step="${step}" type="button">${approveLabel}</button>
    <button class="btn btn-secondary" data-action="reject" data-step="${step}" type="button" ${ready ? '' : 'disabled'}>Reject</button>
    <span class="spacer"></span>
    ${step !== 'validation' ? `<button class="btn btn-quiet" data-action="start-edit" type="button">Edit before approving</button>` : ''}
    <button class="btn btn-quiet" data-action="redraft" data-step="${step}" type="button" ${ready ? '' : 'disabled'}>${regenLabel}</button>
    <span class="disabled-reason ${ready ? 'hidden' : ''}" id="notes-required">Reject and regenerate require reviewer notes.</span>`;
}

function reviewCard(step, actions) {
  const hint = step === 'validation'
    ? 'Required to reject or re-run (minimum 5 characters).'
    : 'Required to reject or regenerate (minimum 5 characters).';
  return `
    <div class="card action-card">
      <div class="card-head">
        <div>
          <div class="card-title">Human review</div>
          <div class="card-sub">Your decision is recorded as a HITLReviewRecord.</div>
        </div>
      </div>
      ${reviewerFields(hint)}
      <div class="action-bar" id="action-bar">${actions}</div>
    </div>`;
}

function renderEvaluationView() {
  const step = 'validation';
  const flags = AppState.validationFlags || [];
  const count = sev => flags.filter(f => f.severity === sev).length;

  const flagItems = flags.length
    ? flags.map(f => `
        <li class="flag">
          <span class="flag-sev ${esc(f.severity)}">${esc(f.severity)}</span>
          <div>
            <div class="flag-section">${esc(f.section)}</div>
            <div class="flag-desc">${esc(f.description)}</div>
            <div class="flag-fix">${esc(f.suggestion)}</div>
          </div>
        </li>`).join('')
    : `<li class="flag"><span></span><div class="muted">No flags returned for this attempt.</div></li>`;

  return `
    ${workspaceHead(step)}

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Evaluation results</div>
          <div class="card-sub">Attempt ${AppState.validationAttempt}. Advisory results, not a certification.</div>
        </div>
        <span class="origin-tag ai">Advisory</span>
      </div>
      <div class="summary-grid">
        <div class="summary-cell"><div class="summary-num">${count('critical')}</div><div class="summary-label">Critical</div></div>
        <div class="summary-cell"><div class="summary-num">${count('major')}</div><div class="summary-label">Major</div></div>
        <div class="summary-cell"><div class="summary-num">${count('minor')}</div><div class="summary-label">Minor</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Advisory AI Flags</div>
          <div class="card-sub">Flags from the hybrid check. Each one should be verified by the reviewer.</div>
        </div>
        <span class="meta-text">${flags.length} flag${flags.length === 1 ? '' : 's'}</span>
      </div>
      <ul class="flag-list">${flagItems}</ul>
    </div>

    <details class="disclosure">
      <summary>Evaluation method</summary>
      <div class="disclosure-body">
        <p>The check combines two layers. The results above are returned as a single list.</p>
        <p class="mt-12"><strong>LLM-based evaluation</strong>: a semantic pass that looks for gaps between the draft and the extracted requirements. It runs only when Bedrock is configured.</p>
        <p class="mt-12"><strong>Deterministic checks</strong>: rule-based checks applied to the input document and content.</p>
        <ul class="rule-list mt-6">${DETERMINISTIC_RULES.map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </div>
    </details>

    <details class="disclosure">
      <summary>Raw evaluation report</summary>
      <div class="disclosure-body">
        <pre class="output-text">${esc(AppState.validation || '')}</pre>
        <div class="mt-12">
          <button class="btn btn-secondary btn-sm" data-action="download-report" type="button">Download report</button>
        </div>
      </div>
    </details>

    <details class="disclosure">
      <summary>Approved draft under evaluation</summary>
      <div class="disclosure-body"><pre class="output-text">${esc(AppState.currentDocument || '')}</pre></div>
    </details>

    ${reviewCard(step, reviewActions(step, 'Accept evaluation results'))}
    <div class="callout subtle mt-12">Accepting records your sign-off on these advisory results and completes the session. It does not certify the document.</div>
    ${requirementsDisclosure()}`;
}

function renderComplete() {
  const rows = STEPS.map(step => {
    const rec = lastRecord(step, ['accept', 'edit']);
    return `
      <tr>
        <td>${STEP_META[step].n} — ${esc(STEP_META[step].title)}</td>
        <td>${rec ? `${esc(DECISION_LABELS[rec.decision])}${rec.reviewer_id ? ` by ${esc(rec.reviewer_id)}` : ''}` : '—'}</td>
        <td class="meta-text">${rec ? esc(formatTime(rec.timestamp)) : ''}</td>
      </tr>`;
  }).join('');

  return `
    <div class="ws-head">
      <div class="ws-kicker">Session complete</div>
      <h1 class="ws-title">All stages approved</h1>
      <p class="ws-desc">Each stage was approved by a human reviewer in sequence. The full decision history is in the HITL Governance Layer panel.</p>
      <div class="ws-status-row"><span class="status-badge is-approved">Complete</span></div>
    </div>

    <div class="card">
      <div class="card-head"><span class="card-title">Approval summary</span></div>
      <table class="summary-table"><tbody>${rows}</tbody></table>
      <div class="action-bar">
        <button class="btn btn-primary" data-action="download-document" type="button">Download approved document</button>
        <button class="btn btn-secondary" data-action="download-report" type="button">Download evaluation report</button>
        <span class="spacer"></span>
        <button class="btn btn-quiet" data-action="new-session" type="button">Start new session</button>
      </div>
    </div>

    <details class="disclosure">
      <summary>Approved document</summary>
      <div class="disclosure-body"><pre class="output-text">${esc(AppState.currentDocument || '')}</pre></div>
    </details>`;
}

function renderWorkspace() {
  const ws = $('workspace');
  const step = AppState.currentStep;

  if (!AppState.sessionId || step === 'upload') {
    ws.innerHTML = renderIntake();
  } else if (step === 'complete') {
    ws.innerHTML = renderComplete();
  } else {
    const status = AppState[`${step}Status`];
    if (status === 'awaiting_review' && AppState.ui.busy !== step) {
      ws.innerHTML = step === 'validation' ? renderEvaluationView() : renderReviewView(step);
    } else {
      ws.innerHTML = renderGenerateView(step);
    }
  }
}

let lastRenderedStep = null;

function render() {
  const viewKey = `${AppState.sessionId}:${AppState.currentStep}`;
  if (lastRenderedStep !== null && lastRenderedStep !== viewKey) window.scrollTo({ top: 0 });
  lastRenderedStep = viewKey;

  const chip = $('session-chip');
  chip.classList.toggle('hidden', !AppState.sessionId);
  $('session-id-display').textContent = AppState.sessionId || '';
  renderIntakeStatus();
  renderStepper();
  renderAudit();
  renderWorkspace();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
async function withBusyButton(btn, label, fn) {
  const original = btn ? btn.innerHTML : '';
  if (btn) { btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>${label}`; }
  try {
    await fn();
  } finally {
    if (btn && btn.isConnected) { btn.disabled = false; btn.innerHTML = original; }
  }
}

async function handleUpload(btn) {
  const inputText = $('input-text-input').value.trim();
  const documentTitle = $('document-title-input').value.trim() || 'Untitled Document';
  if (inputText.length < 50) { toast('Paste at least 50 characters of document text.', 'warning'); return; }

  await withBusyButton(btn, 'Processing', async () => {
    try {
      const data = await api('POST', '/api/document/upload', { input_text: inputText, document_title: documentTitle });
      AppState.sessionId = data.session_id;
      await syncSession();
      resetStepUi();
      render();
      toast(`Session created. ${data.requirements_count} requirements extracted.`, 'success');
    } catch (e) {
      toast(`Upload failed: ${e.message}`, 'error');
    }
  });
}

async function handleLoadSample() {
  try {
    const res = await fetch('/api/sample-document');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    $('input-text-input').value = await res.text();
    $('document-title-input').value = 'Sample Project Brief';
  } catch (e) {
    toast('Could not load the sample document.', 'error');
  }
}

async function handleGenerate(step) {
  AppState.ui.busy = step;
  render();
  try {
    await api('POST', GENERATE_ENDPOINTS[step], { session_id: AppState.sessionId, step });
    await syncSession();
    resetStepUi();
    toast('Output ready. Awaiting your review.', 'info');
  } catch (e) {
    toast(`Generation failed: ${e.message}`, 'error');
  } finally {
    AppState.ui.busy = null;
    render();
  }
}

async function submitReview(step, decision, btn) {
  const notes = AppState.ui.notes.trim();
  const payload = {
    session_id: AppState.sessionId,
    step,
    decision,
    edited_content: decision === 'edit' ? (AppState.ui.editDraft ?? AppState.currentDocument) : null,
    feedback: notes || null,
    reviewer_id: reviewerId.trim() || 'anonymous',
  };

  await withBusyButton(btn, 'Recording', async () => {
    try {
      await api('POST', `/api/step/${step}/review`, payload);
      await syncSession();
      resetStepUi();
      render();
      const messages = {
        accept: ['Approved. The next step is now unlocked.', 'success'],
        edit:   ['Edited content approved. The next step is now unlocked.', 'success'],
        reject: ['Rejected. Generate a new attempt to continue.', 'warning'],
      };
      const [msg, type] = step === 'validation' && decision === 'accept'
        ? ['Evaluation results accepted. Session complete.', 'success']
        : messages[decision] || ['Decision recorded.', 'info'];
      toast(msg, type);
    } catch (e) {
      toast(`Review failed: ${e.message}`, 'error');
    }
  });
}

async function handleRedraft(step, btn) {
  const notes = AppState.ui.notes.trim();
  if (notes.length < 5) { toast('Enter at least 5 characters of guidance.', 'warning'); return; }

  AppState.ui.busy = step;
  await withBusyButton(btn, 'Regenerating', async () => {
    try {
      await api('POST', `/api/step/${step}/redraft`, {
        session_id: AppState.sessionId,
        step,
        notes,
        reviewer_id: reviewerId.trim() || 'anonymous',
      });
      await syncSession();
      resetStepUi();
      toast('New attempt generated. Awaiting your review.', 'info');
    } catch (e) {
      toast(`Regeneration failed: ${e.message}`, 'error');
    }
  });
  AppState.ui.busy = null;
  render();
}

function handleNewSession() {
  if (AppState.sessionId && !confirm('Start a new session? The current session will remain in server memory but will no longer be shown.')) return;
  Object.assign(AppState, initialState());
  render();
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------
$('workspace').addEventListener('click', e => {
  const btn = e.target.closest('[data-action]');
  if (!btn || btn.disabled) return;
  const step = btn.dataset.step;

  switch (btn.dataset.action) {
    case 'upload':       handleUpload(btn); break;
    case 'load-sample':  handleLoadSample(); break;
    case 'generate':     handleGenerate(step); break;
    case 'approve':      submitReview(step, 'accept', btn); break;
    case 'approve-edit': submitReview(step, 'edit', btn); break;
    case 'reject':
      if (!notesReady()) { toast('Enter a reason for rejection in Reviewer notes.', 'warning'); $('review-notes')?.focus(); return; }
      submitReview(step, 'reject', btn);
      break;
    case 'redraft':      handleRedraft(step, btn); break;
    case 'start-edit':
      AppState.ui.editing = true;
      AppState.ui.editDraft = AppState.currentDocument || '';
      renderWorkspace();
      $('edit-area')?.focus();
      break;
    case 'cancel-edit':
      AppState.ui.editing = false;
      AppState.ui.editDraft = null;
      renderWorkspace();
      break;
    case 'download-report':
      if (!AppState.validation) { toast('No evaluation report available.', 'warning'); return; }
      downloadText(AppState.validation, `evaluation-report-${AppState.sessionId}.txt`);
      break;
    case 'download-document':
      downloadText(AppState.currentDocument || '', `approved-document-${AppState.sessionId}.txt`);
      break;
    case 'new-session':  handleNewSession(); break;
  }
});

$('workspace').addEventListener('input', e => {
  const bind = e.target.dataset.bind;
  if (bind === 'reviewer') {
    reviewerId = e.target.value;
    sessionStorage.setItem('hitl-reviewer-id', reviewerId);
  } else if (bind === 'edit') {
    AppState.ui.editDraft = e.target.value;
  } else if (bind === 'notes') {
    AppState.ui.notes = e.target.value;
    const ready = notesReady();
    document.querySelectorAll('[data-action="reject"], [data-action="redraft"]').forEach(b => { b.disabled = !ready; });
    $('notes-required')?.classList.toggle('hidden', ready);
  }
});

$('btn-new-session').addEventListener('click', handleNewSession);

document.addEventListener('DOMContentLoaded', render);
