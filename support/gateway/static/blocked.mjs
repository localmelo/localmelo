import {formatTime, gistSentence, language, nameList, t} from './i18n.mjs';
import {maker} from './dom.mjs';

// The kinds of step auto mode blocks (#339), as the review names them.
export const CATEGORIES = new Set([
  'self', 'credentials', 'money', 'account', 'system', 'app_control', 'open_app', 'persistence', 'remote_code', 'install', 'restore',
  'env', 'download', 'outside_write', 'delete', 'service', 'send_out', 'opaque', 'beyond_request', 'unreviewed',
]);
// What the person's words can allow (their request names the step); the rest (`app_control`, `system`,
// `money`, `account` and `open_app` among them) only an Allow once, and `self` never.
const WORDS_ALLOW = new Set(['install', 'restore', 'env', 'download', 'outside_write', 'delete', 'send_out', 'service', 'beyond_request']);
// How long an Allow once holds, as the server keeps it.
export const ALLOW_ONCE_S = 30 * 60;

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

/** Whether a call's approval is a block of auto mode: origin `auto`, not allowed. */
export const isBlock = (approval) => isObject(approval) && approval.origin === 'auto' && approval.allowed === false;

/** The block's categories the review named, strongest first. */
function categoriesOf(approval) {
  const found = isObject(approval?.review) && Array.isArray(approval.review.categories) ? approval.review.categories : [];
  return found.map(String).filter((category) => CATEGORIES.has(category));
}

/** A block's reason in a few words, in the page's language: "installs software". */
export function blockWords(approval) {
  const [first] = categoriesOf(approval);
  if (first) return t(`auto.words.${first}`);
  return String(approval?.reason ?? '') === 'the turn stopped' ? t('auto.words.stopped') : '';
}

/**
 * The reasons an Ask-mode card gives for a high-impact call, in the page's language (see `reasonLines`),
 * and its question: English keeps the checker's own title, another language asks by the first category.
 */
export function reviewWords(review, folder = '') {
  const found = isObject(review) ? review : {};
  const categories = categoriesOf({review: found});
  const own = typeof found.title === 'string' ? found.title : '';
  const title = categories.length && (language() !== 'en' || !own) ? t(`approval.title.${categories[0]}`, {folder}) : own;
  return {title, reasons: reasonLines(found, categories)};
}

/**
 * The sentences under a blocked step's title, in the page's language. English keeps the rules' own
 * sentences, which name the package, the file or the host; another language names the kind of step
 * in its own words (the step itself is on the card, as written), and the apps a script operates when
 * the review's `gist` names them. A review the model settled says only that the model stopped it, in any
 * language: the title already names what it found. A
 * review with no sentences of its own reads by category in any language. A question about attaching
 * a file to a page (#551) says the file, its folder and the site, and when the person did not name the
 * file, in every language alike, from the facts its `gist` carries.
 */
export function reasonLines(review, categories) {
  const attach = isObject(review.gist) && isObject(review.gist.attach) ? review.gist.attach : null;
  if (attach) {
    const key = attach.named === false ? 'auto.reason.attachUnnamed' : 'auto.reason.attach';
    return [t(key, {file: String(attach.file ?? ''), folder: String(attach.folder ?? ''), site: String(attach.site ?? '')})];
  }
  const written = (Array.isArray(review.reasons) ? review.reasons : []).map(String).filter(Boolean);
  // What the model found is the card's title already: the line says who found it, once, in every language.
  const modelFound = review.by === 'model' && !['beyond_request', 'opaque', 'unreviewed'].includes(categories[0]);
  if (modelFound) return [t('auto.reason.model')];
  if (written.length && language() === 'en') return written;
  const apps = isObject(review.gist) && Array.isArray(review.gist.apps) ? nameList(review.gist.apps) : '';
  return categories.slice(0, 3).map((category) => (
    category === 'app_control' && apps ? t('auto.reason.appControlNamed', {apps}) : t(`auto.reason.${category}`)
  ));
}

/**
 * The sentence an approval card starts with, its warning and the model's note, in the page's language,
 * from the `gist` the server sends (catalog ids and parameters); null when there is none. The sentence is
 * the code's own whenever the rules found the step risky; the model's words then stand in `note`, already
 * labelled as the model's ("melo says: ..."). The warning is `''` unless the model's words do not match
 * what the code does.
 */
export function gistWords(gist) {
  const say = isObject(gist) ? gist.say : null;
  if (!isObject(say) || typeof say.id !== 'string') return null;
  const warn = isObject(gist.warn) && typeof gist.warn.id === 'string' ? gist.warn : null;
  return {
    sentence: gistSentence(say),
    warning: warn ? t(`gist.warn.${warn.id}`, isObject(warn.params) ? warn.params : {}) : '',
    note: typeof gist.note === 'string' && gist.note ? t('gist.say.model', {text: gist.note}) : '',
  };
}

// The engine's words for a control, `a button "Sign in"` (and ` inside a box from accounts.example.org`), as the
// request carries them in `on`: its kind, its label and the other site's box. null when the words are not those.
const CONTROL_KINDS = ['link', 'button', 'textbox', 'checkbox', 'select'];
export function controlOf(on) {
  const found = /^an? ([a-z]+)(?: "(.*)")?(?: inside a box from (\S+))?$/.exec(String(on ?? '').trim());
  if (!found || !CONTROL_KINDS.includes(found[1])) return null;
  return {kind: found[1], label: found[2] ?? '', frame: found[3] ?? ''};
}

// What a click, a choice or typing is on, in a question: “Sign in” when it is named, else the kind of control.
export function actionTitle(key, control, site) {
  return control.label
    ? t(`approval.${key}`, {label: control.label, site})
    : t(`approval.${key}Kind`, {kind: t(`approval.kind.${control.kind}`), site});
}

/**
 * What a page question puts into the page, as its card shows it, or null for another question: typing
 * (`browser-type:<site>`: the whole text) or a form sent (`browser-submit:<site>`: where it goes and what it
 * holds). `title` asks it in words with its site (and the box typed into), `what` names the text, `text` is all
 * of it, uncut. The live approval card and a blocked step's card both show these.
 */
export function pageText(tool, args = {}) {
  const name = String(tool ?? '');
  const a = isObject(args) ? args : {};
  const site = String(a.site || name.slice(name.indexOf(':') + 1));
  if (name.startsWith('browser-type:')) {
    // The box it types into, when melo named it.
    const control = controlOf(a.on);
    const title = control ? actionTitle('typeInto', control, site) : t('approval.typeSite', {site});
    return {title, what: t('approval.actTyped'), text: String(a.text ?? '')};
  }
  if (!name.startsWith('browser-submit:')) return null;
  const rows = (Array.isArray(a.fields) ? a.fields : []).filter(Array.isArray).map(([field, value]) => `${field}: ${value}`);
  const to = a.to ? String(a.to) : t('approval.submitPage');
  return {
    title: t('approval.submitSite', {site}),
    what: t('approval.submitAddress'),
    text: [to, ...(rows.length ? ['', t('approval.submitFields'), ...rows] : [])].join('\n'),
  };
}

/**
 * What a blocked call's card says, or null for a call auto mode did not block. `step` is the call's
 * code or command on one line, or for a question its tool asked itself (the block's `tool`) all it puts
 * into the page (`pageText`), under `stepLabel`; `gist` the sentence the card starts with, its warning and the
 * model's note (or null); `allow` whether Allow once may run it (never for `self`, never for a call a
 * stopped turn left unreviewed, never without its key); `words` whether the person's words could allow it too;
 * `keepSite` the site a scheduled task's run may keep the answer on for its next runs (the server's `keep_site`, only
 * ever for such a run), and `kept` whether it is kept there already.
 */
export function blockFacts({approval, name = '', arguments: args = {}, preview = '', requestId = ''} = {}) {
  if (!isBlock(approval)) return null;
  const categories = categoriesOf(approval);
  const first = categories[0] ?? '';
  const review = isObject(approval.review) ? approval.review : {};
  const self = categories.includes('self');
  const doubts = Array.isArray(review.doubts) && review.doubts.length > 0;
  const page = typeof approval.tool === 'string' ? pageText(approval.tool, approval.arguments) : null;
  return {
    requestId: String(approval.request_id ?? requestId ?? ''),
    stepKey: String(approval.step_key ?? ''),
    category: first,
    title: first ? t('auto.blocked.title', {title: t(`auto.title.${first}`)}) : t('auto.blocked.stopped'),
    reasons: reasonLines(review, categories),
    gist: page ? {sentence: page.title, warning: '', note: ''} : gistWords(review.gist),
    name: String(name),
    step: page ? page.text : stepText(name, args, preview),
    stepLabel: page ? page.what : '',
    self,
    allow: Boolean(first) && !self && Boolean(approval.step_key),
    words: Boolean(first) && !doubts && categories.every((category) => WORDS_ALLOW.has(category)),
    keepSite: !self && typeof approval.keep_site === 'string' ? approval.keep_site : '',
    kept: approval.kept === true,
  };
}

/** Allow once, or where the allowance stands. */
function allowRow(doc, allowing, {busy, onAllow}) {
  const {node} = maker(doc);
  const foot = node('div', 'block-foot');
  if (allowing.state === 'open') {
    const allow = node('button', 'outline-button block-allow', t(busy === 'allowing' ? 'auto.blocked.allowing' : 'auto.blocked.allow'));
    allow.type = 'button';
    allow.disabled = Boolean(busy);
    if (onAllow) allow.addEventListener('click', onAllow);
    foot.append(allow, node('span', 'block-once', t(busy === 'running' ? 'auto.blocked.afterTurn' : 'auto.blocked.once')));
    return foot;
  }
  const words = {
    pending: () => t('auto.blocked.pending', {time: clock(allowing.at)}),
    used: () => t('auto.blocked.used', {time: clock(allowing.at)}),
    expired: () => t('auto.blocked.expired'),
  }[allowing.state];
  foot.append(node('span', 'block-done', words ? words() : ''));
  return foot;
}

/** The second choice of a scheduled task's blocked step: keep the answer for its next runs on the site, or that it is kept. */
function keepRow(doc, facts, {busy, onKeep}) {
  const {node} = maker(doc);
  const row = node('div', 'block-foot block-keep-row');
  const site = facts.keepSite.slice(facts.keepSite.indexOf('://') + 3);
  if (facts.kept) {
    row.append(node('span', 'block-done', t('auto.blocked.kept', {site})));
    return row;
  }
  const keep = node('button', 'outline-button block-keep', t(busy === 'keeping' ? 'auto.blocked.keeping' : 'auto.blocked.keep', {site}));
  keep.type = 'button';
  keep.disabled = Boolean(busy);
  if (onKeep) keep.addEventListener('click', onKeep);
  row.append(keep, node('span', 'block-once', t('auto.blocked.keepNote')));
  return row;
}

/** A call's code or command on one line, for the card's monospace step. */
export function stepText(name, args, preview = '') {
  const source = isObject(args) ? args : {};
  let text = '';
  if (name === 'python_exec') text = String(source.code ?? '');
  else if (name === 'shell_exec') text = Array.isArray(source.command) ? source.command.join(' ') : String(source.command ?? '');
  else if (typeof source.path === 'string') text = source.path;
  const line = text.split('\n').map((part) => part.trim()).filter((part) => part && !part.startsWith('#')).join('; ');
  return (line || String(preview ?? '')).slice(0, 400);
}

/** A step's code or command on one line, cut to *max* characters: for a row that names it. */
export function stepLine(text, max = 120) {
  const line = String(text ?? '').split(/\s+/).filter(Boolean).join(' ');
  return line.length > max ? `${line.slice(0, max - 1).trimEnd()}…` : line;
}

/**
 * Where an Allow once stands: `open` (the button), `pending` (allowed, not run yet), `used` (ran at
 * `at`) or `expired`. `until` is when the allowance runs out (from the server), `used` when a later
 * call of the same step ran on it, `now` the clock, all in seconds.
 */
export function allowState({until = null, used = null, now = 0} = {}) {
  if (used) return {state: 'used', at: used};
  if (until === null || until === undefined) return {state: 'open'};
  return now < until ? {state: 'pending', at: until} : {state: 'expired'};
}

const clock = (seconds) => formatTime(new Date(seconds * 1000), 'clock');

/**
 * The card under a blocked call: what was blocked and why, the step, and Allow once. `allowing` is
 * `{state, at}` from `allowState`; `busy` says a request is under way or a turn runs (the button
 * waits); `developer` adds the tool's name; `onAllow()` sends the Allow once; `error` is the last
 * failure's text. A step of a scheduled task's run (`facts.keepSite`) also offers to keep the answer
 * for that task's next runs on that site (`onKeep()`), with what still stops; once kept, it says so.
 */
export function blockCard(doc, facts, {allowing = {state: 'open'}, busy = false, developer = false, onAllow = null, onKeep = null, error = ''} = {}) {
  const {node} = maker(doc);
  const card = node('section', 'block-card');
  card.dataset.state = facts.self ? 'never' : allowing.state;
  // What the step does comes first, in a plain sentence; a warning follows when the words and the code differ.
  if (facts.gist?.sentence) card.append(node('p', 'block-gist', facts.gist.sentence));
  if (facts.gist?.warning) {
    const warning = node('p', 'block-mismatch', facts.gist.warning);
    warning.setAttribute('role', 'alert');
    card.append(warning);
  }
  if (facts.gist?.note) card.append(node('p', 'block-says', facts.gist.note));
  const head = node('div', 'block-head');
  const mark = node('span', 'block-mark', '⊘');
  mark.setAttribute('aria-hidden', 'true');
  const title = node('h3', 'block-title', facts.title);
  head.append(mark, title);
  if (developer && facts.name) head.append(node('span', 'block-tool', facts.name));
  card.append(head);
  const why = [...facts.reasons];
  why.push(t(facts.self ? 'auto.blocked.self' : facts.finished ? 'auto.blocked.finished' : facts.words ? 'auto.blocked.why' : 'auto.blocked.clickOnly'));
  card.append(node('p', 'block-why', why.join(t('common.sentenceGap'))));
  if (facts.step) {
    // The code follows. A blocked step is never routine, so it is open: the sentence and the kind do not
    // stand in for reading it.
    const step = node('code', 'block-step', facts.step);
    step.title = facts.step;
    const fold = node('details', 'block-code');
    fold.open = true;
    fold.append(node('summary', '', facts.stepLabel || t('gist.showCode')), step);
    card.append(fold);
  }
  if (!facts.allow && !facts.keepSite) return card;
  // A finished job has nothing to allow now; a scheduled task's next runs are still the person's to decide.
  if (facts.allow) card.append(allowRow(doc, allowing, {busy, onAllow}));
  if (facts.keepSite) card.append(keepRow(doc, facts, {busy, onKeep}));
  if (error) {
    const problem = node('p', 'block-error', t('auto.blocked.failed', {problem: error}));
    problem.setAttribute('role', 'alert');
    card.append(problem);
  }
  return card;
}
