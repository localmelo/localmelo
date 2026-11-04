import {formatTime, gistSentence, language, nameList, t} from './i18n.mjs';

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
 * the review's `gist` names them. A review the model settled says what it found by its category. A
 * review with no sentences of its own reads by category in any language.
 */
export function reasonLines(review, categories) {
  const written = (Array.isArray(review.reasons) ? review.reasons : []).map(String).filter(Boolean);
  if (written.length && language() === 'en') return written;
  const modelFound = review.by === 'model' && !['beyond_request', 'opaque', 'unreviewed'].includes(categories[0]);
  if (modelFound) return [t('auto.reason.model', {what: t(`auto.words.${categories[0]}`)})];
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

/**
 * What a blocked call's card says, or null for a call auto mode did not block. `step` is the call's
 * code or command on one line; `gist` the sentence the card starts with, its warning and the model's note (or null);
 * `allow` whether Allow once may run it (never for `self`, never for a call a stopped turn left
 * unreviewed); `words` whether the person's words could allow it too.
 */
export function blockFacts({approval, name = '', arguments: args = {}, preview = '', requestId = ''} = {}) {
  if (!isBlock(approval)) return null;
  const categories = categoriesOf(approval);
  const first = categories[0] ?? '';
  const review = isObject(approval.review) ? approval.review : {};
  const self = categories.includes('self');
  const doubts = Array.isArray(review.doubts) && review.doubts.length > 0;
  return {
    requestId: String(approval.request_id ?? requestId ?? ''),
    stepKey: String(approval.step_key ?? ''),
    category: first,
    title: first ? t('auto.blocked.title', {title: t(`auto.title.${first}`)}) : t('auto.blocked.stopped'),
    reasons: reasonLines(review, categories),
    gist: gistWords(review.gist),
    name: String(name),
    step: stepText(name, args, preview),
    self,
    allow: Boolean(first) && !self && Boolean(approval.step_key),
    words: Boolean(first) && !doubts && categories.every((category) => WORDS_ALLOW.has(category)),
  };
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
 * failure's text.
 */
export function blockCard(doc, facts, {allowing = {state: 'open'}, busy = false, developer = false, onAllow = null, error = ''} = {}) {
  const node = (tag, className, text) => {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  };
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
  why.push(t(facts.self ? 'auto.blocked.self' : facts.words ? 'auto.blocked.why' : 'auto.blocked.clickOnly'));
  card.append(node('p', 'block-why', why.join(t('common.sentenceGap'))));
  if (facts.step) {
    // The code follows. A blocked step is never routine, so it is open: the sentence and the kind do not
    // stand in for reading it.
    const step = node('code', 'block-step', facts.step);
    step.title = facts.step;
    const fold = node('details', 'block-code');
    fold.open = true;
    fold.append(node('summary', '', t('gist.showCode')), step);
    card.append(fold);
  }
  if (!facts.allow) return card;
  const foot = node('div', 'block-foot');
  if (allowing.state === 'open') {
    const allow = node('button', 'outline-button block-allow', t(busy === 'allowing' ? 'auto.blocked.allowing' : 'auto.blocked.allow'));
    allow.type = 'button';
    allow.disabled = Boolean(busy);
    if (onAllow) allow.addEventListener('click', onAllow);
    foot.append(allow);
    foot.append(node('span', 'block-once', t(busy === 'running' ? 'auto.blocked.afterTurn' : 'auto.blocked.once')));
  } else {
    const words = {
      pending: () => t('auto.blocked.pending', {time: clock(allowing.at)}),
      used: () => t('auto.blocked.used', {time: clock(allowing.at)}),
      expired: () => t('auto.blocked.expired'),
    }[allowing.state];
    foot.append(node('span', 'block-done', words ? words() : ''));
  }
  card.append(foot);
  if (error) {
    const problem = node('p', 'block-error', t('auto.blocked.failed', {problem: error}));
    problem.setAttribute('role', 'alert');
    card.append(problem);
  }
  return card;
}
