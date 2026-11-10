/**
 * One step of work in plain words, for a person: `Reading notes.md`, `Read notes.md` once done, and `Read notes.md`
 * as what a step that did not finish was to do (`ask`). Never a tool of ours by name, a command line or typed text.
 */
import {t} from './i18n.mjs';

const TESTS = /(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|(?:npm|pnpm|yarn|bun)(?: run)? test|(?:cargo|go|swift|deno|mix) test|python3? -m (?:pytest|unittest)|make (?:test|check))(?:$|[\s;&|)])/;
const base = (path) => String(path ?? '').replace(/\/+$/, '').split('/').pop();
// A chat's own steps (and a task's question to it): the verb, and what it acts on.
const CHAT_STEPS = {
  task_schedule: ['save', 'steps.aSchedule'],
  task_list: ['look', 'steps.theTasks'],
  task_get: ['read', 'steps.aTask'],
  task_message: ['send', 'steps.aTask'],
  task_set_priority: ['priority', 'steps.aTask'],
  task_answer: ['reply', 'steps.aTask'],
  task_continue: ['continue', 'steps.aTask'],
  ask_person: ['asks', 'steps.you'],
  ask_owner: ['asks', 'steps.itsChat'],
};
const TASK_ACTIONS = new Set(['stop', 'pause', 'resume']);
const SHOWN = 60;
const NAME_SHOWN = 40;
/** A search's or a task's words as a step shows them: `“asyncio TaskGroup”`, cut when long. */
function quoted(text) {
  const words = String(text ?? '').replace(/\s+/g, ' ').trim();
  return words ? `“${words.length > SHOWN ? `${words.slice(0, SHOWN - 1)}…` : words}”` : '…';
}

// The tools whose step, said with no call to name, would not say what the tool does (a search with no words, a keep
// that reads as failed, two memory tools that read alike): their own line in the details panel.
const TOOL_LINES = {
  remember: 'steps.ask.keep',
  web_search: 'steps.tool.webSearch',
  search_files: 'steps.tool.searchFiles',
  search_memory: 'steps.tool.searchMemory',
  open_memory: 'steps.tool.openMemory',
  task_control: 'steps.tool.taskControl',
};

/**
 * What a tool does for the person, as the details panel lists it (`Read a file`, `Run a command`): a step of it as it
 * would ask (`stepWords`, with no call to name), or its own line in `TOOL_LINES`.
 */
export function toolWords(name) {
  const tool = String(name ?? '').startsWith('mcp__') ? String(name).split('__').pop() : String(name ?? '');
  return TOOL_LINES[tool] ? t(TOOL_LINES[tool]) : stepWords(tool, {}, 'ask');
}

/** A task tool's own check turned the call down, so it did nothing (the server's `refused_step`). */
export function refusedStep(output) {
  return /^Not done/.test(String(output ?? '').trim());
}

/**
 * The words the server's `step_said` says: `done` is false while it runs, true once done, or `ask` for a step that did
 * not finish. Only a finished call's *output* says what it came to: a task tool's own check refused it ("Not done: …"),
 * a note was not kept ("Not remembered"), a message to a finished task started a new one ("Started task").
 */
export function stepWords(name, args, done = false, output = '') {
  const tool = String(name ?? '').startsWith('mcp__') ? String(name).split('__').pop() : String(name ?? '');
  const a = args && typeof args === 'object' ? args : {};
  // `steps.now.<verb>` while it runs, `steps.done.<verb>` once done; `what` is a name or `steps.<thing>`.
  const tense = done === 'ask' ? 'ask' : done ? 'done' : 'now';
  const said = String(output ?? '').trim();
  if (done && refusedStep(said)) return t('steps.refused', {step: stepWords(name, args, 'ask')});
  if (done === true && tool === 'task_message' && /^Started task/.test(said)) return stepWords('task_create', {}, true);
  const say = (verb, what = '') => t(`steps.${tense}.${verb}`, {what});
  const file = base(a.path) || t('steps.aFile');
  if (tool === 'file_read') return say('read', file);
  if (tool === 'file_write') return say('write', file);
  if (tool === 'list_files') return say('look', ['', '.'].includes(base(a.path)) ? t('steps.theFolder') : base(a.path));
  if (tool === 'file_edit') return say('edit', file);
  if (tool === 'git_status') return say('read', t('steps.gitState'));
  if (tool === 'shell_exec') return say('run', t(TESTS.test(String(a.command ?? '').replace(/\s+/g, ' ')) ? 'steps.theTests' : 'steps.aCommand'));
  if (tool === 'python_exec') return say('run', 'Python');
  if (tool === 'remember') {
    if (!done) return t('steps.now.keep');
    return t(done === 'ask' || /^Not remembered/.test(said) ? 'steps.failed.keep' : 'steps.done.keep');
  }
  if (tool === 'recall' || tool === 'search_memory' || tool === 'open_memory') return say('remember');
  if (tool === 'browser_open' || tool === 'web_fetch') {
    let host = '';
    try { host = new URL(String(a.url)).hostname.replace(/^www\./, ''); } catch { /* not a page address */ }
    return host ? say('read', host) : say('open', t('steps.aPage'));
  }
  if (tool === 'web_search') return say('search', quoted(a.query));
  if (tool === 'browser_follow') return say('follow', t('steps.aLink'));
  if (tool === 'browser_click' || tool === 'browser_click_point') return say('click', t('steps.thePage'));
  if (tool === 'browser_type') return say('type', t('steps.aField'));
  if (tool === 'browser_select') return say('choose', t('steps.aList'));
  if (tool === 'browser_attach') return say('attach', base(a.file) || t('steps.aFile'));
  if (tool === 'browser_screenshot') return say('lookAt', t('steps.thePage'));
  if (tool.startsWith('browser_')) return say('read', t('steps.thePage'));
  if (tool === 'search_files') return say('find', quoted(a.pattern));
  if (tool === 'task_create') return say('start', a.title ? quoted(a.title) : t('steps.aTask'));
  if (tool === 'task_control') return say(TASK_ACTIONS.has(a.action) ? a.action : 'use', t('steps.aTask'));
  if (CHAT_STEPS[tool]) return say(CHAT_STEPS[tool][0], t(CHAT_STEPS[tool][1]));
  // An outside tool, or a name a model made up, reads by its own name.
  return say('use', tool.replace(/_/g, ' ').trim().slice(0, NAME_SHOWN) || t('steps.aTool'));
}
