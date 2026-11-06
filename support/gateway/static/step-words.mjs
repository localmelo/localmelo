/**
 * One step of work in plain words, for a person: `Reading notes.md`, `Read notes.md` once done, and `Read notes.md`
 * as what a step that did not finish was to do (`ask`). Never a tool's name, a command line or typed text.
 */
import {t} from './i18n.mjs';

const TESTS = /(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|(?:npm|pnpm|yarn|bun)(?: run)? test|(?:cargo|go|swift|deno|mix) test|python3? -m (?:pytest|unittest)|make (?:test|check))(?:$|[\s;&|)])/;
const base = (path) => String(path ?? '').replace(/\/+$/, '').split('/').pop();

/** The words the server's `activity_text` says: `done` is false while it runs, true once done, or `ask` for a step that did not finish. */
export function stepWords(name, args, done = false) {
  const tool = String(name ?? '').startsWith('mcp__') ? String(name).split('__').pop() : String(name ?? '');
  const a = args && typeof args === 'object' ? args : {};
  // `steps.now.<verb>` while it runs, `steps.done.<verb>` once done; `what` is a name or `steps.<thing>`.
  const tense = done === 'ask' ? 'ask' : done ? 'done' : 'now';
  const say = (verb, what = '') => t(`steps.${tense}.${verb}`, {what});
  const file = base(a.path) || t('steps.aFile');
  if (tool === 'file_read') return say('read', file);
  if (tool === 'file_write') return say('write', file);
  if (tool === 'list_files') return say('look', ['', '.'].includes(base(a.path)) ? t('steps.theFolder') : base(a.path));
  if (tool === 'file_edit') return say('edit', file);
  if (tool === 'git_status') return say('read', t('steps.gitState'));
  if (tool === 'shell_exec') return say('run', t(TESTS.test(String(a.command ?? '').replace(/\s+/g, ' ')) ? 'steps.theTests' : 'steps.aCommand'));
  if (tool === 'python_exec') return say('run', 'Python');
  if (tool === 'recall' || tool === 'search_memory' || tool === 'open_memory' || tool === 'remember') return say('remember');
  if (tool === 'browser_open') {
    let host = '';
    try { host = new URL(String(a.url)).hostname.replace(/^www\./, ''); } catch { /* not a page address */ }
    return host ? say('read', host) : say('open', t('steps.aPage'));
  }
  if (tool === 'web_search') {
    const words = String(a.query ?? '').replace(/\s+/g, ' ').trim();
    return say('search', words ? `“${words.length > 60 ? `${words.slice(0, 59)}…` : words}”` : '…');
  }
  if (tool === 'browser_follow') return say('follow', t('steps.aLink'));
  if (tool === 'browser_click' || tool === 'browser_click_point') return say('click', t('steps.thePage'));
  if (tool === 'browser_type') return say('type', t('steps.aField'));
  if (tool === 'browser_select') return say('choose', t('steps.aList'));
  if (tool === 'browser_attach') return say('attach', base(a.file) || t('steps.aFile'));
  if (tool === 'browser_screenshot') return say('lookAt', t('steps.thePage'));
  if (tool.startsWith('browser_')) return say('read', t('steps.thePage'));
  return say('use', t('steps.aTool'));
}
