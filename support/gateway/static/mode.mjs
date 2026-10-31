import {t} from './i18n.mjs';

/** The two ways melo works in a chat (#339): asks first, or works on its own and blocks risky steps. */
export const MODES = ['ask', 'auto'];

/** A mode read from a session or the settings: `auto`, else `ask`. */
export const modeOf = (value) => (value === 'auto' ? 'auto' : 'ask');

/**
 * Wire the composer's Ask/Auto chip and its menu (`mode-picker`, `mode-btn`, `mode-text` and an empty
 * `mode-menu` in index.html; the menu's content is drawn here). `state()` gives `{mode, fallback,
 * visible, busy}`: the chat's mode (or the one the next new chat gets), the mode new chats start in,
 * whether the chip shows, and whether a switch is being saved. `pick(mode)` saves a choice and
 * resolves when it is kept. `openSettings()` opens Settings at Chats and folders. The menu is a `menu`
 * of two `menuitemradio`s: arrows move, Enter or Space picks, Escape closes.
 */
export function setupModeChip({doc, state, pick, openSettings}) {
  const $ = (id) => doc.getElementById(id);
  let open = false;
  let items = [];

  function node(tag, className, text) {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  }

  function menu() {
    const {mode, fallback} = state();
    const title = node('p', 'mode-menu-title', t('auto.menu.title'));
    title.id = 'mode-menu-title';
    items = MODES.map((name) => {
      const item = node('button', 'mode-item');
      item.type = 'button';
      item.setAttribute('role', 'menuitemradio');
      item.setAttribute('aria-checked', String(name === mode));
      item.dataset.mode = name;
      const check = node('span', 'mode-check', '✓');
      check.setAttribute('aria-hidden', 'true');
      const text = node('span', 'mode-item-text');
      text.append(node('span', 'mode-item-name', t(`auto.mode.${name}`)), node('span', 'mode-item-words', t(`auto.mode.${name}Words`)));
      item.append(check, text);
      item.addEventListener('click', () => choose(name));
      item.addEventListener('keydown', keys);
      return item;
    });
    const foot = node('p', 'mode-menu-foot', `${t('auto.menu.default', {mode: t(`auto.mode.${fallback}`)})} `);
    const link = node('a', '', t('auto.menu.change'));
    link.href = '#settings';
    link.addEventListener('click', (event) => {
      event.preventDefault?.();
      toggle(false);
      openSettings();
    });
    link.addEventListener('keydown', keys);
    foot.append(link);
    $('mode-menu').setAttribute('aria-labelledby', 'mode-menu-title');
    $('mode-menu').replaceChildren(title, ...items, foot);
  }

  function draw() {
    const {mode, visible, busy} = state();
    const chip = $('mode-btn');
    $('mode-picker').hidden = !visible;
    chip.dataset.mode = mode;
    chip.disabled = Boolean(busy);
    const name = t(`auto.mode.${mode}`);
    if ($('mode-text').textContent !== name) $('mode-text').textContent = name;
    chip.setAttribute('aria-label', t('auto.chip', {mode: name}));
    if (!visible) toggle(false);
  }

  function toggle(next, focus = false) {
    if (next === open) return;
    open = next;
    if (next) menu();
    $('mode-menu').hidden = !next;
    $('mode-btn').setAttribute('aria-expanded', String(next));
    if (next && focus) (items.find((item) => item.getAttribute('aria-checked') === 'true') ?? items[0])?.focus();
  }

  async function choose(mode) {
    toggle(false);
    $('mode-btn').focus();
    if (mode !== state().mode) await pick(mode);
    draw();
  }

  function keys(event) {
    const at = items.indexOf(doc.activeElement);
    const to = {ArrowDown: (at + 1) % items.length, ArrowUp: (at + items.length - 1) % items.length, Home: 0, End: items.length - 1}[event.key];
    if (to !== undefined) {
      event.preventDefault?.();
      items[to].focus();
    } else if (event.key === 'Escape') {
      event.preventDefault?.();
      event.stopPropagation?.();
      toggle(false);
      $('mode-btn').focus();
    } else if (event.key === 'Tab') toggle(false);
  }

  $('mode-btn').addEventListener('click', () => toggle(!open, true));
  $('mode-btn').addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault?.();
      toggle(true, true);
    }
  });

  return {update: draw, isOpen: () => open, close: () => toggle(false)};
}
