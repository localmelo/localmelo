import {t} from './i18n.mjs';

/** The two ways melo works in a chat (#339): asks first, or works on its own and blocks risky steps. */
export const MODES = ['ask', 'auto'];

/** A mode read from a session or the settings: `auto`, else `ask`. */
export const modeOf = (value) => (value === 'auto' ? 'auto' : 'ask');

// The grants of a site question (#465): opening or reading a site, and clicking or choosing on it.
const SITE_GRANTS = [['browser-act:', 'act'], ['browser:', 'open']];

/**
 * The chat's grants of a site, in the order given: `{name, kind, site}` with `kind` `open` or `act` and
 * `site` the host. Any other grant (a tool for the session) is left out.
 */
export function siteGrants(names = []) {
  const found = [];
  for (const name of names) {
    const text = String(name);
    const match = SITE_GRANTS.find(([prefix]) => text.startsWith(prefix));
    if (!match) continue;
    const rest = text.slice(match[0].length);
    let site = rest;
    try {
      site = new URL(rest).hostname || rest;
    } catch {
      // A grant's name holds an origin; keep what it says when it is not one.
    }
    found.push({name: text, kind: match[1], site});
  }
  return found;
}

/**
 * Wire the composer's Ask/Auto chip and its menu (`mode-picker`, `mode-btn`, `mode-text` and an empty
 * `mode-menu` in index.html; the menu's content is drawn here). `state()` gives `{mode, fallback,
 * visible, busy, grants, everywhere}`: the chat's mode (or the one the next new chat gets), the mode new chats start
 * in, whether the chip shows, whether a switch is being saved, the names of the chat's grants, and which of
 * them hold because the main chat allowed them (taking one back takes it away in every chat). `pick(mode)`
 * saves a choice and resolves when it is kept. `openSettings()` opens Settings at Chats and folders.
 * `remove(name)` takes back one grant and resolves when it is gone. The menu is a `menu` of two
 * `menuitemradio`s, then the sites allowed in this chat (#465), each with a Remove `menuitem`: arrows move,
 * Enter or Space picks, Escape closes.
 */
export function setupModeChip({doc, state, pick, openSettings, remove = null}) {
  const $ = (id) => doc.getElementById(id);
  let open = false;
  let items = [];
  // Every control the arrows move through: the two modes, then each grant's Remove.
  let controls = [];

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
    const grants = remove ? grantsList(state().grants ?? [], state().everywhere ?? []) : [];
    $('mode-menu').setAttribute('aria-labelledby', 'mode-menu-title');
    $('mode-menu').replaceChildren(title, ...items, ...grants, foot);
  }

  // The sites this chat allowed for its rest (#465), each with Remove; nothing when there are none.
  function grantsList(names, everywhere = []) {
    const found = siteGrants(names);
    const fromMain = new Set(everywhere);
    const removes = [];
    if (!found.length) {
      controls = [...items];
      return [];
    }
    const box = node('div', 'mode-grants');
    box.setAttribute('role', 'group');
    const heading = node('p', 'mode-grants-title', t('auto.menu.grants'));
    heading.id = 'mode-grants-title';
    box.setAttribute('aria-labelledby', heading.id);
    box.append(heading);
    for (const [index, grant] of found.entries()) {
      const what = t(grant.kind === 'act' ? 'auto.menu.grantAct' : 'auto.menu.grantOpen', {site: grant.site});
      const row = node('div', 'mode-grant');
      const drop = node('button', 'link-button mode-grant-remove', t('auto.menu.remove'));
      drop.type = 'button';
      drop.setAttribute('role', 'menuitem');
      drop.setAttribute('aria-label', t('auto.menu.removeOne', {what}));
      drop.addEventListener('click', () => take(grant.name, drop));
      drop.addEventListener('keydown', keys);
      removes.push(drop);
      const words = node('span', 'mode-grant-what', what);
      if (fromMain.has(grant.name)) {
        // The main chat allowed it: say so, and that Remove reaches every chat.
        const from = node('span', 'mode-item-words mode-grant-from', t('auto.menu.fromMain'));
        from.id = `mode-grant-from-${index}`;
        drop.setAttribute('aria-describedby', from.id);
        const text = node('span', 'mode-item-text');
        text.append(words, from);
        row.append(text, drop);
      } else row.append(words, drop);
      box.append(row);
    }
    controls = [...items, ...removes];
    return [box];
  }

  async function take(name, button) {
    button.disabled = true;
    try {
      await remove(name);
    } finally {
      button.disabled = false;
    }
    if (!open) return;
    menu();
    (controls[0] ?? null)?.focus();
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
    const moving = controls.length ? controls : items;
    const at = moving.indexOf(doc.activeElement);
    const to = {ArrowDown: (at + 1) % moving.length, ArrowUp: (at + moving.length - 1) % moving.length, Home: 0, End: moving.length - 1}[event.key];
    if (to !== undefined) {
      event.preventDefault?.();
      moving[to].focus();
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
