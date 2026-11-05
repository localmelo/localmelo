import {t} from './i18n.mjs';
import {maker} from './dom.mjs';

/**
 * melo_serve's chat models in the console: the list the header's model chip and the composer's model button
 * open when melo_serve is the model server, the same rows in Settings → Model, and `/model`.
 *
 * The rows are `GET /api/serve`'s, worded by the gateway in the page's language, so the list reads as the
 * terminal's and the app's. A switch is `POST /api/serve/switch`; its state rides on the health poll
 * (`switch`), so the chip, the list and Settings follow it with no poll of their own. While one runs the other
 * rows are disabled, and the row it switches to says what it waits for, with Cancel. A pick that cannot load
 * is dimmed with its reason and never chosen.
 */

const ACTIVE = ['waiting', 'loading'];

/** Whether melo_serve serves the chats: its chat models switch from here. */
export const servesModels = (health) => health?.model?.backend === 'melo_serve';

/** The switch under way in *health*, or null. */
export const activeSwitch = (health) => (ACTIVE.includes(health?.switch?.state) ? health.switch : null);

/** A row's words as nodes: the fit word in its own span, coloured, unless *plain*. */
export function wordNodes(doc, pick, plain = false) {
  const {node} = maker(doc);
  const words = String(pick.words ?? '');
  const fit = String(pick.fit ?? '');
  if (plain || !fit || !words.endsWith(fit)) return [node('span', '', words)];
  const word = node('span', 'pick-fit', fit);
  if (pick.tight) word.dataset.fit = 'tight';
  return [node('span', '', words.slice(0, words.length - fit.length)), word];
}

/**
 * Wire the list. `openers` are `{button, menu}` pairs: the header chip with its menu below it, and the
 * composer's model button with one above the composer. `state()` gives `{health}`, the last `GET /api/health`.
 * `changed(switch)` takes a switch the page should show before the next health poll brings it; `notify(text)`
 * says a refusal or a switch that did not finish. Returns the list's moves, which Settings and `/model` use.
 */
export function setupServe({doc, api, state, notify, changed, openers}) {
  const {node, button} = maker(doc);
  let rows = null;
  let foot = '';
  let failure = '';
  let loading = null;
  let open = -1;
  let items = [];
  let seen = {id: 0, state: ''};
  const listeners = new Set();

  const health = () => state().health;
  const current = () => rows?.find((row) => row.current) ?? null;

  async function load() {
    loading ||= api('/serve')
      .then((body) => {
        rows = Array.isArray(body.picks) ? body.picks : [];
        foot = String(body.foot ?? '');
        failure = '';
      })
      .catch((error) => {
        failure = error.message;
      })
      .finally(() => {
        loading = null;
        redraw();
      });
    return loading;
  }

  function redraw() {
    if (open >= 0) draw();
    for (const listener of listeners) listener();
  }

  function row(pick, sw) {
    const target = sw && sw.pick === pick.id;
    if (target) {
      // The pick being switched to: the pulsing dot in the check's place, what it waits for, and Cancel.
      const item = node('div', 'mode-item pick-waiting');
      item.setAttribute('role', 'menuitemradio');
      item.setAttribute('aria-checked', 'false');
      item.setAttribute('aria-busy', 'true');
      const slot = node('span', 'mode-check pick-slot');
      slot.setAttribute('aria-hidden', 'true');
      slot.append(node('span', 'pick-busy'));
      const text = node('span', 'mode-item-text');
      text.append(node('span', 'mode-item-name', pick.name), node('span', 'mode-item-words', sw.detail || sw.title));
      item.append(slot, text);
      if (sw.state === 'waiting') {
        const stop = button('outline-button pick-cancel', t('common.cancel'), () => cancel());
        stop.addEventListener('keydown', keys);
        item.append(stop);
        items.push(stop);
      }
      return item;
    }
    const blocked = Boolean(sw) || !pick.loadable;
    const item = button('mode-item', null, () => choose(pick));
    item.setAttribute('role', 'menuitemradio');
    item.setAttribute('aria-checked', String(Boolean(pick.current)));
    if (blocked) item.setAttribute('aria-disabled', 'true');
    item.dataset.pick = pick.id;
    const check = node('span', 'mode-check', '✓');
    check.setAttribute('aria-hidden', 'true');
    const text = node('span', 'mode-item-text');
    const words = node('span', 'mode-item-words');
    words.append(...wordNodes(doc, pick, blocked));
    text.append(node('span', 'mode-item-name', pick.name), words);
    item.append(check, text);
    item.addEventListener('keydown', keys);
    if (!blocked) items.push(item);
    return item;
  }

  function draw() {
    const menu = openers[open].menu;
    const sw = activeSwitch(health());
    // The arrows move through what can be chosen: the picks that load, or Cancel while a switch waits.
    items = [];
    const drawn = (rows ?? []).map((pick) => row(pick, sw));
    const said = failure || (rows === null ? t('settings.state.checking') : '');
    const parts = said ? [node('p', 'mode-menu-title', said)] : [];
    menu.setAttribute('aria-label', t('serve.listLabel'));
    menu.replaceChildren(...parts, ...drawn, node('p', 'mode-menu-foot', foot || t('serve.foot')));
  }

  function toggle(index, next, focus = false) {
    if (next && !servesModels(health())) return false;
    const before = open;
    open = next ? index : -1;
    openers.forEach(({button: opener, menu}, at) => {
      menu.hidden = at !== open;
      opener.setAttribute('aria-expanded', String(at === open));
    });
    if (!next || before === index) return next;
    draw();
    load().then(() => {
      if (focus && open === index) (items.find((item) => item.getAttribute('aria-checked') === 'true') ?? items[0])?.focus();
    });
    return true;
  }

  async function choose(pick) {
    if (activeSwitch(health()) || !pick.loadable) return;
    if (pick.current) {
      close(true);
      return;
    }
    try {
      await switchTo(pick.id);
    } catch (error) {
      notify(error.message);
    }
  }

  /** Start the switch to the pick *id*; throws what the gateway refuses. */
  async function switchTo(id) {
    const body = await api('/serve/switch', {body: {pick: id}});
    changed(body.switch);
    redraw();
    return body.switch;
  }

  async function cancel() {
    try {
      const body = await api('/serve/switch/cancel', {method: 'POST'});
      changed(body.switch);
    } catch (error) {
      notify(error.message);
    }
    redraw();
    openers[open]?.button.focus?.();
  }

  function close(focus = false) {
    const was = open;
    toggle(-1, false);
    if (focus && was >= 0) openers[was].button.focus?.();
  }

  function keys(event) {
    const at = items.indexOf(doc.activeElement);
    const to = {ArrowDown: (at + 1) % items.length, ArrowUp: (at + items.length - 1) % items.length, Home: 0, End: items.length - 1}[event.key];
    if (to !== undefined && items.length) {
      event.preventDefault?.();
      items[to].focus();
    } else if (event.key === 'Escape') {
      event.preventDefault?.();
      event.stopPropagation?.();
      close(true);
    } else if (event.key === 'Tab') close();
  }

  /** The health changed: a switch that ended reads the rows again, and one that failed is said once. */
  function update() {
    const now = health();
    if (!servesModels(now)) {
      if (open >= 0) close();
      return;
    }
    const sw = now?.switch;
    if (sw && (sw.id !== seen.id || sw.state !== seen.state)) {
      const ended = seen.id === sw.id && ACTIVE.includes(seen.state) && !ACTIVE.includes(sw.state);
      seen = {id: sw.id, state: sw.state};
      if (ended && sw.state === 'failed') notify(t('serve.failed', {name: sw.name, error: sw.error}));
      if (ended) load();
      redraw();
    }
    // What is served changed (here, or with melo-serve in a terminal): the rows say so again.
    if (rows === null || (now?.model?.name && current() && current().model !== now.model.name)) {
      if (!loading) load();
    }
  }

  openers.forEach(({button: opener}, index) => {
    opener.addEventListener('keydown', (event) => {
      if (!servesModels(health()) || (event.key !== 'ArrowDown' && event.key !== 'ArrowUp')) return;
      event.preventDefault?.();
      toggle(index, true, true);
    });
  });

  return {
    serves: () => servesModels(health()),
    load,
    update,
    switchTo,
    cancel,
    close,
    /** Open or close the list of opener *index*; false when it is not melo_serve's (the caller opens Settings). */
    click: (index) => (open === index ? (close(), true) : toggle(index, true, true)),
    isOpen: () => open >= 0,
    rows: () => rows,
    current,
    /** The pick a typed word names: its short name, id, name or served model. */
    named: (typed) => {
      const wanted = String(typed).trim().toLowerCase();
      return (rows ?? []).find((pick) => [pick.short, pick.id, pick.name, pick.model].some((name) => String(name ?? '').toLowerCase() === wanted)) ?? null;
    },
    /** The chip's words: the model and its backend, or the switch under way. */
    chip: () => {
      const sw = activeSwitch(health());
      if (sw) return {name: sw.title, backend: ''};
      const model = health()?.model;
      const now = current();
      return {name: now?.name || model?.name || '', backend: now?.backend_name || ''};
    },
    onChange: (listener) => listeners.add(listener),
  };
}
