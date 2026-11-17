import {t} from './i18n.mjs';
import {maker} from './dom.mjs';

/**
 * The chat models in the console: the list the header's model chip and the composer's model button open, the
 * same rows in Settings → Model (melo_serve's), and `/model`. The list holds melo_serve's chat models when it is
 * the model server and, once a hosted service has a key, one group per server and service (`GET /api/chat-models`:
 * the local server's models and each keyed service's, the one in use marked). With no hosted key and another
 * model server it does not exist, as before.
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

// A row of a group is picked by this id: the group, a separator, the model ("" for a server that serves its own).
const SEP = '\u0000';
const choiceId = (group, model) => `${group.id}${SEP}${model}`;

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
export function setupServe({doc, api, state, notify, changed, openers, modelChanged = () => {}}) {
  const {node, button} = maker(doc);
  let rows = null;
  let foot = '';
  let failure = '';
  let loading = null;
  let open = -1;
  let items = [];
  let seen = {id: 0, state: ''};
  const listeners = new Set();
  // `GET /api/chat-models`: null before it was asked; `full` says the models are listed (not only the groups).
  let choices = null;
  let choicesLoading = null;
  let choicesFull = false;

  const health = () => state().health;
  const current = () => rows?.find((row) => row.current) ?? null;
  // Whether a hosted service has a key: what Settings said at the start (`keyed`) until the list itself is asked.
  const hostedKeys = () => (choices !== null ? choices.hosted === true : Boolean(state().keyed?.length));
  const groups = () => (choices?.hosted === true ? choices.groups : []);
  /** Whether there is a list: melo_serve's, or the groups of the services that have a key. */
  const lists = () => servesModels(health()) || hostedKeys();

  /** The groups (with their models when `full`), asked once at the start and again when it matters. */
  function loadChoices(full) {
    if (choicesLoading) return full && !choicesFull ? choicesLoading.then(() => loadChoices(true)) : choicesLoading;
    choicesFull = full;
    choicesLoading = api(full ? '/chat-models' : '/chat-models?models=0')
      .then((body) => {
        const listed = Array.isArray(body.groups) ? body.groups : [];
        choices = {hosted: body.hosted === true && listed.length > 0, groups: listed, full};
      })
      .catch(() => {})
      .finally(() => {
        choicesLoading = null;
        redraw();
      });
    return choicesLoading;
  }

  async function load() {
    const groupsLoaded = loadChoices(true);
    if (!servesModels(health())) return groupsLoaded;
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
    return Promise.all([loading, groupsLoaded]);
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

  /** A group's heading and its rows: the models, the one row of a server that serves its own, or why the listing failed. */
  function groupNodes(group, sw) {
    const out = [node('p', 'mode-menu-title', group.label)];
    const rowOf = (model, name, on) => {
      const item = button('mode-item', null, () => chooseChoice(group, model));
      item.setAttribute('role', 'menuitemradio');
      item.setAttribute('aria-checked', String(on));
      if (sw) item.setAttribute('aria-disabled', 'true');
      item.dataset.pick = choiceId(group, model);
      const check = node('span', 'mode-check', '✓');
      check.setAttribute('aria-hidden', 'true');
      const text = node('span', 'mode-item-text');
      text.append(node('span', 'mode-item-name', name));
      item.append(check, text);
      item.addEventListener('keydown', keys);
      if (!sw) items.push(item);
      return item;
    };
    if (group.problem) {
      const item = node('div', 'mode-item');
      item.setAttribute('aria-disabled', 'true');
      const text = node('span', 'mode-item-text');
      text.append(node('span', 'mode-item-words', group.problem));
      item.append(node('span', 'mode-check'), text);
      out.push(item);
    } else if (group.serves) {
      out.push(rowOf('', group.label, Boolean(group.current)));
    } else {
      for (const model of group.models ?? []) out.push(rowOf(model, model, group.current === model));
    }
    return out;
  }

  function draw() {
    const menu = openers[open].menu;
    const sw = activeSwitch(health());
    // The arrows move through what can be chosen: the picks that load, or Cancel while a switch waits.
    items = [];
    const serving = servesModels(health());
    const drawn = serving ? (rows ?? []).map((pick) => row(pick, sw)) : [];
    for (const group of groups()) drawn.push(...groupNodes(group, sw));
    const waiting = serving ? rows === null : choices?.full !== true;
    const said = failure || (waiting ? t('settings.state.checking') : '');
    const parts = said ? [node('p', 'mode-menu-title', said)] : [];
    menu.setAttribute('aria-label', t(hostedKeys() ? 'serve.listLabelAll' : 'serve.listLabel'));
    menu.replaceChildren(...parts, ...drawn, node('p', 'mode-menu-foot', foot || t('serve.foot')));
  }

  function toggle(index, next, focus = false) {
    if (next && !lists()) return false;
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

  /** Pick a model of a group: saved at once, used from the next message; a refusal is said. */
  async function chooseChoice(group, model) {
    if (activeSwitch(health())) return;
    const on = group.serves ? Boolean(group.current) : group.current === model;
    if (on) {
      close(true);
      return;
    }
    try {
      await switchChoice(group.id, model);
      close(true);
    } catch (error) {
      notify(error.message);
    }
  }

  /** Switch the chat to *model* of the group *id*; the chip and the composer name it at once. Throws what the gateway refuses. */
  async function switchChoice(id, model) {
    const body = await api('/chat-models/switch', {body: {group: id, model}});
    if (body.model) modelChanged(body.model);
    await loadChoices(true);
    return body;
  }

  /** Start the switch to the pick *id*; throws what the gateway refuses. */
  async function switchTo(id) {
    if (id.includes(SEP)) {
      const [group, model] = id.split(SEP);
      await switchChoice(group, model);
      return {title: t('slash.modelSet', {model: model || groups().find((item) => item.id === group)?.label || group})};
    }
    const body = await api('/serve/switch', {body: {pick: id}});
    // Seen from here on, so that one that fails before the next poll is said too.
    seen = {id: body.switch.id, state: body.switch.state};
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
    if (!lists()) {
      if (open >= 0) close();
      return;
    }
    if (!servesModels(now)) return;
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
      if (!lists() || (event.key !== 'ArrowDown' && event.key !== 'ArrowUp')) return;
      event.preventDefault?.();
      toggle(index, true, true);
    });
  });

  return {
    serves: lists,
    load,
    /**
     * The groups may have changed (a key was saved or removed, the connection was): ask again, in full while the list is
     * open. With no hosted key known and nothing asked before, nothing is asked: a local-only page makes no new request.
     */
    reloadChoices: (force = false) => (force || choices !== null || hostedKeys() ? loadChoices(open >= 0) : undefined),
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
      const is = (names) => names.some((name) => String(name ?? '').toLowerCase() === wanted);
      const served = (rows ?? []).find((pick) => is([pick.short, pick.id, pick.name, pick.model]));
      if (served) return served;
      for (const group of groups()) {
        const pick = (model, name, on) => ({id: choiceId(group, model), name, short: name, model, current: on, loadable: true, group: group.id});
        if (group.serves && is([group.id, group.label])) return pick('', group.label, Boolean(group.current));
        const model = (group.models ?? []).find((name) => is([name, `${group.id}:${name}`]));
        if (model !== undefined) return pick(model, model, group.current === model);
      }
      return null;
    },
    /** What `/model` says when a typed word names no row: melo_serve's own words, or the list's. */
    unknown: (typed) => t(hostedKeys() ? 'serve.errors.unknownModel' : 'serve.errors.unknownPick', {pick: typed}),
    /** The chip's words: the model and its backend, or the switch under way. */
    chip: () => {
      const sw = activeSwitch(health());
      if (sw) return {name: sw.title, backend: ''};
      const model = health()?.model;
      const now = current();
      return {name: now?.name || model?.name || '', backend: now?.backend_name || (hostedKeys() ? model?.backend_label || '' : '')};
    },
    onChange: (listener) => listeners.add(listener),
  };
}
