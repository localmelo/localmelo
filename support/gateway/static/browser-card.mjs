/**
 * The browser card: what a chat's browser says in the conversation, and the dock that keeps it in sight.
 *
 * When melo browses, the chat shows one card for the run, in the reply like a message: the live page, what melo is
 * doing, who has the page, and one button, **Open browser**, which (like a click on the picture) shows the page big at
 * once in the Browser view that covers the chat (browser-live.mjs). A finished card's picture opens its last picture the
 * same way. A
 * task's browser shows the same card in the chat that asked, under **Working for this chat** and the task's goal, with
 * **Stop**, which ends that task; **Done** gives a page the person has back to that run. The
 * gateway writes the card as an `attention.browser` event (support/gateway/console/browser_cards.py): `readBrowser`
 * and `mergeBrowsers` keep the newest row of each card by its key, from `GET /turns` and from the chat's own event
 * stream. The card says only what the program wrote: a state (`working`, `waiting` for the person, `you` while they
 * have the page, `done`, `stopped`), the kind of step (`action`) and the site (a bare host).
 *
 * A card's node is made once and kept (`sync`), so a redraw of the conversation moves the same node and the same
 * canvas rather than building another. It watches its source on the page's one stream (browser-stream.mjs) only
 * while it is on screen and the run is active; a card that scrolls out of view hands the page to the dock, a bar
 * at the top of the chat that watches the same source, so a source is still one socket and one decoder. A picture
 * that is not arriving never reads as live: the card says `Not updating` or `Reconnecting…` and the picture dims.
 * When the run ends the card keeps the last picture this page painted, or says what it can.
 */
import {t} from './i18n.mjs';
import {maker} from './dom.mjs';

/** The states in which a run uses the browser: the card watches its source, and the chat's chip names it. */
export const ACTIVE = new Set(['working', 'waiting', 'you']);
const STATES = new Set([...ACTIVE, 'done', 'stopped']);
const ACTIONS = new Set(['open', 'read', 'press', 'type', 'wait', 'search']);
// The glyph of each state, the same on the card, the dock and the panel; its word is the one thing that says it in text.
const GLYPHS = {working: '⊕', waiting: '!', you: '‖', done: '✓', stopped: '■', stale: '◌'};
const WORDS = {
  working: 'browser.state.working',
  waiting: 'browser.state.waiting',
  you: 'browser.inControl',
  done: 'browser.state.done',
  stopped: 'browser.state.stopped',
};

/** The one word for a state, in the page's language. */
export const stateWord = (state) => t(WORDS[state] ?? WORDS.working);

/** A browser card as the page keeps it, from a `browser` row of `GET /turns` or an `attention.browser` frame; null if unusable. */
export function readBrowser(row) {
  const data = row?.data && typeof row.data === 'object' ? {...row.data, id: row.id, ts: row.ts} : row;
  const id = Number(data?.id);
  const key = String(data?.card_key ?? '');
  if (!Number.isFinite(id) || !key || !STATES.has(data?.state)) return null;
  return {
    id,
    key,
    // The chat that shows the card, the chat or task whose browser it is, and the run and browser generation that name its source.
    owner: String(data.owner_chat_id ?? ''),
    session: String(data.source_session_id ?? data.owner_chat_id ?? ''),
    task: data.source_kind === 'task',
    goal: String(data.goal ?? '').slice(0, 120),
    turn: Number(data.source_turn) || 0,
    run: String(data.run_id ?? ''),
    gen: String(data.browser_session_id ?? ''),
    state: data.state,
    action: ACTIONS.has(data.action) ? data.action : '',
    site: String(data.site ?? '').slice(0, 120),
    by: data.by === 'you' ? 'you' : '',
  };
}

/** *cards* with *more* added: of each card (by its key) the newest row, in the order of the newest rows. */
export function mergeBrowsers(cards, more) {
  const byKey = new Map(cards.map((card) => [card.key, card]));
  for (const row of more) {
    const card = readBrowser(row);
    const had = card && byKey.get(card.key);
    if (card && (!had || card.id >= had.id)) byKey.set(card.key, card);
  }
  return [...byKey.values()].sort((left, right) => left.id - right.id);
}

/** The source of a card on the stream: its browser, at its run and browser generation, and for a task the chat it shows in. */
export const sourceOf = (card) => ({session: card.session, gen: card.gen, run: card.run, ...(card.task ? {owner: card.owner} : {})});

/**
 * What a card shows, from the card (the gateway's words), the stream's state (`{meta, link, picture}`, or null while
 * nothing is watched) and whether this page has painted a picture of it. `own` is the state that colours it (one
 * map, in the stylesheet); `fresh` says the picture is arriving: a picture that is not is dimmed, and one whose
 * stream broke or stopped changing says so. `size` is `full` while the run is active, `kept` once it ended with a
 * picture on the page, else `line`.
 */
export function browserLook(card, stream = null, {retained = false} = {}) {
  const active = ACTIVE.has(card.state);
  const meta = stream?.meta ?? null;
  const gone = active && (meta?.state === 'closed' || meta?.state === 'idle');
  const picture = retained && !gone;
  const fresh = Boolean(stream?.picture) && stream?.link === 'open' && meta?.state === 'live';
  const broken = active && picture && (stream?.link === 'retry' || meta?.state === 'stale');
  const own = broken ? 'stale' : card.state;
  let sub = card.site;
  if (broken && stream?.link === 'retry') sub = t('browser.reconnecting');
  else if (!broken && card.state === 'you') sub = t('browser.state.paused');
  else if (card.state === 'stopped' && card.by === 'you') sub = t('browser.state.byYou');
  let note = '';
  if (active && !picture) note = gone ? t(`browser.closed.${meta.reason || 'tab'}`) : t('browser.loading.note');
  return {
    own,
    glyph: GLYPHS[own],
    word: broken ? t('browser.stale.title') : stateWord(card.state),
    sub,
    now: card.state === 'working' && !broken && card.action ? t(`browser.action.${card.action}`) : '',
    note,
    picture,
    fresh: !active || !picture || fresh,
    size: active ? 'full' : picture ? 'kept' : 'line',
    openable: active && !gone,
    // A task's card names who it works for while it does; once over, its goal is what it was.
    name: card.task ? (active ? t('browser.forThisChat') : card.goal || t('browser.title')) : t('browser.title'),
    goal: card.task && active ? card.goal : '',
    stoppable: active && card.task,
    givable: card.state === 'you',
  };
}

/** A text node's words, written only when they change. */
function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

/**
 * The cards of the open chat. *streams* is the page's transport (browser-stream.mjs), *open(card, picture)* shows a
 * card's source in the Browser view (*picture* is the card's canvas: all a finished card has to show), *stop(card)*
 * ends the task it names and *done(card)* gives the person's page back to its run:
 * each acts on that card's source, and answers whether it went. *observe(onChange)* makes a visibility watcher, `{add(node), remove(node)}`, calling
 * `onChange(node, visible)`; without one every card counts as visible. *dock* is the page's dock bar, `{root,
 * canvas, mark, text, jump, open}`.
 */
export function setupBrowserCards({doc, streams, open, stop = null, done = null, observe = null, dock = null}) {
  const {node, button} = maker(doc);
  const recs = new Map();
  let order = [];
  const watcher = observe ? observe((target, visible) => {
    const rec = [...recs.values()].find((candidate) => candidate.root === target);
    if (!rec) return;
    rec.visible = visible;
    // The page changes hands without a gap: whoever takes it over watches before whoever leaves lets go, so the
    // source keeps its socket.
    if (visible) {
      reconcile(rec);
      refreshDock();
    } else {
      refreshDock();
      reconcile(rec);
    }
  }) : null;

  /** One press of *act* at a time: the button waits for the card to change, and comes back if the press did nothing. */
  async function press(rec, control, act) {
    control.disabled = true;
    if ((await act?.(rec.card)) === false) control.disabled = false;
  }

  function make(card) {
    const root = node('section', 'bcard');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', t('browser.title'));
    // The keys come back to a card when the Browser view that opened from it closes.
    root.tabIndex = -1;
    const mark = node('span', 'own-mark');
    mark.setAttribute('aria-hidden', 'true');
    const word = node('span', 'own-word');
    const sub = node('span', 'bcard-sub');
    const line = node('span', 'bcard-state');
    line.append(word, sub);
    const name = node('span', 'bcard-name', t('browser.title'));
    const goal = node('span', 'bcard-goal');
    const titles = node('span', 'bcard-titles');
    titles.append(name, goal, line);
    const head = node('header', 'bcard-head');
    head.append(mark, titles);
    const canvas = node('canvas', 'own-ring bcard-canvas');
    canvas.setAttribute('role', 'img');
    // The picture opens the browser too; the button below is the one stop on the keyboard.
    const stage = button('bcard-picture', '', () => open(rec.card, rec.canvas), {label: t('browser.open')});
    stage.tabIndex = -1;
    stage.append(canvas);
    const note = node('p', 'bcard-note');
    const nowText = node('span', 'bcard-now-text');
    const now = node('p', 'bcard-now');
    now.append(node('span', 'own-dot'), nowText);
    const actions = node('div', 'bcard-actions');
    const opener = button('outline-button wide', t('browser.open'), () => open(rec.card, rec.canvas));
    const give = button('outline-button', t('browser.giveBack'), () => press(rec, give, done), {label: t('browser.giveBackLabel')});
    const halt = button('quiet-button', t('common.stop'), () => press(rec, halt, stop));
    actions.append(opener, give, halt);
    root.append(head, stage, note, now, actions);
    const rec = {
      card, root, mark, name, word, sub, goal, canvas, stage, note, now, nowText, actions, opener, give, halt,
      stream: null, retained: false, visible: observe ? null : true, unwatch: null,
      sink: {canvas, update: (state) => { rec.stream = state; if (state.picture) rec.retained = true; paint(rec); refreshDock(); }},
    };
    watcher?.add(root);
    return rec;
  }

  function paint(rec) {
    const look = browserLook(rec.card, rec.stream, {retained: rec.retained});
    const {root} = rec;
    root.dataset.own = look.own;
    root.dataset.size = look.size;
    root.dataset.fresh = String(look.fresh);
    setText(rec.mark, look.glyph);
    setText(rec.name, look.name);
    setText(rec.word, look.word);
    setText(rec.sub, look.sub ? ` · ${look.sub}` : '');
    rec.goal.hidden = !look.goal;
    setText(rec.goal, look.goal);
    rec.stage.hidden = !look.picture;
    rec.canvas.setAttribute('aria-label', `${t('browser.picture')}: ${rec.card.site}`);
    rec.note.hidden = !look.note;
    setText(rec.note, look.note);
    rec.now.hidden = !look.now;
    setText(rec.nowText, look.now);
    rec.opener.hidden = !look.openable;
    rec.give.hidden = !(look.givable && done);
    rec.halt.hidden = !(look.stoppable && stop);
    rec.actions.hidden = rec.opener.hidden && rec.give.hidden && rec.halt.hidden;
  }

  /** Watch the card's source while it is on screen and its run is active, and let go otherwise. */
  function reconcile(rec) {
    const want = rec.visible === true && ACTIVE.has(rec.card.state);
    if (want && !rec.unwatch) rec.unwatch = streams.watch(sourceOf(rec.card), rec.sink);
    else if (!want && rec.unwatch) {
      rec.unwatch();
      rec.unwatch = null;
      rec.stream = null;
      paint(rec);
    }
  }

  // ── The dock ──

  let docked = null;
  const dockSink = dock && {
    canvas: dock.canvas,
    update: (state) => { docked.stream = state; if (state.picture) docked.retained = true; paintDock(); },
  };

  /** The dock shows the newest active card while it is not on screen, and watches its source. */
  function refreshDock() {
    if (!dock) return;
    const rec = order.map((key) => recs.get(key)).filter((found) => found && ACTIVE.has(found.card.state)).at(-1);
    const away = rec && rec.visible === false && rec.root.isConnected !== false;
    if (!away) {
      if (docked) {
        docked.unwatch?.();
        docked = null;
      }
      dock.root.hidden = true;
      return;
    }
    if (!docked || docked.key !== rec.card.key) {
      docked?.unwatch?.();
      docked = {key: rec.card.key, rec, stream: null, retained: false, unwatch: null};
      docked.unwatch = streams.watch(sourceOf(rec.card), dockSink);
    }
    dock.root.hidden = false;
    paintDock();
  }

  function paintDock() {
    if (!docked) return;
    const look = browserLook(docked.rec.card, docked.stream, {retained: docked.retained});
    dock.root.dataset.own = look.own;
    dock.root.dataset.fresh = String(look.fresh);
    dock.canvas.hidden = !look.picture;
    setText(dock.mark, look.glyph);
    setText(dock.text, [docked.rec.card.task ? docked.rec.card.goal || look.name : look.name, look.word, look.now || look.sub].filter(Boolean).join(' · '));
    dock.open.hidden = !look.openable;
  }
  if (dock) {
    dock.open.addEventListener('click', () => docked && open(docked.rec.card, docked.rec.canvas));
    dock.jump.addEventListener('click', () => docked?.rec.root.scrollIntoView?.({block: 'center'}));
  }

  return {
    /**
     * Make the cards of the open chat current: a node for each (made once, then kept), the others let go. Call it
     * before the conversation is drawn, and again when the open chat changes.
     */
    sync(cards) {
      const keys = new Set(cards.map((card) => card.key));
      for (const [key, rec] of recs) {
        if (keys.has(key)) continue;
        rec.unwatch?.();
        watcher?.remove(rec.root);
        recs.delete(key);
      }
      order = cards.map((card) => card.key);
      for (const card of cards) {
        let rec = recs.get(card.key);
        if (!rec) {
          rec = make(card);
          recs.set(card.key, rec);
        }
        rec.card = card;
        paint(rec);
        reconcile(rec);
      }
      refreshDock();
    },

    /** The node of the card with this key (the same node at every redraw), or null. */
    nodeOf: (key) => recs.get(key)?.root ?? null,

    /** Give the keys back to a card: its Open browser button, else its picture, else the card. False when there is no such card. */
    focus(key) {
      const rec = recs.get(key);
      if (!rec) return false;
      const target = [rec.opener, rec.stage].find((found) => !found.hidden) ?? rec.root;
      target.focus?.();
      return true;
    },
  };
}
