/**
 * Bringing in what other assistants on this Mac already know: the main chat's first-run card, and the
 * section of the Memory tab, which share one list of files.
 *
 * The card asks `GET /api/memory/import/offer` first and looks for files (`GET /api/memory/import/scan`,
 * which writes nothing) only when the offer was never made. When files hold notes it shows one row per file:
 * its path (home as `~`), a checkbox (checked), the notes count, and a disclosure with the sample lines and
 * what was left out. "Bring them in" posts the checked files (`POST /api/memory/import`) and offers Undo
 * (`DELETE /api/memory/import/{batch_id}`); "Not now" records the offer as made (`DELETE
 * /api/memory/import/offer`). Nothing is brought in until the person presses the button, and nothing of a
 * file is shown but the backend's `sample`. A scan that finds nothing shows no card. The page shows or hides
 * the card with `show(visible)`; a hidden card asks nothing.
 */
import {formatTime, t} from "./i18n.mjs";

const REASONS = ["secret", "third_party", "wrong_owner", "duplicate", "mechanics"];

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function button(doc, className, text, onClick) {
  const node = el(doc, "button", className, text);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

/** A path with the home folder written as `~`. */
export function homePath(path, home) {
  const base = String(home ?? "").replace(/\/+$/, "");
  return base && String(path).startsWith(`${base}/`) ? `~${String(path).slice(base.length)}` : String(path);
}

/** What a file left out, in plain words ("2 left out: secrets, other people's details"); "" for nothing. */
export function leftOut(filtered) {
  const found = REASONS.map((reason) => [reason, Number(filtered?.[reason]) || 0]).filter(([, count]) => count > 0);
  const n = found.reduce((sum, [, count]) => sum + count, 0);
  const reasons = found.map(([reason]) => t(`importCard.reason.${reason}`)).join(t("importCard.join"));
  return n ? t("importCard.leftOut", {n, reasons}) : "";
}

/** A scan as the page uses it: the files with notes in them first, paths ready to show. */
export function readScan(data) {
  const sources = (Array.isArray(data?.sources) ? data.sources : []).map((source) => ({
    id: String(source.id),
    path: homePath(source.path, data.home),
    kept: Number(source.items_kept) || 0,
    sample: (Array.isArray(source.sample) ? source.sample : []).map(String),
    left: leftOut(source.filtered),
  }));
  return {batchId: String(data?.batch_id ?? ""), partial: data?.partial === true, sources};
}

/** Whether a scan holds a note to bring in. */
export const hasNotes = (scan) => scan.sources.some((source) => source.kept > 0);

/** The ids a scan starts with checked: every file with a note in it. */
export const allKept = (scan) => new Set(scan.sources.filter((source) => source.kept > 0).map((source) => source.id));

/** One row per file: checkbox, path, notes; a disclosure shows the sample lines and what was left out. */
export function sourceList({doc, scan, chosen, busy, toggle}) {
  const list = el(doc, "ul", "import-sources");
  for (const source of scan.sources) {
    const row = el(doc, "li", "import-source");
    const pick = el(doc, "label", "import-pick");
    const box = el(doc, "input");
    box.type = "checkbox";
    box.checked = chosen.has(source.id);
    box.disabled = busy || source.kept === 0;
    box.addEventListener("change", () => toggle(source.id, box.checked));
    pick.append(box, el(doc, "span", "import-path", source.path), el(doc, "span", "import-count", t("memoryTab.notes", {n: source.kept})));
    row.append(pick);
    if (source.sample.length || source.left) {
      const more = el(doc, "details", "import-more");
      more.append(el(doc, "summary", "", t("importCard.look")));
      if (source.sample.length) {
        const samples = el(doc, "ul", "import-samples");
        samples.append(...source.sample.map((line) => el(doc, "li", "", line)));
        more.append(samples);
      }
      if (source.left) more.append(el(doc, "p", "import-left", source.left));
      row.append(more);
    }
    list.append(row);
  }
  return list;
}

const failure = (error, fallback) => error?.message || t(fallback);

export function setupImportCard({root, api}) {
  const doc = root.ownerDocument;
  root.hidden = true;
  let shown = false;
  let started = false; // Asked once per page load: a failure is not asked again until the page reloads.
  let phase = "idle"; // idle | checking | offer | done | undone | gone
  let busy = false;
  let scan = null;
  let chosen = new Set();
  let result = null;
  let fault = "";

  async function look() {
    scan = readScan(await api("/memory/import/scan"));
    chosen = allKept(scan);
  }

  function render() {
    root.hidden = !shown || ["idle", "checking", "gone"].includes(phase);
    if (root.hidden) return;
    const refocus = root.contains?.(doc.activeElement);
    const parts = [];
    if (phase === "offer") {
      const bring = button(doc, "primary-button", busy ? t("importCard.working") : t("importCard.bring"), pressBring);
      const sync = () => { bring.disabled = busy || chosen.size === 0; };
      sync();
      const later = button(doc, "link-button", t("importCard.later"), decline);
      later.disabled = busy;
      const list = sourceList({doc, scan, chosen, busy, toggle: (id, on) => { if (on) chosen.add(id); else chosen.delete(id); sync(); }});
      const actions = el(doc, "div", "import-actions");
      actions.append(bring, later);
      parts.push(el(doc, "h2", "", t("importCard.title")), el(doc, "p", "", t("importCard.lead")), list);
      if (scan.partial) parts.push(el(doc, "p", "import-note", t("importCard.partial")));
      if (fault) parts.push(el(doc, "p", "import-fault", fault));
      parts.push(actions);
    } else {
      const line = el(doc, "p", "import-done");
      if (phase === "undone") line.append(t("importCard.undone"));
      else if (result.written) {
        const undoButton = button(doc, "link-button", t("memory.undo"), undo);
        undoButton.disabled = busy;
        line.append(el(doc, "span", "import-ok", t("importCard.done", {n: result.written})), " · ", undoButton);
      } else line.append(t("importCard.nothingNew"));
      parts.push(line, button(doc, "quiet-button import-close", t("importCard.close"), () => { phase = "gone"; render(); }));
      if (fault) parts.push(el(doc, "p", "import-fault", fault));
    }
    root.replaceChildren(...parts);
    root.setAttribute("aria-busy", String(busy));
    if (refocus) root.focus?.();
  }

  async function ask() {
    started = true;
    phase = "checking";
    try {
      if ((await api("/memory/import/offer")).pending === true) {
        await look();
        phase = hasNotes(scan) ? "offer" : "gone"; // nothing to offer: no card, and the offer stays open
      } else {
        phase = "gone";
      }
    } catch {
      phase = "gone"; // An offer nobody asked for does not nag.
    }
    render();
  }

  /** Run *work* with the card's buttons off; a failure is said on the card, in the server's words. */
  async function working(work, fallback) {
    busy = true;
    fault = "";
    render();
    try {
      await work();
    } catch (error) {
      fault = failure(error, fallback);
      // A scan the gateway forgot is read again, so the next press has a batch to write.
      if (error.status === 404 && phase === "offer") await look().catch(() => {});
    }
    busy = false;
    render();
  }

  const pressBring = () => chosen.size && !busy && working(async () => {
    const done = await api("/memory/import", {method: "POST", body: {batch_id: scan.batchId, source_ids: [...chosen]}});
    result = {batchId: scan.batchId, written: done.written_ids?.length ?? 0};
    phase = "done";
  }, "importCard.bringFailed");

  const undo = () => !busy && working(async () => {
    await api(`/memory/import/${encodeURIComponent(result.batchId)}`, {method: "DELETE"});
    phase = "undone";
  }, "importCard.undoFailed");

  async function decline() {
    phase = "gone";
    render();
    await api("/memory/import/offer", {method: "DELETE"}).catch(() => {}); // The card is gone either way.
  }

  return {
    /** Show or hide the card; the first time it is shown it asks whether to offer. */
    show(visible) {
      shown = visible;
      if (visible && !started) ask();
      else render();
    },
  };
}

/** The day of a note batch, "Oct 31" (with the year for another year). */
function dayOf(at) {
  const date = new Date(at * 1000);
  return formatTime(date, date.getFullYear() === new Date().getFullYear() ? "day" : "dayYear");
}

/** The batches the notes in `rows` were brought in by: `[{id, count, at}]`, newest first. */
export function batches(rows) {
  const found = new Map();
  for (const row of rows) {
    if (!row.imported || !row.batch_id) continue;
    const batch = found.get(row.batch_id) ?? {id: row.batch_id, count: 0, at: 0};
    batch.count += 1;
    batch.at = Math.max(batch.at, Number(row.created_at) || 0);
    found.set(row.batch_id, batch);
  }
  return [...found.values()].sort((left, right) => right.at - left.at);
}

/**
 * The Memory tab's "From other assistants": "Look for notes" scans and shows the card's list, "Bring them in"
 * writes the checked files; the batches already brought in (from the tab's notes) each have Undo. `rows()` is
 * the tab's loaded notes and `reload()` reads them again.
 */
export function importSection({doc, api, rows, reload, redraw}) {
  const root = el(doc, "section", "memory-import");
  let phase = "idle"; // idle | found
  let busy = false;
  let scan = null;
  let chosen = new Set();
  let line = "";
  let fault = "";

  async function working(work, fallback) {
    busy = true;
    fault = "";
    line = "";
    redraw();
    try {
      await work();
    } catch (error) {
      fault = failure(error, fallback);
    }
    busy = false;
    redraw();
  }
  const lookFor = () => !busy && working(async () => {
    scan = readScan(await api("/memory/import/scan"));
    chosen = allKept(scan);
    phase = hasNotes(scan) ? "found" : "idle";
    line = phase === "found" ? "" : t(scan.sources.length ? "importCard.nothingNew" : "memoryTab.import.none");
  }, "memoryTab.import.lookFailed");
  const bring = () => !busy && working(async () => {
    const done = await api("/memory/import", {method: "POST", body: {batch_id: scan.batchId, source_ids: [...chosen]}});
    line = done.written_ids?.length ? t("importCard.done", {n: done.written_ids.length}) : t("importCard.nothingNew");
    phase = "idle";
    await reload();
  }, "importCard.bringFailed");
  const take = (id) => !busy && working(async () => {
    await api(`/memory/import/${encodeURIComponent(id)}`, {method: "DELETE"});
    line = t("importCard.undone");
    await reload();
  }, "importCard.undoFailed");

  function draw() {
    const parts = [el(doc, "h3", "memory-sub", t("memoryTab.import.title")), el(doc, "p", "memory-line", t("memoryTab.import.lead"))];
    if (phase === "found") {
      const go = button(doc, "primary-button", busy ? t("importCard.working") : t("importCard.bring"), bring);
      const sync = () => { go.disabled = busy || chosen.size === 0; };
      sync();
      parts.push(sourceList({doc, scan, chosen, busy, toggle: (id, on) => { if (on) chosen.add(id); else chosen.delete(id); sync(); }}));
      if (scan.partial) parts.push(el(doc, "p", "import-note", t("importCard.partial")));
      const actions = el(doc, "div", "memory-actions");
      const cancel = button(doc, "quiet-button", t("memoryTab.cancel"), () => { phase = "idle"; redraw(); });
      cancel.disabled = busy;
      actions.append(go, cancel);
      parts.push(actions);
    } else {
      const look = button(doc, "outline-button", busy ? t("memoryTab.import.looking") : t("memoryTab.import.look"), lookFor);
      look.disabled = busy;
      parts.push(look);
    }
    if (line) parts.push(el(doc, "p", "import-note", line));
    if (fault) parts.push(el(doc, "p", "memory-fault", fault));
    const past = batches(rows());
    if (past.length) {
      const list = el(doc, "ul", "import-batches");
      for (const batch of past) {
        const item = el(doc, "li", "import-batch");
        const words = t("memoryTab.import.batch", {notes: t("memoryTab.notes", {n: batch.count}), day: dayOf(batch.at)});
        const undoButton = button(doc, "link-button", t("memory.undo"), () => take(batch.id));
        undoButton.disabled = busy;
        item.append(el(doc, "span", "", words), undoButton);
        list.append(item);
      }
      parts.push(list);
    }
    root.replaceChildren(...parts);
  }
  return {root, draw};
}
