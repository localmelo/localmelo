/**
 * The welcome's setup card: set a model server up on this Mac after one OK, connect the one that already
 * runs, or point to the other way, an account or a key, which the card hands to Settings (`openSettings("model")`).
 *
 * While the page shows it (`show(true)`), it asks `GET /api/setup/detect` and draws what it
 * found: a server with a model to use, a server that needs an API key, or nothing. "Use this
 * model" saves the choice as the settings dialog does (`PUT /api/settings` with the revision
 * just read), runs the connection test, and calls `connected()` so the page reads its health
 * again. Hidden, the card asks nothing. It never picks a model for the user: with several
 * models and none known to be loaded, the user chooses from the list.
 *
 * With nothing found, the card asks `GET /api/setup/plan` for what this Mac can run and shows the
 * terminal's list: each row says what, how big and where, and a row that is not here yet can be turned
 * off (the list is asked again with `?skip=` for its totals and what leaving a row out leads to). One
 * button starts `POST /api/setup/start`; the job then rides on detect (`job`), so the card keeps its one loop
 * of asking, and draws a row per step with its progress, in place so a focused Stop stays focused. A job that
 * ends `done` and was seen running connects to the model it set up, as "Use this model" does; one that was
 * stopped, failed or left the chat model out says so above the list, which asks again and offers to continue.
 * When there is no plan (no uv, no network, not a Mac), the commands that start a server by hand stay.
 * While it says that no server runs, or a job runs, it asks again every few seconds, so the model appears
 * without a click; a hidden card or tab asks nothing.
 */
import {t} from "./i18n.mjs";
import {maker} from "./dom.mjs";
import {STATE_MARKS} from "./marks.mjs";

const RECHECK_MS = 3000;
const JOB_MS = 1500;
// One shape for each state of a step: the glyph differs, not only the colour.
const MARKS = {waiting: STATE_MARKS.waiting, running: STATE_MARKS.running, done: STATE_MARKS.ok, skipped: "·", stopped: STATE_MARKS.stopped, failed: STATE_MARKS.failed};

const host = (url) => {
  try {
    return new URL(url).host;
  } catch {
    return url || "";
  }
};
const firstLine = (error) => String(error.message).split(/\r?\n/)[0];
// What a person agrees to by pressing the button: the same list again means the same OK.
const shape = (list) => JSON.stringify([
  list.pick?.id, list.rows?.map((r) => [r.id, r.state, r.size, r.where, r.off]), list.download, list.notes, list.action, list.can_set_up,
]);

export function setupModelCard({
  root, api, openSettings, connected, schedule = setTimeout, cancel = clearTimeout,
  writeText = (text) => globalThis.navigator.clipboard.writeText(text),
}) {
  const doc = root.ownerDocument;
  root.hidden = true;
  let shown = false;
  let version = 0; // A newer request, or a hide, drops an older answer.
  let phase = "idle"; // idle | checking | ready | connecting | failed
  let data = null;
  let picked = "";
  let choosing = false;
  let failure = "";
  let refocus = false;
  let timer = null; // The next look, while the card says that no server runs or a job runs.
  let plan; // undefined: not asked; null: being asked; else the answer of GET /setup/plan.
  let off = new Set(); // The ids of the rows the person turned off.
  let startError = "";
  let starting = false;
  let waiting = false; // melo_serve is being asked again for the list on screen.
  let seen = 0; // The job this card saw run: its model is connected when it ends.
  let joined = 0; // The job whose model was connected.
  let live = null; // The view that is drawn in place: {key, update}.

  const {node: el, button} = maker(doc);

  /** The job that is running, if any. */
  const running = () => (data?.job?.state === "running" ? data.job : null);

  /** What an ended job says above the list: it stopped, failed, or left the chat model out. */
  function banner() {
    const job = data?.job;
    if (!job || job.state === "running") return null;
    if (job.state === "cancelled") return {kind: "stopped", title: t("setup.stoppedTitle"), text: t("setup.stoppedText")};
    if (job.state === "failed") return {kind: "failed", title: t("setup.failedTitle"), text: job.error};
    return job.saved ? null : {kind: "nochat", title: t("setup.noChatTitle"), text: t("setup.noChatText")};
  }

  /** The words of the card's state, and its main action. */
  function copy() {
    const who = data?.name || t("setup.aServer");
    const where = host(data?.url);
    if (phase === "checking") return {title: t("setup.looking")};
    if (phase === "connecting") {
      return {title: t("setup.connecting", {server: who}), text: t("setup.connectingText")};
    }
    if (phase === "failed") {
      const title = data ? t("setup.connectFailed", {server: who}) : t("setup.lookFailed");
      return {title, text: failure, label: t("setup.tryAgain"), run: data ? use : detect};
    }
    const job = running();
    if (job) return {title: t("setup.running", {model: job.name}), text: t("setup.runningText"), view: "job"};
    if (data.state === "found") {
      return {title: t("setup.found", {server: who}), text: t("setup.runningAt", {address: where}), label: t("setup.use"), run: use};
    }
    if (data.state === "needs_key") {
      return {title: t("setup.needsKey"), text: t("setup.needsKeyText", {address: where}), label: t("setup.checkAgain"), run: detect};
    }
    if (plan?.available) return {title: t("setup.planTitle"), view: "plan"};
    // Until the list is known, and when there is none, the commands that start a server by hand stay.
    const answered = data.url !== "";
    const {name, commands, note, guide} = data.start;
    const by = {
      text: answered ? t("setup.answeredAt", {address: where, problem: data.problem}) : t("setup.startOne", {server: name}),
      commands: answered ? [] : commands,
      note: answered ? "" : t("setup.note", {note}),
      guide,
    };
    if (!plan) return {title: t("setup.planning"), ...by};
    return {
      title: answered ? t("setup.noModel", {server: who}) : t("setup.none"),
      ...by,
      lead: plan.reason,
      aside: [plan.detail, plan.hint].filter(Boolean).join(" "),
      label: t("setup.checkAgain"),
      run: detect,
    };
  }

  function modelRow(id, onPick) {
    const row = el(onPick ? "label" : "div", "setup-model");
    if (onPick) {
      const radio = el("input");
      radio.type = "radio";
      radio.name = "setup-model";
      radio.value = id;
      radio.checked = id === picked;
      radio.addEventListener("change", () => onPick(id));
      row.append(radio);
    }
    row.append(el("span", "setup-model-id", id));
    if (data.loaded.includes(id)) row.append(el("span", "setup-loaded", t("setup.loaded")));
    return row;
  }

  // ── The list a person says yes to ──

  /** One row of the list: a box to turn it off when it can be, what it is, how big, where it goes. */
  function planRow(row) {
    const node = el("li", "setup-row");
    const lead = el("span", "setup-lead");
    // A row that has to be set up shows its box on and locked, so the rows line up and the person sees why it cannot be left out.
    let box = null;
    if (row.state !== "here") {
      box = el("input");
      box.type = "checkbox";
      box.disabled = !row.off;
      box.addEventListener("change", () => turn(row.id, box.checked));
      lead.append(box);
    }
    const mark = el("span", "setup-mark");
    mark.setAttribute("aria-hidden", "true");
    const name = el("span", "setup-name");
    const size = el("span", "setup-size");
    const kind = el("span", "setup-kind");
    const where = el("span", "setup-where");
    const detail = el("span", "setup-detail");
    detail.append(kind, where);
    // The whole row is the box's label, so a click anywhere on it turns the row on or off; the label draws no box of its own.
    const pick = el("label", "setup-pick");
    pick.append(lead, name, size, detail);
    node.append(pick);
    const fill = (found) => {
      node.dataset.state = found.state;
      name.textContent = found.name;
      size.textContent = found.size;
      kind.textContent = found.kind;
      where.textContent = found.where;
      if (box) {
        box.checked = !off.has(found.id);
        box.setAttribute("aria-label", t("setup.rowOn", {name: found.name}));
      } else {
        mark.textContent = STATE_MARKS.ok;
        lead.replaceChildren(mark);
      }
    };
    fill(row);
    return {node, fill};
  }

  /** The list: built once, then redrawn in place as rows are turned off. */
  function planNodes() {
    const mac = el("p", "setup-mac");
    const intro = el("p");
    const rows = el("ul", "setup-rows");
    rows.setAttribute("aria-label", t("setup.rows"));
    const total = el("p", "setup-total");
    const notes = el("div", "setup-notes");
    const wait = el("p", "setup-wait");
    wait.setAttribute("role", "status");
    const fault = el("p", "setup-fault");
    fault.setAttribute("role", "alert");
    const made = new Map();
    const go = button("primary-button", "", () => start());
    const update = () => {
      mac.textContent = plan.mac;
      intro.textContent = plan.intro;
      for (const row of plan.rows) {
        if (!made.has(row.id)) {
          made.set(row.id, planRow(row));
          rows.append(made.get(row.id).node);
        } else {
          made.get(row.id).fill(row);
        }
      }
      total.textContent = [plan.download, plan.free].filter(Boolean).join(" ");
      notes.replaceChildren(...plan.notes.map((line) => el("p", "setup-note", line)));
      wait.textContent = waiting ? t("setup.planning") : "";
      wait.hidden = !waiting;
      fault.textContent = startError;
      fault.hidden = !startError;
      go.textContent = plan.action;
      go.disabled = starting || !plan.can_set_up;
    };
    update();
    return {nodes: [mac, intro, rows, total, notes, wait, fault], go, update};
  }

  /** Turn a row off or on, then ask for the list worded for it: its totals and what it leads to. */
  async function turn(id, on) {
    if (on) off.delete(id);
    else off.add(id);
    const mine = version;
    waiting = true; // After a minute melo_serve is asked again, which takes seconds: say so.
    live?.update?.();
    try {
      const found = await api(planPath());
      if (mine !== version) return;
      plan = found;
      startError = "";
    } catch (error) {
      if (mine !== version) return;
      startError = firstLine(error);
    } finally {
      waiting = false;
    }
    live?.update?.();
  }

  const planPath = () => "/setup/plan" + (off.size ? "?skip=" + encodeURIComponent([...off].join(",")) : "");

  async function askPlan() {
    const mine = version;
    plan = null;
    render();
    try {
      let found = await api(off.size ? "/setup/plan" : planPath());
      if (mine !== version) return;
      if (off.size) {
        // A row stays off only if the new list still lets the person turn it off; the list is then worded for those.
        const open = new Set((found.rows ?? []).filter((r) => r.off).map((r) => r.id));
        off = new Set([...off].filter((id) => open.has(id)));
        if (off.size) found = await api(planPath());
        if (mine !== version) return;
      }
      plan = found;
    } catch {
      if (mine !== version) return;
      plan = {available: false, reason: "", detail: "", hint: ""}; // No list: the commands stay.
    }
    render();
  }

  /** One OK: start the job for the list on screen. */
  async function start(retried = false) {
    if (starting || !plan?.available) return;
    const mine = version;
    starting = true;
    startError = "";
    live?.update?.();
    let stale = false;
    try {
      const started = await api("/setup/start", {method: "POST", body: {pick: plan.pick.id, skip: [...off]}});
      if (mine !== version) return;
      data = {...data, job: started.job};
      seen = started.job.id;
    } catch (error) {
      if (mine !== version) return;
      stale = error.code === "no_plan"; // The list is out of date: ask again, and let the person look before the next OK.
      startError = stale ? t("setup.refreshed") : firstLine(error);
    } finally {
      starting = false;
    }
    if (stale) {
      // The list is out of date: ask again. The same list is the same OK, so it starts at once; a changed one waits for the person.
      const agreed = shape(plan);
      await askPlan();
      if (!retried && mine === version && plan?.available && shape(plan) === agreed) {
        startError = "";
        return start(true);
      }
    } else {
      render();
    }
    watch();
  }

  // ── The job ──

  /** One step of the job: a mark, its name, and its line or its bar. */
  function jobRow(row) {
    const node = el("li", "setup-step");
    const mark = el("span", "setup-mark");
    mark.setAttribute("aria-hidden", "true");
    const name = el("span", "setup-name");
    const word = el("span", "setup-state");
    const line = el("span", "setup-line");
    const bar = el("progress", "setup-bar");
    bar.setAttribute("max", "100");
    bar.hidden = true;
    node.append(mark, name, word, line, bar);
    const fill = (found) => {
      node.dataset.state = found.state;
      mark.textContent = MARKS[found.state] ?? "·";
      name.textContent = found.name;
      word.textContent = t(`setup.state.${found.state}`); // Ours, so translated; the line is the job's own words.
      line.textContent = found.state === "running" ? found.progress : found.line;
      const bytes = found.state === "running" && found.total_bytes > 0;
      bar.hidden = found.state !== "running";
      bar.setAttribute("aria-label", found.name);
      if (bytes) bar.value = Math.round((100 * found.done_bytes) / found.total_bytes);
      else bar.removeAttribute("value"); // No size yet: the bar is indeterminate.
    };
    fill(row);
    return {node, fill};
  }

  /** The steps and the Stop button: built once, then updated in place by each look. */
  function jobNodes() {
    const rows = el("ul", "setup-steps");
    rows.setAttribute("aria-label", t("setup.steps"));
    const made = new Map();
    const stopButton = button("outline-button", t("setup.stop"), stop);
    const update = () => {
      const job = running();
      if (!job) return;
      for (const row of job.rows) {
        if (!made.has(row.id)) {
          made.set(row.id, jobRow(row));
          rows.append(made.get(row.id).node);
        } else {
          made.get(row.id).fill(row);
        }
      }
      stopButton.textContent = job.stopping ? t("setup.stopping") : t("setup.stop");
      stopButton.disabled = job.stopping;
    };
    update();
    const actions = el("div", "setup-actions");
    actions.append(stopButton);
    return {nodes: [rows, actions], update};
  }

  async function stop() {
    const mine = version;
    try {
      const stopped = await api("/setup/cancel", {method: "POST"});
      if (mine !== version) return;
      data = {...data, job: stopped.job};
    } catch {
      // It ended on its own meanwhile: the next look says how.
    }
    live?.update?.();
  }

  /** The connection a job that ended `done` leads to: the model it set up, as "Use this model" does. */
  function maybeConnect() {
    const job = data?.job;
    if (job?.state !== "done" || !job.saved || job.id !== seen || joined === job.id) return;
    if (data.state !== "found" || !picked) return;
    joined = job.id;
    use();
  }

  /** What follows an answer: the list when nothing runs, or the connection a finished job leads to. */
  function follow() {
    if (data?.state === "none" && !running() && plan === undefined) askPlan();
    maybeConnect();
  }

  function render() {
    const view = copy();
    const job = data?.job;
    const key = `${view.view}|${phase}|${job?.id}|${job?.state}|${view.view === "plan" ? banner()?.kind : ""}`;
    if (view.view && live?.key === key) {
      live.update();
      return;
    }
    refocus = root.contains(doc.activeElement);
    const {title, text, lead, aside, commands = [], note, label, run, guide} = view;
    const settled = phase === "ready" || phase === "failed";
    const found = settled && data?.state === "found" && !view.view;
    const list = found && phase === "ready" && (choosing || !data.suggested);
    const heading = el("h2", "", title);
    heading.id = "setup-title";
    const parts = [heading];
    if (view.view === "plan") {
      const said = banner();
      if (said) {
        const block = el("div", "setup-banner");
        block.dataset.kind = said.kind;
        block.setAttribute("role", "status");
        block.append(el("strong", "", said.title));
        if (said.text) block.append(el("span", "", said.text));
        parts.push(block);
      }
    }
    if (lead) parts.push(el("p", "setup-lead-in", lead));
    if (aside) parts.push(el("p", "setup-aside", aside)); // The words of whoever said no: not translated, so smaller.
    if (text) parts.push(el("p", "", text));
    // The commands, each on one line (the block scrolls, so a flag never breaks): one click selects
    // them all, a focusable group lets the keyboard scroll it, and a button copies them.
    if (commands.length) {
      const block = el("pre", "setup-commands", commands.join("\n"));
      block.tabIndex = 0;
      block.setAttribute("role", "group");
      block.setAttribute("aria-label", t("setup.commands"));
      parts.push(block, el("p", "", note));
    }
    let main = label ? button("primary-button", label, run) : null;
    let built = null;
    if (view.view === "job") {
      built = jobNodes();
      parts.push(...built.nodes);
    } else if (view.view === "plan") {
      built = planNodes();
      main = built.go;
      parts.push(...built.nodes);
    }
    live = built ? {key, update: built.update} : null;
    if (list) {
      const group = el("fieldset", "setup-models");
      group.append(el("legend", "", t("setup.pick")));
      group.append(...data.models.map((id) => modelRow(id, (value) => {
        picked = value;
        main.disabled = false;
      })));
      parts.push(group);
      if (data.loaded.length < data.models.length) {
        parts.push(el("p", "", t("setup.slowLoad")));
      }
      main.disabled = !picked;
    } else if (found && phase === "ready") {
      parts.push(modelRow(data.suggested));
    }
    const actions = el("div", "setup-actions");
    if (main) actions.append(main);
    if (found && !list && data.models.length > 1) actions.append(button("link-button", t("setup.another"), choose));
    if (phase === "failed" || (phase === "ready" && data?.state !== "found" && view.view !== "job")) actions.append(button("link-button", t("setup.address"), () => openSettings("model")));
    if (commands.length) {
      const copyButton = button("link-button", t("setup.copy"), async () => {
        try {
          await writeText(commands.join("\n"));
          copyButton.textContent = t("setup.copied");
        } catch {
          copyButton.textContent = t("setup.copyFailed");
        }
      });
      actions.append(copyButton);
    }
    if (guide) {
      const link = el("a", "link-button", t("setup.guide"));
      Object.assign(link, {href: guide, target: "_blank", rel: "noopener noreferrer"});
      actions.append(link);
    }
    if (actions.children.length) parts.push(actions);
    // The other way to a model, named on every state of the card but the one that is connecting.
    if (phase !== "connecting") {
      const other = el("div", "setup-other");
      other.append(
        el("h3", "", t("setup.accountTitle")),
        el("p", "", t("setup.accountText")),
        button("outline-button", t("setup.account"), () => openSettings("model")),
      );
      parts.push(other);
    }
    root.replaceChildren(...parts);
    root.setAttribute("aria-busy", String(phase === "checking" || phase === "connecting" || plan === null));
    if (refocus) root.focus(); // The control that was clicked is gone: the next Tab reaches the first one.
  }

  function choose() {
    choosing = true;
    phase = "ready";
    render();
  }

  function stopWatching() {
    if (timer !== null) cancel(timer);
    timer = null;
  }

  /** Whether the answer differs from what the card shows: the server, or the job and its steps. */
  const changed = (found) => found.state !== data.state || found.url !== data.url || found.problem !== data.problem
    || JSON.stringify(found.job) !== JSON.stringify(data.job);

  /** While no server runs or a job runs, ask again every few seconds; the card changes only when the answer does. */
  function watch() {
    stopWatching();
    if (!shown || phase !== "ready" || !(data?.state === "none" || running())) return;
    const mine = version;
    timer = schedule(async () => {
      timer = null;
      if (mine !== version) return;
      if (!doc.hidden) {
        try {
          const found = await api("/setup/detect");
          if (mine !== version) return;
          if (changed(found)) {
            const ended = running() && found.job?.state !== "running";
            data = found;
            picked = found.suggested || "";
            if (found.job?.state === "running") seen = found.job.id;
            if (found.state !== "none" || ended) plan = undefined; // What is here changed: ask again.
            render();
            follow();
          }
        } catch {
          // The card stays as it is; the next look tries again.
        }
      }
      if (mine === version) watch();
    }, running() ? JOB_MS : RECHECK_MS);
  }

  async function detect() {
    const mine = ++version;
    phase = "checking";
    data = null;
    picked = "";
    choosing = false;
    plan = undefined;
    off = new Set();
    startError = "";
    live = null;
    render();
    try {
      const found = await api("/setup/detect");
      if (mine !== version) return;
      data = found;
      picked = found.suggested || "";
      if (found.job?.state === "running") seen = found.job.id;
      phase = "ready";
    } catch (error) {
      if (mine !== version) return;
      failure = firstLine(error);
      phase = "failed";
    }
    render();
    watch();
    if (phase === "ready") follow();
  }

  async function use() {
    if (!data || !picked) return;
    const mine = ++version;
    phase = "connecting";
    render();
    try {
      const settings = await api("/settings");
      const candidate = {
        chat_backend: data.backend,
        embedding_backend: settings.embedding_backend || "none",
        workspace: settings.workspace,
        sections: {[data.backend]: {chat_url: data.url, chat_model: picked}},
      };
      await api("/settings", {method: "PUT", body: {...candidate, revision: settings.revision}});
      const test = await api("/settings/test", {body: candidate});
      if (test.ok !== true) throw new Error(test.error || t("setup.noAnswer"));
      if (mine !== version) return;
      phase = "ready"; // Saved and answering: the page may hide the card now.
      await connected();
    } catch (error) {
      if (mine !== version) return;
      failure = firstLine(error);
      phase = "failed";
      render();
    }
  }

  return {
    /** Show or hide the card: showing asks the gateway what is on this Mac, hiding forgets it. */
    show(visible) {
      if (visible === shown) return;
      shown = visible;
      root.hidden = !visible;
      version += 1;
      stopWatching();
      if (visible) {
        detect();
      } else {
        phase = "idle";
        data = null;
        live = null;
      }
    },
    /** Whether a connection to the chosen server is being made or has just failed: the page keeps the card up. */
    holding: () => data !== null && (phase === "connecting" || phase === "failed"),
  };
}
