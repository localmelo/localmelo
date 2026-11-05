/**
 * The approval card: the question a tool call asks before it runs, with the step, what each answer
 * covers and the answers themselves. The chat and the Tasks page draw the same card; each passes the
 * `view` through which the card talks to it:
 *
 * - `scope`: "chat" or "task", for the words of what an answer covers;
 * - `main`: the card is in the main chat, where a kept answer counts in every chat;
 * - `workspace`: the label of the folder the request runs in, when the request has none;
 * - `rerender()`: the card changed itself (a folded box opened, the typing of a reason ended);
 * - `answer(request, state, choice)`: the person chose "once", "session", "always" or "deny";
 * - `reload(id)`: read the request's details again;
 * - `takeOver()`: the chat opens the Browser panel when the person agrees to take over a page.
 *
 * `states` keeps each request's state (the deny box, the folded code, an answer under way) by request
 * id, so that a redraw of the page does not lose it; the page owns it. `openWorkspace` is the label of
 * the folder of the chat that is open, for a request and a view that name none. `later(fn, ms)` is
 * `setTimeout`, which the page (and its tests) may replace.
 */
import { gistWords, reviewWords } from "./blocked.mjs";
import { maker } from "./dom.mjs";
import { t } from "./i18n.mjs";
import { formatBytes } from "./turns.mjs";

/** How long a new card holds every answer, so that a click meant for the page cannot answer it. */
export const GUARD_MS = 500;

export function approvalNow() {
  return globalThis.performance?.now?.() ?? Date.now();
}

/**
 * The milliseconds the guard still holds a card, in whole milliseconds and rounded up (0 once it has run
 * out). `setTimeout` takes whole milliseconds and cuts a fraction off, so a timer set for the exact rest
 * fired up to a millisecond too soon, saw the guard still up and left every button grey until the next
 * redraw, which in a waiting chat is half a minute away.
 */
export function guardLeft(state) {
  return Math.max(0, Math.ceil(GUARD_MS - (approvalNow() - state.mountedAt)));
}

// "5 lines removed, 1 added": the size of what a file write replaces.
function diffSummary({added, removed}) {
  const lines = (n) => t("diff.lines", { n });
  if (!added && !removed) return t("diff.none");
  if (!removed) return t("diff.added", { lines: lines(added) });
  return added ? t("diff.both", { lines: lines(removed), n: added }) : t("diff.removed", { lines: lines(removed) });
}

// The rows of a diff, numbered, each with a sign so that colour is not the only cue; the removed
// and added text sit in del and ins, which a screen reader names.
function diffBox(doc, rows) {
  const { node: el } = maker(doc);
  const box = el("div", "approval-content approval-diff");
  for (const row of rows) {
    const line = el("div", `diff-row diff-${row.kind}`);
    if (row.kind === "gap") {
      line.setAttribute("role", "separator");
      line.setAttribute("aria-label", t("diff.skipped"));
      line.append(el("span", "diff-num"), el("span", "diff-sign"), el("span", "diff-text", "…"));
    } else {
      const tag = row.kind === "removed" ? "del" : row.kind === "added" ? "ins" : "span";
      const sign = {removed: "−", added: "+", context: " "}[row.kind];
      line.append(el("span", "diff-num", String(row.line)), el("span", "diff-sign", sign), el(tag, "diff-text", row.text));
    }
    box.append(line);
  }
  return box;
}

// *asked* is the request that started the turn: a high-impact card shows it, so the
// person can compare what they asked for with what melo is about to do.
export function approvalCard({ doc, request, view, asked = "", states, openWorkspace = "", later = setTimeout }) {
  const { node: el, button } = maker(doc);
  const id = String(request.request_id);
  let state = states.get(id);
  if (!state) {
    state = {mountedAt: approvalNow(), deny: false, reason: "", showAll: false, submitting: false};
    states.set(id, state);
  }
  const card = el("section", "approval-card");
  card.id = `approval-card-${id}`;
  card.dataset.requestId = id;
  card.setAttribute("role", "region");
  card.setAttribute("aria-label", t("approval.needed"));
  const tool = String(request.tool_name || "tool");
  const args = request.arguments || {};
  const workspace = request.workspace_label || view.workspace || openWorkspace || t("approval.theWorkspace");
  const folder = workspace.replace(/\/+$/, "").split("/").pop() || workspace;
  const scope = t(`approval.scope.${view.scope === "task" ? "task" : "chat"}`);
  // The main chat's own question (a task's card keeps "this task"): what the person allows in the main
  // chat counts in every chat, the tasks they start and the terminal, so each answer that is kept says so.
  const everywhere = view.scope !== "task" && view.main === true;
  // A plain question, with the tool id in a caption and what "Allow for this chat" covers.
  let title = t("approval.useTool", { tool });
  let caption = tool;
  let content = JSON.stringify(args, null, 2);
  let rule = everywhere ? t("approval.rule.toolMain", { tool }) : t("approval.rule.tool", { tool, scope });
  let what = t("approval.arguments");
  let diff = null; // What a file write replaces, when the gateway could compare.
  let why = ""; // Why it could not, in words.
  let noBox = false; // A read outside the workspace shows its folder, not its arguments.
  if (tool === "shell_exec") {
    what = t("approval.command");
    title = t("approval.runCommand", { folder });
    caption = `${tool} · ${workspace}`;
    content = typeof args.command === "string" ? args.command : JSON.stringify(args.command ?? "");
  } else if (tool === "python_exec") {
    what = t("approval.pythonCode");
    title = t("approval.runPython", { folder });
    caption = `${tool} · ${workspace}`;
    content = String(args.code ?? "");
  } else if (tool === "file_write") {
    const target = request.target;
    const path = target?.path_label || String(args.path ?? t("approval.theFile"));
    content = String(args.content ?? "");
    what = t("approval.fileContent");
    rule = everywhere ? t("approval.rule.writeMain") : t("approval.rule.write", { scope });
    if (target?.diff?.note) why = target.diff.note;
    else diff = target?.diff ?? null;
    if (diff) what = t("approval.changesTo", { path });
    caption = !target ? t("approval.checkingFile", { tool }) : diff ? `${diffSummary(diff)} · ${tool}` : tool;
    title = !target ? t("approval.write", { path }) : !target.exists ? t("approval.create", { path })
      : diff ? t("approval.replace", { path }) : t("approval.replaceSize", { path, size: formatBytes(target.size_bytes) });
  } else if (tool.startsWith("browser:")) {
    // A site melo wants to open in Safari, or read with web_fetch: one question per site and chat,
    // and one answer covers both tools.
    const site = String(args.site || tool.slice(8));
    const reading = args.via === "web_fetch";
    what = t("approval.willDo");
    title = t(reading ? "approval.readSite" : "approval.openSite", { site });
    caption = t(reading ? "approval.readCaption" : "approval.browserCaption");
    rule = everywhere ? t(reading ? "approval.rule.readMain" : "approval.rule.openMain", { site })
      : t(reading ? "approval.rule.read" : "approval.rule.open", { site, scope });
    // Chrome in the background (`args.browser`) is worded for itself: no window comes to the front.
    content = t(reading ? "approval.readContent" : args.browser === "chrome" ? "approval.siteContentChrome" : "approval.siteContent", { site });
    if (reading && args.url) {
      // The whole address reaches the site, path and query too: the card shows it.
      why = content;
      content = String(args.url);
      what = t("approval.readAddress");
    }
  } else if (tool.startsWith("browser-act:") || tool.startsWith("browser-type:") || tool.startsWith("browser-submit:")) {
    // Acting on a page asks apart from reading it: clicks and choices are one question per site and
    // chat. Typing is its own question that shows all the text, and a form sent from the page is
    // another, with the address it goes to; both ask every time.
    const sending = tool.startsWith("browser-submit:");
    const typing = tool.startsWith("browser-type:");
    const site = String(args.site || tool.slice(tool.indexOf(":") + 1));
    what = t("approval.willDo");
    title = t(sending ? "approval.submitSite" : typing ? "approval.typeSite" : "approval.actSite", { site });
    caption = t(sending ? "approval.submitCaption" : typing ? "approval.typeCaption" : "approval.actCaption");
    rule = sending || typing ? t(sending ? "approval.rule.submitOnce" : "approval.rule.typeOnce")
      : everywhere ? t("approval.rule.actMain", { site }) : t("approval.rule.act", { site, scope });
    const chrome = args.browser === "chrome";
    content = t(sending ? "approval.submitContent"
      : typing ? (chrome ? "approval.typeContentChrome" : "approval.typeContent")
      : chrome ? "approval.actContentChrome" : "approval.actContent", { site });
    if (sending) {
      // Where the form goes (or that the page decides) and what it holds, nothing secret in it.
      why = content;
      const rows = Array.isArray(args.fields) ? args.fields.map(([name, value]) => `${name}: ${value}`) : [];
      content = [args.to ? String(args.to) : t("approval.submitPage"), ...(rows.length ? ["", t("approval.submitFields"), ...rows] : [])].join("\n");
      what = t("approval.submitAddress");
    } else if (typing) {
      // The whole text, uncut: it is what goes into the page.
      why = content;
      content = String(args.text ?? "");
      what = t("approval.actTyped");
    }
  } else if (tool.startsWith("handoff:")) {
    // A page asked to open an address of another app's kind, which the browser in the background does not
    // open itself: melo opens it on a yes, and a kept answer covers that kind of address where it is kept
    // (in the main chat, every chat). In Auto the step is high impact: no kept answer is offered.
    const app = String(args.app || tool.slice(tool.indexOf(":") + 1));
    const site = String(args.site || "");
    what = t("approval.handoffAddress");
    title = t("approval.handoffTitle", { app });
    caption = t("approval.handoffCaption");
    rule = everywhere ? t("approval.rule.handoffMain", { app }) : t("approval.rule.handoff", { app, scope });
    why = t("approval.handoffContent", { app, site });
    content = String(args.url || "");
  } else if (tool.startsWith("browser-dialog:")) {
    // A confirm box the page put up while melo acted: OK is asked every time; Cancel otherwise.
    const site = String(args.site || tool.slice(tool.indexOf(":") + 1));
    what = t("approval.dialogText");
    title = t("approval.dialogTitle", { site });
    caption = t("approval.dialogCaption");
    rule = t("approval.rule.dialogOnce");
    why = t("approval.dialogContent", { site });
    content = String(args.text ?? "");
  } else if (tool.startsWith("browser-point:")) {
    // A click (or a text) at a point melo cannot check: a secret field, a consent control, or a control inside a
    // box from another site that melo knows nothing of. Asked every time; the card says where it lands, in
    // the engine's words (English), and the same in Ask and in Auto.
    const site = String(args.site || tool.slice(tool.indexOf(":") + 1));
    what = t("approval.pointOn");
    title = t(String(args.how || "click").startsWith("type") ? "approval.pointTypeSite" : "approval.pointSite", { site });
    caption = t("approval.pointCaption");
    rule = t("approval.rule.pointOnce");
    why = t(args.browser === "chrome" ? "approval.pointContentChrome" : "approval.pointContent", { site });
    content = `${String(args.how || "click")} ${String(args.on || "a point of the page")}`;
  } else if (tool.startsWith("desktop-act:")) {
    // A button in another app's window that appeared after melo's own step in the browser: asked every
    // time, and the card names the app, the window and what is pressed.
    const app = String(args.app || tool.slice(tool.indexOf(":") + 1));
    what = t("approval.willDo");
    title = t("approval.desktopTitle", { app });
    caption = t("approval.desktopCaption");
    rule = t("approval.rule.desktopOnce");
    content = t("approval.desktopContent", { app, on: String(args.on || ""), window: String(args.window || "") });
    // A click at a point is asked once for both tries: if the background one does not reach the window, the
    // second moves the pointer and brings the app forward.
    if (args.delivery === "foreground_retry") why = t("approval.desktopForeground", { app });
  } else if (tool.startsWith("takeover:")) {
    // melo reached something only the person can do on a page (a sign-in, a code, a QR code, a human check): the
    // card offers them the page in the Browser panel. Their yes is "Take over"; nothing of it is kept for the chat.
    const site = String(args.site || tool.slice(tool.indexOf(":") + 1));
    const need = ["sign-in", "password", "code", "card", "qr", "captcha"].includes(args.need) ? args.need : "sign-in";
    what = t("approval.willDo");
    title = t(`approval.takeover.${need}`, { site });
    caption = t("approval.takeoverCaption");
    rule = t("approval.rule.takeover");
    why = t("approval.takeoverContent");
    content = "";
    noBox = true;
  } else if (tool.startsWith("window-read:")) {
    // A window another app put up right after melo's own step: nothing about it goes to the model unless
    // the person allows it here, in Ask and in Auto alike. The window's title is for the person only.
    const app = String(args.app || tool.slice(tool.indexOf(":") + 1));
    what = t("approval.seeWindow");
    title = t("approval.seeTitle", { app });
    caption = t("approval.seeCaption");
    rule = everywhere ? t("approval.rule.seeMain", { app })
      : t(view.scope === "task" ? "approval.rule.seeTask" : "approval.rule.see", { app });
    content = String(args.window || "");
  }
  // A high-impact call (installs, downloads, changes outside the folder) asks every time: the
  // card says why, offers no chat-wide allow, and starts on Deny. A site Auto stopped only because the
  // request did not ask for it (#465) says why too, but may be allowed for the chat: the server says so.
  const review = request.review;
  // A take-over, a window another app opened, a press in one and a click at a point melo cannot check are asked
  // of the person every time: Auto's fixed "opaque" review does not reword them or add reasons (its title is
  // about code, and its sentence is English), so the card keeps its own words in Ask and in Auto.
  const takeover = tool.startsWith("takeover:");
  const pointing = tool.startsWith("browser-point:");
  const personOnly = takeover || pointing || tool.startsWith("window-read:") || tool.startsWith("desktop-act:");
  const stopped = review?.verdict === "high" && !personOnly;
  const siteChoice = stopped && request.offers_chat === true;
  const high = stopped && !siteChoice;
  const reviewed = stopped ? reviewWords({ ...review, gist: request.gist }, folder) : { title: "", reasons: [] };
  if (high) {
    card.dataset.impact = "high";
    title = reviewed.title || title;
    caption = t("approval.asksEveryTime", { tool });
  }
  // A path outside the workspace (#405): the card names the real folder. A read may be allowed for
  // the chat or always; a write asks every time and is a high-impact card.
  const outside = outsideOf(request);
  const outsideFolder = outside ? String(outside.folder_label || outside.folder || "") : "";
  const grantable = outside?.access === "read" && outside.grantable === true;
  if (outside) {
    card.dataset.outside = outside.access;
    const asked = outside.access === "write" ? "write" : tool === "list_files" ? "list" : "read";
    title = t(`approval.outside.${asked}`, { folder: outsideFolder });
    caption = t(`approval.outside.caption.${outside.access === "write" ? "write" : "read"}`, { tool, workspace });
    if (outside.access === "read") { content = ""; noBox = true; }
  }
  // What the step will do comes first, in one plain sentence; a warning follows when the model's words
  // and the code differ. The sentence is the code's own whenever the rules found the step risky, and the
  // model's words then stand below it, labelled as the model's. A write, an edit and a tool's card already
  // say as much in their question.
  const gist = outside ? null : gistWords(request.gist);
  const said = request.gist?.say?.id;
  const plain = request.gist?.by === "code" && !gist?.warning && ["write", "edit", "tool"].includes(said);
  if (gist?.sentence && !plain) card.append(el("p", "approval-gist", gist.sentence));
  if (gist?.warning) {
    const warning = el("p", "approval-mismatch", gist.warning);
    warning.setAttribute("role", "alert");
    card.append(warning);
  }
  if (gist?.note) card.append(el("p", "approval-says", gist.note));
  card.append(el("p", "approval-title", title), el("p", "approval-caption", caption));
  if (outside) card.append(folderBox(doc, outside, tool, outsideFolder));
  if (stopped && reviewed.reasons.length) {
    card.append(el("p", "approval-why", t("approval.why")));
    const reasons = el("ul", "approval-reasons");
    for (const reason of reviewed.reasons) reasons.append(el("li", "", reason));
    card.append(reasons);
  }
  if (stopped && asked) {
    const text = asked.length > 200 ? `${asked.slice(0, 200).trimEnd()}…` : asked;
    card.append(el("p", "approval-asked", t("approval.youAsked", { text })));
  }
  if (why) card.append(el("p", "approval-note", why));
  const lines = diff ? diff.rows : noBox ? [] : content.split("\n");
  const canFold = tool === "python_exec" || tool === "file_write";
  const shown = !canFold || state.showAll ? lines : lines.slice(0, 20);
  // A long command scrolls inside its box: make the box reachable and named for a keyboard user.
  const body = diff ? diffBox(doc, shown) : el("pre", "approval-content", shown.join("\n"));
  body.tabIndex = 0;
  body.setAttribute("role", "group");
  body.setAttribute("aria-label", what);
  // The code of a routine command or script is folded: the sentence and the kind say what it is for. A step
  // the rules found risky (or never reviewed) shows its code open, until the person folds it.
  const fold = !outside && (tool === "shell_exec" || tool === "python_exec") ? el("details", "approval-code") : null;
  const codeHolder = fold ?? card;
  if (fold) {
    fold.append(el("summary", "", t("gist.showCode")));
    if (state.codeOpen ?? review?.verdict !== "routine") fold.open = true;
    fold.addEventListener("toggle", () => { state.codeOpen = fold.open === true; });
  }
  if (shown.length) codeHolder.append(body);
  if (canFold && !state.showAll && lines.length > 20) {
    const count = diff ? lines.filter((row) => row.kind !== "gap").length : lines.length;
    codeHolder.append(button("link-button", t("approval.showLines", { n: count }), () => {
      state.showAll = true;
      view.rerender();
    }));
  }
  if (fold && shown.length) card.append(fold);
  card.append(...explanationNodes(doc, request.explanation));
  const actions = el("div", "approval-actions");
  const denyView = el("div", "deny-view");
  // On a high-impact card the safe answer is the filled one.
  const once = button(high ? "outline-button" : "primary-button", takeover ? t("approval.takeoverYes") : t("common.allowOnce"), () => {
    view.answer(request, state, "once");
    // The panel opens and the keys go to the page as soon as the gateway hands it over.
    if (takeover) view.takeOver?.();
  });
  once.dataset.approvalOnce = "";
  once.dataset.approvalFocus = "once";
  const seeing = tool.startsWith("window-read:");
  const seeApp = String(args.app || "");
  const sessionWords = outside ? (everywhere ? t("approval.outside.everywhere") : t("approval.outside.chat", { scope }))
    : seeing ? (everywhere ? t("approval.seeEverywhere", { app: seeApp })
      : t(view.scope === "task" ? "approval.seeTask" : "approval.seeChat", { app: seeApp }))
      : everywhere ? t("approval.allowEverywhere") : t("approval.allowFor", { scope });
  const session = button("outline-button", sessionWords, () => view.answer(request, state, "session"));
  session.dataset.approvalFocus = "session";
  const always = button("outline-button", t("approval.outside.always"), () => view.answer(request, state, "always"));
  always.dataset.approvalFocus = "always";
  always.dataset.approvalAlways = "";
  const deny = button(high ? "primary-button" : "quiet-button", takeover ? t("approval.takeoverNo") : t("common.deny"), () => {
    if (takeover) {
      // "Not now" needs no reason: melo carries on without the page.
      view.answer(request, state, "deny");
      return;
    }
    state.deny = true;
    actions.hidden = true;
    grant.hidden = true;
    if (wide) wide.hidden = true;
    if (kept) kept.hidden = true;
    denyView.hidden = false;
    reason.focus();
  });
  deny.dataset.approvalDeny = "";
  deny.dataset.approvalFocus = "deny";
  // The answers offered: a folder outside the workspace that may be kept has four.
  // Typing and a form's send ask every time, like a high-impact call: their answers are never kept for the chat.
  const everyTime = high || takeover || pointing || tool.startsWith("browser-submit:") || tool.startsWith("browser-type:")
    || tool.startsWith("desktop-act:");
  const offered = everyTime ? [once, deny] : outside ? (grantable ? [once, session, always, deny] : [once, deny]) : [once, session, deny];
  actions.append(...offered);
  card.append(actions);
  const command = tool === "shell_exec" || tool === "python_exec";
  if (outside) {
    rule = outside.access === "write" ? t("approval.rule.outsideWrite", { workspace })
      : grantable ? (everywhere ? t("approval.rule.outsideReadMain", { folder: outsideFolder })
        : t("approval.rule.outsideRead", { scope, folder: outsideFolder })) : t("approval.rule.outsideOnce");
  } else if (high) rule = everywhere ? t("approval.rule.highMain", { workspace }) : t("approval.rule.high", { scope, workspace });
  else if (command) {
    const shell = tool === "shell_exec";
    rule = everywhere ? t(shell ? "approval.rule.shellMain" : "approval.rule.pythonMain", { workspace })
      : t(shell ? "approval.rule.shell" : "approval.rule.python", { scope, workspace });
  }
  const grant = el("p", "approval-warning", rule);
  grant.id = `approval-scope-${id}`;
  if (offered.includes(session)) session.setAttribute("aria-describedby", grant.id);
  if (offered.includes(always)) always.setAttribute("aria-describedby", grant.id);
  card.append(grant);
  // In the main chat a kept answer counts everywhere: one line says so, where, and how to take it back.
  const wide = everywhere && offered.includes(session)
    ? el("p", "approval-warning approval-everywhere", t("approval.everywhere"))
    : null;
  if (wide) {
    wide.id = `approval-wide-${id}`;
    session.setAttribute("aria-describedby", `${grant.id} ${wide.id}`);
    card.append(wide);
  }
  // The server says, by code, when the person's Allow is kept for the chat's later tasks (#369).
  const rememberedKey = view.scope === "task" ? "task" : everywhere ? "everywhere" : "chat";
  const kept = !high && request.remembered === true
    ? el("p", "approval-remembered", t(`approval.remembered.${rememberedKey}`))
    : null;
  if (kept) card.append(kept);
  if (state.submitting) card.append(el("p", "approval-sending", t("approval.sending")));
  const note = t("approval.instead");
  const label = el("label", "", note);
  const reason = el("input", "deny-reason");
  reason.value = state.reason;
  reason.dataset.approvalFocus = "reason";
  reason.addEventListener("compositionstart", () => { state.composing = true; });
  reason.addEventListener("compositionend", () => {
    state.composing = false;
    state.reason = reason.value.slice(0, 500);
    view.rerender();
  });
  reason.maxLength = 500;
  reason.setAttribute("aria-label", note);
  const counter = el("span", "deny-counter", `${state.reason.length} / 500`);
  reason.addEventListener("input", () => {
    state.reason = reason.value.slice(0, 500);
    counter.textContent = `${state.reason.length} / 500`;
  });
  const confirm = button("outline-button", t("common.deny"), () => view.answer(request, state, "deny"));
  confirm.dataset.approvalConfirm = "";
  const back = button("quiet-button", t("approval.back"), () => {
    state.deny = false;
    denyView.hidden = true;
    actions.hidden = false;
    grant.hidden = false;
    if (wide) wide.hidden = false;
    if (kept) kept.hidden = false;
    deny.focus();
  });
  reason.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      view.answer(request, state, "deny");
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      back.click();
    }
  });
  label.append(reason);
  denyView.append(label, counter, confirm, back);
  denyView.hidden = !state.deny;
  actions.hidden = state.deny;
  grant.hidden = state.deny;
  if (wide) wide.hidden = state.deny;
  if (kept) kept.hidden = state.deny;
  card.append(denyView);
  if (request.arguments_complete !== true) {
    card.append(el("p", "approval-warning", request.arguments_complete === false
      ? t("approval.truncated") : t("approval.incomplete")));
  }
  if (request.arguments_complete !== true || (tool === "file_write" && !request.target)) {
    card.append(button("link-button", t("approval.reload"), () => view.reload(id)));
  }
  const controls = [...offered, confirm];
  // A button is grey for one of three reasons, and each has words on the card: the guard (half a second),
  // an answer under way ("Sending your answer…") and details that are missing (the notes above).
  const enable = () => {
    const left = guardLeft(state);
    const unavailable = request.arguments_complete !== true || (tool === "file_write" && !request.target);
    for (const control of controls) control.disabled = left > 0 || state.submitting || (unavailable && control !== deny && control !== confirm);
    // A timer that fires before the guard has run out sets itself again: nothing else would.
    if (left > 0) later(enable, left);
  };
  enable();
  return card;
}

/** The request's path outside the workspace (#405), or null. */
function outsideOf(request) {
  const outside = request?.outside;
  return outside && typeof outside === "object" && (outside.access === "read" || outside.access === "write") ? outside : null;
}

/** The folder a call outside the workspace opens: its real path, and what melo wants to do there. */
function folderBox(doc, outside, tool, folderLabel) {
  const { node: el } = maker(doc);
  const box = el("div", "approval-folder");
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", t("approval.outside.folder"));
  const path = String(outside.path || "");
  const name = path.replace(/\/+$/, "").split("/").pop() || path;
  const what = outside.access === "write" ? t("approval.outside.writes", { name })
    : tool === "list_files" ? t("approval.outside.lists") : t("approval.outside.reads", { name });
  const where = el("p", "approval-folder-path", folderLabel);
  where.title = String(outside.folder || folderLabel);
  box.append(el("p", "approval-folder-label", t("approval.outside.folder")), where, el("p", "approval-folder-what", what));
  return box;
}

/**
 * melo's explanation of a task's step (#369), under the step and the review's reasons: words for the
 * person to read, labelled as melo's. The step, the buttons and what an answer approves come from code.
 */
function explanationNodes(doc, explanation) {
  const { node: el } = maker(doc);
  if (explanation?.state === "pending") return [el("p", "approval-explaining", t("approval.explaining"))];
  const text = explanation?.state === "ready" ? String(explanation.text ?? "").trim() : "";
  if (!text) return [];
  const box = el("section", "approval-explain");
  box.setAttribute("aria-label", t("approval.explanation"));
  box.append(
    el("p", "approval-explain-label", t("approval.explanation")),
    el("p", "approval-explain-text", text),
    el("p", "approval-explain-note", t("approval.explanationNote")),
  );
  return [box];
}
