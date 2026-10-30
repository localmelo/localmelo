/**
 * The folder a new chat works in: a chip in the composer and the panel it opens.
 *
 * The panel offers the default folder, the recent ones (the folders of the listed chats,
 * newest first), "Choose a folder…" and, in the macOS app, "Show in Finder". The app answers
 * through a message handler (`localmeloFolder`): `{action: "choose"}` resolves a path picked in
 * the system's panel, or null; `{action: "reveal", path}` resolves whether the folder was shown.
 * A browser has no handler, so "Choose a folder…" opens a path field. The page only keeps the
 * pick (`pick(path)`); the chat's first message sends it as `workspace`. `null` is the default.
 */
const bare = (path) => String(path ?? "").replace(/\/+$/, "");
const labelOf = (path, sessions) => sessions.find((item) => item.workspace === path)?.workspace_label || path;

/** The last part of a folder's path: `localmelo-workspace` for `~/localmelo-workspace`. */
export function folderName(path) {
  const text = bare(path);
  return text.slice(text.lastIndexOf("/") + 1) || String(path ?? "");
}

/**
 * The folders to offer: the default, the current pick when no recent folder is it, then up to
 * `limit` recent ones. Each is `{path, label, kind, current}`; the default's `path` is null.
 */
export function folderChoices({sessions, fallback, current, limit = 4}) {
  const choices = [{path: null, label: fallback, kind: "default"}];
  const seen = new Set([bare(fallback)]);
  const recent = [];
  for (const item of [...sessions].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))) {
    const label = item.workspace_label || item.workspace;
    if (!item.workspace || seen.has(bare(item.workspace)) || seen.has(bare(label))) continue;
    seen.add(bare(item.workspace)).add(bare(label));
    recent.push({path: item.workspace, label, kind: "recent"});
    if (recent.length === limit) break;
  }
  if (current !== null && !recent.some((choice) => choice.path === current)) {
    choices.push({path: current, label: labelOf(current, sessions), kind: "pick"});
  }
  return [...choices, ...recent].map((choice) => ({...choice, current: choice.path === current}));
}

/**
 * Wire the chip and its panel (the ids are in index.html). `state()` gives `{sessions, fallback,
 * current}`; `pick(path)` takes the choice; `notify(text)` says what could not be done; `bridge()` is
 * the app's message handler, or nothing in a browser. `update(visible)` redraws the chip and is cheap.
 */
export function setupFolderPicker({doc, state, pick, notify, bridge}) {
  const $ = (id) => doc.getElementById(id);
  const put = (id, text) => { if ($(id).textContent !== text) $(id).textContent = text; };
  const hint = $("folder-path-hint");
  const plain = hint.textContent;
  let open = false;

  function node(tag, className, text) {
    const made = doc.createElement(tag);
    made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  }

  function entry(choice) {
    const button = node("button", "folder-item");
    button.type = "button";
    if (choice.kind === "recent") button.dataset.folderRecent = choice.path;
    if (choice.current) button.setAttribute("aria-current", "true");
    const text = node("span", "folder-item-text");
    text.append(node("span", "folder-item-name", folderName(choice.label)), node("span", "folder-item-where", choice.label));
    const mark = node("span", "folder-mark", choice.current ? "✓" : "");
    mark.setAttribute("aria-hidden", "true");
    button.append(mark, text);
    if (choice.kind === "default") button.append(node("span", "folder-tag", "Default"));
    button.addEventListener("click", () => select(choice.path));
    return button;
  }

  function select(path) {
    pick(path !== null && bare(path) === bare(state().fallback) ? null : path);
    toggle(false);
    $("folder-btn").focus();
  }

  function toggle(next) {
    if (next === open) return;
    open = next;
    $("folder-menu").hidden = !next;
    $("folder-btn").setAttribute("aria-expanded", String(next));
    if (next) {
      $("folder-list").replaceChildren(...folderChoices(state()).map(entry));
      $("folder-finder").hidden = !bridge();
      return;
    }
    $("folder-path-row").hidden = true;
    $("folder-path").value = "";
    $("folder-path").removeAttribute("aria-invalid");
    hint.textContent = plain;
  }

  /** The app's panel when there is one; else the path field. */
  async function browse() {
    const handler = bridge();
    if (!handler) {
      $("folder-path-row").hidden = false;
      $("folder-path").focus();
      return;
    }
    const picked = await handler.postMessage({action: "choose"}).catch(() => null);
    if (typeof picked === "string" && picked.startsWith("/")) select(picked);
    else $("folder-btn").focus();
  }

  function usePath() {
    const text = $("folder-path").value.trim();
    if (/^(\/|~\/)\S/.test(text)) select(text);
    else {
      $("folder-path").setAttribute("aria-invalid", "true");
      hint.textContent = `That is not a full path. ${plain}`;
    }
  }

  async function reveal() {
    const {fallback, current} = state();
    const shown = await bridge()?.postMessage({action: "reveal", path: current ?? fallback}).catch(() => false);
    if (shown === true) toggle(false);
    else notify("melo could not open this folder in Finder. It may not exist yet.");
  }

  $("folder-btn").addEventListener("click", () => toggle(!open));
  $("folder-choose").addEventListener("click", browse);
  $("folder-use").addEventListener("click", usePath);
  $("folder-finder").addEventListener("click", reveal);
  $("folder-path").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    // The panel sits inside the composer's form, where Enter would send the message.
    event.preventDefault();
    usePath();
  });

  return {
    update(visible) {
      const {sessions, fallback, current} = state();
      const label = current === null ? fallback : labelOf(current, sessions);
      $("folder-picker").hidden = !visible || !label;
      if ($("folder-picker").hidden) toggle(false);
      put("folder-text", `In ${folderName(label)}`);
      put("folder-full", label);
    },
    isOpen: () => open,
    close: () => toggle(false),
  };
}
