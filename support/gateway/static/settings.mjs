import { folderChoices, folderName } from "./folder.mjs";
import { formatTime, t } from "./i18n.mjs";
import { setupPermissionsTab } from "./permissions-tab.mjs";
import { whenWords } from "./scheduled-tab.mjs";

// A message that tells the user to run a melo command ("... (run 'melo setup')") is advice for a
// terminal. This dialog is where the user fixes it, so the advice is left out.
const terminalAdvice = (message) => /\brun 'melo /i.test(message);

// The dialog's words for a model server: plain ones by default (`settings.plain.*`), and the technical ones a
// developer reads today (`settings.developer.*`: backend, API URL, adapter). `layout()` writes the set of the mode.
const WORD_NAMES = ["description", "adapter", "url", "urlHint", "discover", "needUrl"];
const wordsOf = (set) => Object.fromEntries(WORD_NAMES.map((name) => [name, t(`settings.${set}.${name}`)]));

/** The sections, in the order of the list; each is named `settings.section.<id>`. Developer is listed in developer mode only. */
export const SECTIONS = ["model", "memory", "chats", "permissions", "scheduled", "appearance", "developer"];
const THEMES = ["system", "light", "dark"];
const RECALL = ["auto", "bm25", "embedding", "llm"];
// How memory finds notes, in the words of the Memory section and the phone's list.
const recallWords = (name) => t(`settings.recall.${RECALL.includes(name) ? name : "auto"}`);
// What the app shows on the desktop (the words are the app's menu's: `app.desktop.<place>`).
const DESKTOP = ["pill", "pet", "nothing"];
// The words under "Other apps' windows" for each state of the desktop control helper (`settings.appWindows.status.<key>`).
const HELPER_LINES = {installed: "installed", granted: "granted", no_grant: "noGrant", failed: "failed"};
// The console's language: automatic, or one the console has. Each language is named in itself.
const LANGUAGE_SETTINGS = ["auto", "en", "zh-Hans"];
// How a new chat starts (#339): `[approvals] mode`.
const APPROVAL_MODES = ["ask", "auto"];
// Where melo browses: `[browser] driver` (Chrome in the background, Safari's windows, or off).
const BROWSER_DRIVERS = ["chrome", "safari-automation", "off"];
const BROWSER_WORDS = {"chrome": "chrome", "safari-automation": "safari", "off": "off"};
// `melo -h all` lists the settings by section; `settings.group.<section>` are the names people read.
const GROUPS = new Set(["agent", "retrieval", "memory", "endpoint", "sessions", "cli", "ui", "scheduler", "browser", "tools", "gateway", "console", "workspace"]);
const groupName = (section) => (section === "" ? t("settings.group.general") : GROUPS.has(section) ? t(`settings.group.${section}`) : section);
const LOCAL_FIELDS = ["chat_url", "chat_model"];
const CLOUD_FIELDS = ["api_key_env", "api_key", "base_url", "chat_model", "thinking"];
// What the keychain reports about a service's saved key, by what a settings section calls it.
const KEY_STATES = {keychain: "saved", missing: "missing", locked: "locked", unavailable: "unavailable"};
// The words for a key that cannot be used, by what the keychain reported.
const KEY_PROBLEMS = {missing: "settings.hosted.keyMissing", locked: "settings.hosted.keyLocked", unavailable: "settings.hosted.keyUnavailable"};
// How long "Remove" waits for its second click.
const REMOVE_WAIT_MS = 5000;

/** Whether a hosted service's model thinks: the person's choice, else the model's default in its preset. */
export function thinks(preset, data) {
  if (data.thinking === "on" || data.thinking === "off") return data.thinking === "on";
  return !(preset.thinking_off_models || []).includes(data.chat_model);
}

/** "checked just now", "checked 12 s ago", "checked 3 min ago" or "checked at 9:05 AM". */
export function checkedWords(secondsAgo, now = Date.now()) {
  const ago = Math.max(0, Math.round(Number(secondsAgo) || 0));
  if (ago < 5) return t("settings.checked.justNow");
  if (ago < 60) return t("settings.checked.seconds", {n: ago});
  if (ago < 3600) return t("settings.checked.minutes", {n: Math.floor(ago / 60)});
  return t("settings.checked.at", {time: formatTime(new Date(now - ago * 1000), "clock")});
}

/** A key's last part in words: `agent.window_messages` reads "Window messages". */
export function keyWords(key) {
  const last = String(key).slice(String(key).lastIndexOf(".") + 1).replace(/_s$/, " (s)").replace(/_/g, " ")
    .replace(/\b(mcp|url|llm|gpu)\b/g, (word) => word.toUpperCase());
  return last.charAt(0).toUpperCase() + last.slice(1);
}

/**
 * Wire the Settings dialog: a list of sections, the chosen one beside it (a sheet on a phone). Each change saves
 * at once with the revision it read (a 409 asks for a reload) and offers Undo; the connection is the one change
 * made in a form, saved with "Save connection".
 *
 * The caller owns the entry buttons and what a save refreshes (`onSaved`). `developer()` is the page's mode until
 * the settings load; `health()` the last `GET /api/health`; `sessions()` the listed chats, for the recent folders;
 * `openTab(id)` closes the dialog and opens the panel's Memory or Scheduled tab; `desktop()` and `folders()` are the
 * macOS app's message handlers (none in a browser); `narrow()` says whether the dialog is the phone's sheet;
 * `reloadPage()` loads the page again once a new language is saved, so every screen reads in it. The Permissions
 * section (`permissions-tab.mjs`) lists what the person allowed and revokes it; it reads the list each time it is chosen.
 * Returns `open(section)`.
 */
export function setupSettings({
  api, onSaved, developer = () => false, health = () => null, sessions = () => [], openTab = () => {},
  desktop = () => null, folders = () => null, narrow = () => globalThis.matchMedia?.("(max-width: 700px)")?.matches === true,
  clock = Date.now, reloadPage = () => globalThis.location?.reload(),
}) {
  const $ = (id) => document.getElementById(id);
  const dialog = $("settings-dialog");
  const chat = $("s-adapter");
  const embedding = $("s-embed-backend");
  const save = $("create-btn");
  const discover = $("discover-btn");
  const error = $("create-error");
  const reload = $("settings-reload");
  const status = $("settings-status");
  const modelStatus = $("discover-status");
  const keyStatus = $("s-key-status");
  // The line under the key: `quiet` text is announced to a screen reader and not drawn.
  const keySays = (text, quiet = false) => { keyStatus.className = quiet ? "field-hint sr-only" : "field-hint"; keyStatus.textContent = text; };
  const envStatus = $("s-key-env-status");
  const testButton = $("settings-test");
  const testStatus = $("settings-test-status");

  // What the person allowed (`GET /api/grants`): the section draws and revokes by itself and tells the list when its count changes.
  const permissions = setupPermissionsTab({ doc: document, api, clock, onChange: () => notes() });
  $("s-permissions").append(permissions.root);

  let theme = "system";
  try {
    const stored = localStorage.getItem("localmelo-console.theme");
    if (THEMES.includes(stored)) theme = stored;
  } catch { /* Storage may be unavailable. */ }
  function applyTheme(value) {
    theme = value;
    document.documentElement.dataset.theme = value;
    for (const name of THEMES) $(`settings-theme-${name}`).checked = name === value;
  }
  function storeTheme(value) {
    applyTheme(value);
    try { localStorage.setItem("localmelo-console.theme", value); } catch { /* Keep the current theme. */ }
    notes();
  }
  applyTheme(theme);

  const inputIds = ["s-adapter", "s-chat-url", "s-key-env", "s-key-input", "s-key-save", "s-key-replace", "s-key-remove", "s-base-url",
    "s-chat-model", "s-thinking", "s-embed-backend", "s-embed-url", "s-embed-model", "s-embed-use"];
  // Each saves on its own; they wait while a change is being saved.
  const switchIds = ["s-across-chats", "s-web-fetch", "s-web-search", "s-app-windows", "s-follow-up", "s-reflection", "s-developer", ...RECALL.map((name) => `s-recall-${name}`),
    ...LANGUAGE_SETTINGS.map((name) => `s-language-${name}`), ...APPROVAL_MODES.map((name) => `s-approvals-${name}`),
    ...BROWSER_DRIVERS.map((name) => `s-browser-${name}`)];
  let settings = null;
  let sections = {};
  let generation = 0;
  let discoveryVersion = 0;
  let keyVersion = 0;
  let keychainVersion = 0;
  let testVersion = 0;
  let busy = false;
  let saving = false;
  let words = wordsOf("plain");
  let isDeveloper = false;
  let current = "model";
  // The embedding model the last "Discover models" listed on the chat server; "" before one did.
  let offered = "";
  // The hosted services (`choices.hosted`) and what the keychain holds for each, as last asked or changed.
  let presets = [];
  const keychain = new Map();
  // Whether the saved key is being replaced, and whether "Remove" waits for its second click.
  let replacing = false;
  let removeArmed = null;
  // The last test of this dialog: {at, ok, failed} (`failed` is "chat" or "embedding"), or null.
  let tested = null;
  let checkin = null;
  let helper = null; // the desktop control helper's status from the health route; null until it is read
  let helperAsked = false;
  let place = null; // what is on the desktop, as the app says; null without the app
  let undo = null;

  // --- sections ---------------------------------------------------------------------------------------------

  const listed = () => SECTIONS.filter((id) => id !== "developer" || isDeveloper);
  function view(name) {
    $("settings-frame").dataset.view = name;
  }
  function select(name, {focus = false, show = true} = {}) {
    if (!listed().includes(name)) name = "model";
    current = name;
    for (const id of SECTIONS) {
      const tab = $(`settings-tab-${id}`);
      tab.setAttribute("aria-selected", String(id === name));
      tab.tabIndex = id === name ? 0 : -1;
      $(`settings-panel-${id}`).hidden = id !== name;
    }
    if (show) view("section");
    if (focus) $(`settings-tab-${name}`).focus?.();
    if (name === "scheduled") loadCheckin();
    if (name === "permissions") permissions.show();
  }
  for (const id of SECTIONS) {
    const tab = $(`settings-tab-${id}`);
    tab.addEventListener("click", () => select(id));
    tab.addEventListener("keydown", (event) => {
      const order = listed();
      const at = order.indexOf(id);
      const next = {ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: order.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      // Arrows move along the list; on a phone they move the focus without opening the section.
      select(order[(next + order.length) % order.length], {focus: true, show: !narrow()});
    });
  }
  $("settings-back").addEventListener("click", () => {
    view("list");
    $(`settings-tab-${current}`).focus?.();
  });

  function layout(developerMode) {
    isDeveloper = developerMode;
    words = wordsOf(developerMode ? "developer" : "plain");
    $("settings-description").textContent = words.description;
    $("s-adapter-label").textContent = words.adapter;
    $("s-chat-url-label").textContent = words.url;
    $("s-chat-url-hint").textContent = words.urlHint;
    dialog.dataset.layout = developerMode ? "developer" : "default";
    $("settings-tab-developer").hidden = !developerMode;
    $("s-recall-key").hidden = !developerMode;
    if (!listed().includes(current)) select("model", {show: false});
    recall();
    followUp();
  }

  // Following up on the tasks a chat started (`[tasks] follow_up`, #411): a developer setting, so its
  // switch shows in developer mode only, and not at all for a gateway that does not report it.
  function followUp() {
    $("s-follow-up-row").hidden = !isDeveloper || !settings || settings.follow_up === undefined;
    $("s-follow-up").checked = settings?.follow_up !== false;
  }

  // --- the connection form (as before: a draft per server, saved together) -----------------------------------

  function invalidateTest() {
    testVersion += 1;
    tested = null;
    testStatus.textContent = "";
    testStatus.dataset.state = "";
    testButton.disabled = busy || !settings || !chat.value;
    model();
  }

  function showError(message, conflict = false) {
    error.textContent = message;
    error.hidden = !message;
    reload.hidden = !conflict;
    foot();
  }
  // The foot under the section shows only while it has something to say.
  function foot() {
    $("settings-foot").hidden = !$("settings-saved-text").textContent && error.hidden && reload.hidden;
  }

  function enable(ready) {
    const chosen = !!chat.value; // Nothing is saved until the user chooses a server.
    for (const id of inputIds) $(id).disabled = !ready;
    for (const id of ["s-chat-url", "s-chat-model"]) $(id).disabled = !ready || !chosen;
    for (const id of switchIds) $(id).disabled = !ready || saving;
    $("s-checkin").disabled = !ready || saving || !checkin;
    save.disabled = !ready || !chosen;
    discover.disabled = !ready || !chosen || isCloud();
    testButton.disabled = !ready || !chosen;
  }

  // A hosted service is a preset; the select's value is a preset's id for it, a backend's key for a local server.
  const preset = (id = chat.value) => presets.find((item) => item.id === id);
  const isCloud = (id = chat.value) => preset(id) !== undefined;

  // The preset the saved connection is on: a hosted backend's section names it, an older one is its backend's own.
  function savedPreset(loaded = settings) {
    const backend = loaded?.chat_backend;
    if (!backend) return undefined;
    return presets.find((item) => item.id === loaded.sections[backend]?.preset) ?? presets.find((item) => item.id === backend);
  }

  function option(choice) {
    const node = document.createElement("option");
    node.value = choice.key;
    node.textContent = choice.label;
    return node;
  }

  /** Fill a select with `choices`, or with named `groups` of choices; `blank` adds an empty first entry. */
  function options(node, choices, selected, blank = "", groups = null) {
    const all = blank ? [{key: "", label: blank}, ...choices] : choices;
    node.replaceChildren();
    for (const choice of all) node.append(option(choice));
    for (const group of groups || []) {
      const holder = document.createElement("optgroup");
      holder.label = group.label;
      for (const choice of group.choices) holder.append(option(choice));
      node.append(holder);
    }
    const every = [...all, ...(groups || []).flatMap((group) => group.choices)];
    node.value = every.some((choice) => choice.key === selected) ? selected : every[0]?.key || "";
  }

  function section(key) {
    return sections[key] || (sections[key] = {});
  }

  function drafts(loaded) {
    sections = {};
    presets = loaded.choices.hosted || [];
    keychain.clear();
    replacing = false;
    removeArmed = null;
    // Only writable fields go back to the server; never copy api_key_set.
    const local = loaded.choices.chat_backends.filter((choice) => choice.kind !== "cloud");
    for (const choice of local) {
      const fields = ["chat_url", "chat_model", "embedding_url", "embedding_model"];
      const saved = loaded.sections[choice.key] || {};
      const defaults = loaded.defaults[choice.key] || {};
      sections[choice.key] = Object.fromEntries(fields.map((field) => [field, saved[field] || defaults[field] || ""]));
    }
    const on = savedPreset(loaded);
    for (const item of presets) {
      const mine = sectionOf(loaded, item);
      const defaults = loaded.defaults[item.backend] || {};
      // A service that shares its section with another (DeepSeek and OpenAI) starts from its own defaults.
      sections[item.id] = item === on
        ? {api_key_env: mine.api_key_env || "", api_key: mine.api_key || "", base_url: mine.base_url || item.stored_url,
          chat_model: mine.chat_model || item.models[0] || "", thinking: mine.thinking || ""}
        : {api_key_env: "", api_key: "", base_url: item.stored_url, chat_model: item.models[0] || defaults.chat_model || "", thinking: ""};
    }
    if (on && KEY_STATES[sectionOf(loaded, on).key]) keychain.set(on.id, KEY_STATES[sectionOf(loaded, on).key]);
    options(chat, [], on?.id ?? loaded.chat_backend, loaded.chat_backend ? "" : t("settings.chooseServer"), [
      {label: t("settings.choose.local"), choices: local},
      {label: t("settings.choose.hosted"), choices: presets.map((item) => ({key: item.id, label: item.label}))},
    ]);
    options(embedding, loaded.choices.embedding_backends, loaded.embedding_backend || "none");
    offered = "";
  }

  // The saved section of the preset's backend.
  const sectionOf = (loaded, item) => loaded.sections[item.backend] || {};

  function fillModels(...lists) {
    const names = [...new Set(lists.flat().filter(Boolean))];
    $("s-model-options").replaceChildren(...names.map((name) => Object.assign(document.createElement("option"), {value: name})));
  }

  function renderChat() {
    discoveryVersion += 1;
    const item = preset();
    const cloud = item !== undefined;
    const data = section(chat.value);
    $("s-chat-url-field").hidden = cloud;
    $("s-cloud-fields").hidden = !cloud;
    $("s-chat-url").value = data.chat_url || "";
    $("s-chat-model").value = data.chat_model || "";
    $("s-key-env").value = data.api_key_env || "";
    $("s-base-url").value = data.base_url || "";
    fillModels(cloud ? item.models : []);
    modelStatus.textContent = cloud ? t(item.models.length ? "settings.hosted.modelHint" : "settings.typeModel") : words.discover;
    discover.hidden = cloud;
    discover.disabled = busy || cloud || !chat.value;
    replacing = false;
    removeArmed = null;
    if (cloud) renderHosted(item);
    else { keyVersion += 1; envStatus.textContent = ""; keySays(""); }
  }

  // The part of the form that is the hosted service's: its address, its key, and its thinking switch.
  function renderHosted(item) {
    const data = section(item.id);
    $("s-base-url-field").hidden = !item.free_address;
    $("s-address-text").hidden = item.free_address;
    $("s-address-text").textContent = t("settings.hosted.addressIs", {address: data.base_url || item.base_url});
    $("s-key-env-details").open = !!data.api_key_env;
    $("s-thinking-field").hidden = !item.thinking_switch;
    $("s-thinking").checked = thinks(item, data);
    keySays("");
    renderKey();
    if (!keychain.has(item.id)) askKeychain(item);
    checkKey();
  }

  // The key row: "Saved in the keychain" with Replace and Remove, or a field to paste it in.
  function renderKey() {
    const item = preset();
    if (!item) return;
    const data = section(item.id);
    const have = keychain.get(item.id) === "saved" && !!data.api_key;
    $("s-key-saved").hidden = !have || replacing;
    $("s-key-entry").hidden = have && !replacing;
    $("s-key-help").hidden = have && !replacing;
    $("s-key-remove").textContent = t(removeArmed ? "settings.hosted.removeSure" : "settings.hosted.remove");
    const problem = KEY_PROBLEMS[keychain.get(item.id)];
    if (have) state("s-key-state", "ok", t("settings.hosted.keySavedState"));
    else if (problem && data.api_key) state("s-key-state", "error", t(problem));
    $("s-key-input").disabled = busy || !settings;
    $("s-key-save").disabled = busy || !settings;
  }

  // Whether the keychain holds a key for the service; no key is read.
  async function askKeychain(item) {
    const version = ++keychainVersion;
    const currentGeneration = generation;
    try {
      const result = await api(`/settings/key-status?${new URLSearchParams({preset: item.id})}`);
      if (version !== keychainVersion || currentGeneration !== generation || !dialog.open) return;
      keychain.set(item.id, result.keychain || "unknown");
      // A key saved earlier is the service's key from now on.
      const data = section(item.id);
      if (result.keychain === "saved" && !data.api_key) data.api_key = item.key_ref;
    } catch {
      if (version !== keychainVersion || currentGeneration !== generation || !dialog.open) return;
      keychain.set(item.id, "unknown");
    }
    renderKey();
  }

  function renderEmbedding() {
    const enabled = embedding.value !== "none" && !!embedding.value;
    $("s-embed-fields").hidden = !enabled;
    const data = enabled ? section(embedding.value) : {};
    $("s-embed-url").value = data.embedding_url || "";
    $("s-embed-model").value = data.embedding_model || "";
  }

  // "Remember across chats with <model>?": offered when the server lists an embedding model (or one is in use already).
  function renderOffer() {
    const inUse = embedding.value !== "none" && !!embedding.value && !!$("s-embed-model").value;
    if (inUse) offered = $("s-embed-model").value; // unchecking keeps the offer, so it can be checked again
    const name = offered;
    $("s-embed-offer").hidden = !name;
    $("s-embed-use").checked = inUse;
    $("s-embed-offer-label").textContent = t("settings.offer", {model: name});
    $("s-memory-hint").textContent = t(inUse ? "settings.memoryHint.meaning" : offered ? "settings.memoryHint.words" : "settings.memoryHint.wordsOffer");
  }

  function editor(open) {
    $("s-editor").hidden = !open;
    $("s-change").setAttribute("aria-expanded", String(open));
    $("s-change").textContent = t(open ? "common.close" : "settings.change");
  }

  // The environment variable is optional: it is checked when a name is typed, and says it wins when it is set.
  async function checkKey() {
    const version = ++keyVersion;
    const currentGeneration = generation;
    const name = $("s-key-env").value;
    if (!name) {
      envStatus.textContent = "";
      return;
    }
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name)) {
      envStatus.textContent = t("settings.key.invalid");
      return;
    }
    envStatus.textContent = t("settings.key.checking");
    try {
      const result = await api(`/settings/key-status?${new URLSearchParams({name})}`);
      if (version === keyVersion && currentGeneration === generation && dialog.open) {
        envStatus.textContent = t(result.set ? "settings.key.set" : "settings.key.notSet");
      }
    } catch {
      if (version === keyVersion && currentGeneration === generation && dialog.open) {
        envStatus.textContent = t("settings.key.failed");
      }
    }
  }

  function candidate() {
    const chatData = section(chat.value);
    const item = preset();
    const selectedSections = {[item ? item.backend : chat.value]: item
      ? Object.fromEntries(CLOUD_FIELDS.map((field) => [field, String(chatData[field] ?? "")]))
      : {chat_url: chatData.chat_url, chat_model: chatData.chat_model}};
    if (embedding.value !== "none") {
      const data = section(embedding.value);
      selectedSections[embedding.value] = {
        ...selectedSections[embedding.value],
        embedding_url: data.embedding_url,
        embedding_model: data.embedding_model,
      };
    }
    return {chat_backend: item ? item.backend : chat.value, embedding_backend: embedding.value, workspace: settings.workspace, sections: selectedSections};
  }

  // Whether the form holds the connection that is saved now, field for field.
  function draftIsSaved() {
    if (!settings || !chat.value) return false;
    const {chat_backend: backend, embedding_backend: embed, sections: fields} = candidate();
    const now = savedConnection(settings);
    return JSON.stringify([backend, embed, fields]) === JSON.stringify([now.chat_backend, now.embedding_backend, now.sections]);
  }

  // The connection as `loaded` saved it, for Undo: its writable fields only.
  function savedConnection(loaded) {
    const backend = loaded.chat_backend;
    const embed = loaded.embedding_backend || "none";
    const fields = savedPreset(loaded) ? CLOUD_FIELDS : LOCAL_FIELDS;
    const pick = (data, names) => Object.fromEntries(names.map((name) => [name, String(data?.[name] ?? "")]));
    const out = {[backend]: pick(loaded.sections[backend], fields)};
    if (embed !== "none") out[embed] = {...out[embed], ...pick(loaded.sections[embed], ["embedding_url", "embedding_model"])};
    return {chat_backend: backend, embedding_backend: embed, sections: out};
  }

  // --- what each section shows ------------------------------------------------------------------------------

  function state(id, kind, word, ...rest) {
    const node = $(id);
    node.dataset.state = kind;
    const parts = [Object.assign(document.createElement("span"), {className: "st-dot"}),
      Object.assign(document.createElement("span"), {className: "st-word", textContent: word})];
    for (const item of rest.filter(Boolean)) {
      const dot = Object.assign(document.createElement("span"), {textContent: "·"});
      dot.setAttribute("aria-hidden", "true");
      parts.push(dot, Object.assign(document.createElement("span"), {className: item.mono ? "st-mono" : "", textContent: item.text ?? item}));
    }
    node.replaceChildren(...parts);
    node.dataset.text = [word, ...rest.filter(Boolean).map((item) => item.text ?? item)].join(" · ");
  }

  const ago = (at) => checkedWords((clock() - at) / 1000, clock());

  function model() {
    if (!settings) return;
    const backend = settings.chat_backend;
    const saved = settings.sections[backend] || {};
    const name = saved.chat_model || "";
    const probe = health()?.model_server;
    const said = (key) => t(`settings.state.${key}`);
    if (!backend) state("s-chat-state", "off", said("notSetUp"), said("chooseBelow"));
    else if (tested?.ok || (tested && tested.failed === "embedding")) state("s-chat-state", "ok", said("connected"), {text: name, mono: true}, ago(tested.at));
    else if (tested) state("s-chat-state", "error", said("unreachable"), {text: name, mono: true}, ago(tested.at));
    else if (savedPreset()) {
      const set = saved.key === "env" || saved.key === "keychain";
      state("s-chat-state", set ? "ok" : "error", said(set ? "keySet" : "keyNotSet"), {text: name, mono: true});
    } else if (probe?.reachable === true) state("s-chat-state", "ok", said("connected"), {text: name, mono: true}, checkedWords(probe.checked_s_ago));
    else if (probe?.reachable === false) state("s-chat-state", "error", said("unreachable"), {text: name, mono: true}, checkedWords(probe.checked_s_ago));
    else state("s-chat-state", "busy", said("checking"), {text: name, mono: true});

    const embed = settings.embedding_backend;
    const embedName = embed && embed !== "none" ? settings.sections[embed]?.embedding_model || "" : "";
    if (!embedName) state("s-embed-state", "off", t("common.off"), said("wordsOnly"));
    else if (tested?.ok) state("s-embed-state", "ok", said("connected"), {text: embedName, mono: true}, ago(tested.at));
    else if (tested?.failed === "embedding") state("s-embed-state", "error", said("notAnswering"), {text: embedName, mono: true}, ago(tested.at));
    else state("s-embed-state", "busy", said("notChecked"), {text: embedName, mono: true});

    const choice = settings.choices.chat_backends.find((item) => item.key === backend);
    const hosted = savedPreset();
    $("s-server-address").textContent = !backend ? t("settings.server.none") : hosted ? saved.base_url || hosted.base_url || t("settings.server.ownAddress") : saved.chat_url || "";
    $("s-server-kind").textContent = !choice ? t("settings.server.chooseBelow")
      : hosted ? hostedKind(hosted, saved) : choice.label;
    notes();
  }

  // "DeepSeek · key in the keychain", "OpenAI · key in OPENAI_API_KEY" or "DeepSeek · no key yet".
  function hostedKind(item, data) {
    if (data.key === "env") return t("settings.server.keyIn", {label: item.label, env: data.api_key_env || t("settings.server.anEnvVar")});
    if (data.key === "keychain") return t("settings.server.keyKeychain", {label: item.label});
    if (data.api_key_env) return t("settings.server.keyIn", {label: item.label, env: data.api_key_env});
    return t("settings.server.keyNone", {label: item.label});
  }

  function recall() {
    if (!settings) return;
    const value = RECALL.includes(settings.memory_retriever) ? settings.memory_retriever : "auto";
    for (const name of RECALL) {
      // Default mode offers the plain two; a value set in developer mode shows, so a choice is always marked.
      $(`s-recall-row-${name}`).hidden = !isDeveloper && !["auto", "bm25"].includes(name) && name !== value;
      $(`s-recall-${name}`).checked = name === value;
    }
    for (const node of dialog.querySelectorAll?.("[data-developer]") ?? []) node.hidden = !isDeveloper;
  }

  function folder() {
    if (!settings) return;
    $("s-folder").textContent = settings.workspace;
    const recent = folderChoices({sessions: sessions() || [], fallback: settings.workspace, current: null, limit: 5})
      .filter((choice) => choice.kind === "recent");
    const rows = recent.map((choice) => {
      const row = Object.assign(document.createElement("li"), {className: "st-row"});
      const text = Object.assign(document.createElement("div"), {className: "st-text"});
      text.append(Object.assign(document.createElement("span"), {className: "st-label", textContent: folderName(choice.label)}),
        Object.assign(document.createElement("span"), {className: "st-mono", textContent: choice.label}));
      const make = Object.assign(document.createElement("button"), {type: "button", className: "quiet-button", textContent: t("settings.folder.makeDefault")});
      make.setAttribute("aria-label", t("settings.folder.makeDefaultLabel", {folder: choice.label}));
      make.addEventListener("click", () => setFolder(choice.label.startsWith("~/") || choice.label.startsWith("/") ? choice.label : choice.path));
      const actions = Object.assign(document.createElement("div"), {className: "st-actions"});
      actions.append(make);
      row.append(text, actions);
      return row;
    });
    if (!rows.length) rows.push(Object.assign(document.createElement("li"), {className: "st-empty", textContent: t("settings.folder.recentEmpty")}));
    $("s-recent").replaceChildren(...rows);
    extraFolders();
    // A gateway that does not report it has no mode for new chats to change.
    $("s-approvals-row").hidden = settings.approvals === undefined;
    for (const name of APPROVAL_MODES) $(`s-approvals-${name}`).checked = name === (settings.approvals || "ask");
    // Reading public web pages (`[tools] web_fetch`); a gateway that does not report it has no switch.
    $("s-web-fetch-row").hidden = settings.web_fetch === undefined;
    $("s-web-fetch").checked = settings.web_fetch !== false;
    // Searching the web (`[web] search`, off until switched on); a gateway that does not report it has no switch.
    $("s-web-search-row").hidden = settings.web_search === undefined;
    $("s-web-search").checked = settings.web_search === true;
    // Where melo browses (`[browser] driver`); a gateway that does not report it has no choice. Without
    // Google Chrome a line says to install it first.
    $("s-browser-row").hidden = settings.browser_driver === undefined;
    for (const name of BROWSER_DRIVERS) $(`s-browser-${name}`).checked = name === (settings.browser_driver || "off");
    $("s-browser-chrome-missing").hidden = settings.chrome_installed !== false;
    // Other apps' windows (`[browser] app_windows`): shown while browsing is on; a gateway that does not
    // report it has no switch. Without the desktop control helper, a line says what it needs.
    $("s-app-windows-row").hidden = settings.app_windows === undefined || settings.browser !== true;
    $("s-app-windows").checked = settings.app_windows === true;
    $("s-app-windows-helper").hidden = settings.desktop_helper !== false;
    renderHelper();
    followUp();
    // Reflecting after many steps (`[agent] reflection`, on by default); a gateway that does not report it has no switch.
    $("s-reflection-row").hidden = settings.reflection === undefined;
    $("s-reflection").checked = settings.reflection !== false;
    notes();
  }

  // What the desktop control helper last reported (`GET /api/health?helper=check`, asked once while the row shows a
  // helper that is installed and melo watches other apps' windows): installed, granted, or the grant it lacks.
  function renderHelper() {
    const line = $("s-app-windows-status");
    const watching = settings?.browser === true && settings.app_windows === true && settings.desktop_helper === true;
    if (!watching) { helper = null; helperAsked = false; } else if (!helperAsked) loadHelper();
    const key = watching && helper ? HELPER_LINES[helper.state] : undefined;
    line.hidden = key === undefined;
    if (key === undefined) return;
    const quiet = key === "granted" && helper.pictures === false ? "grantedNoPictures" : key;
    line.textContent = t(`settings.appWindows.status.${quiet}`, {app: helper.app});
  }
  async function loadHelper() {
    helperAsked = true;
    const currentGeneration = generation;
    try {
      const body = await api("/health?helper=check");
      if (currentGeneration !== generation) return;
      helper = body?.desktop_helper ?? null;
    } catch {
      if (currentGeneration !== generation) return;
      helper = null; // the line stays out
    }
    renderHelper();
  }

  // The folders "Always allow this folder" kept (#405): each can be removed, with Undo. One melo never
  // opens is shown struck through and is never used. A gateway that does not report the list hides it.
  function extraFolders() {
    const kept = Array.isArray(settings.extra_folders) ? settings.extra_folders : null;
    for (const id of ["s-extra-title", "s-extra-hint", "s-extra"]) $(id).hidden = kept === null;
    if (kept === null) return;
    const make = (tag, className, text) => Object.assign(document.createElement(tag), {className, textContent: text});
    const rows = kept.map((item) => {
      const path = String(item.path ?? "");
      const row = make("li", "st-row", "");
      row.dataset.valid = String(item.valid !== false);
      const text = make("div", "st-text", "");
      text.append(make("span", "st-label", String(item.name || folderName(path))), make("span", "st-mono", path));
      if (item.valid === false) text.append(make("p", "st-hint", t("settings.extra.invalid")));
      const remove = make("button", "quiet-button", t("settings.extra.remove"));
      remove.type = "button";
      remove.disabled = saving;
      remove.setAttribute("aria-label", t("settings.extra.removeLabel", {folder: path}));
      remove.addEventListener("click", () => removeExtra(path));
      const actions = make("div", "st-actions", "");
      actions.append(remove);
      row.append(text, actions);
      return row;
    });
    if (!rows.length) rows.push(make("li", "st-empty", t("settings.extra.empty")));
    $("s-extra").replaceChildren(...rows);
  }
  function removeExtra(path) {
    const before = (settings?.extra_folders || []).map((item) => String(item.path ?? ""));
    if (!before.includes(path)) return;
    const after = before.filter((item) => item !== path);
    commit({extra_folders: after}, t("settings.extra.removed", {folder: path}),
      () => commit({extra_folders: before}, t("settings.undone", {what: t("settings.extra.restored", {folder: path})})));
  }

  function scheduled() {
    $("s-checkin-hint").textContent = t("sched.checkinLine", {n: checkin?.every_min || 30});
    $("s-checkin").checked = checkin?.on === true;
    if (!checkin) state("s-checkin-state", "off", t(checkin === null ? "sched.reading" : "settings.checkin.unread"));
    else if (checkin.on && checkin.next_at) state("s-checkin-state", "ok", t("common.on"), t("settings.checkin.next", {when: whenWords(checkin.next_at, clock() / 1000)}));
    else state("s-checkin-state", "off", t(checkin.on ? "common.on" : "common.off"));
    $("s-checkin").disabled = !checkin || saving;
    notes();
  }

  function appearance() {
    $("s-desktop-row").hidden = place === null;
    for (const name of DESKTOP) $(`s-desktop-${name}`).checked = name === place;
    $("s-developer").checked = settings?.mode === "developer";
    // A gateway that does not report the language has no language to change.
    $("s-language-row").hidden = settings?.language === undefined;
    for (const name of LANGUAGE_SETTINGS) $(`s-language-${name}`).checked = name === (settings?.language || "auto");
    notes();
  }

  function developerList() {
    if (!settings) return;
    $("settings-file").textContent = t("settings.developerFile", {path: settings.file_label || "~/.cache/localmelo/config.toml"});
    $("settings-revision").textContent = t("settings.revision", {revision: settings.revision || t("settings.notSaved")});
    const make = (tag, className, text) => Object.assign(document.createElement(tag), {className, textContent: text});
    const blocks = [];
    for (const group of settings.developer_settings || []) {
      const list = make("ul", "st-group st-dense", "");
      for (const item of group.keys) {
        const row = make("li", "st-row", "");
        const text = make("div", "st-text", "");
        const about = [item.key, item.flag, item.issue ? t("settings.comparison", {issue: item.issue}) : "", item.file ? "" : t("settings.oneRunOnly")];
        text.append(make("span", "st-label", keyWords(item.key)), make("span", "st-key", about.filter(Boolean).join(" · ")));
        const value = make("span", "st-value st-mono", item.value);
        value.dataset.default = String(item.value === item.default);
        row.append(text, value);
        list.append(row);
      }
      blocks.push(make("h4", "st-group-title", groupName(group.section)), list);
    }
    if (settings.run_options?.length) {
      const list = make("ul", "st-group st-dense", "");
      for (const item of settings.run_options) {
        const row = make("li", "st-row", "");
        const text = make("div", "st-text", "");
        text.append(make("span", "st-label", item.text.charAt(0).toUpperCase() + item.text.slice(1)), make("span", "st-key", item.option));
        row.append(text);
        list.append(row);
      }
      blocks.push(make("h4", "st-group-title", t("settings.runOptions")), list);
    }
    $("s-developer-options").replaceChildren(...blocks);
    notes();
  }

  // The second line of each entry in the phone's list.
  function notes() {
    if (!settings) return;
    const put = (id, text) => { $(`settings-note-${id}`).textContent = text; };
    const chatState = $("s-chat-state").dataset?.text || "";
    put("model", chatState.split(" · ").slice(0, 2).join(" · "));
    const on = $("s-across-chats").checked;
    put("memory", on ? t("settings.note.memoryOn", {how: recallWords(settings.memory_retriever)}) : t("common.off"));
    put("chats", folderName(settings.workspace));
    const granted = permissions.count();
    put("permissions", granted === null ? "" : granted ? t("settings.note.grants", {n: granted}) : t("settings.note.grantsNone"));
    put("scheduled", checkin ? t(checkin.on ? "settings.note.checkinOn" : "settings.note.checkinOff") : "");
    put("appearance", t(`settings.note.theme.${theme}`));
    put("developer", t("settings.note.count", {n: (settings.developer_settings || []).reduce((sum, group) => sum + group.keys.length, 0)}));
  }

  function renderAll() {
    $("s-across-chats").checked = settings.across_chats !== false;
    model();
    recall();
    folder();
    scheduled();
    appearance();
    developerList();
  }

  // --- saving one change ------------------------------------------------------------------------------------

  function saved(text, back) {
    undo = back;
    $("settings-saved-mark").hidden = !text;
    $("settings-saved-text").textContent = text;
    $("settings-undo").hidden = !back;
    foot();
  }

  function conflictOf(err) {
    return err?.status === 409 && err?.code === "settings_changed";
  }

  function failure(err, fallback) {
    const conflict = conflictOf(err);
    const errors = err?.code === "invalid_settings" && Array.isArray(err.details?.errors)
      ? err.details.errors.filter((item) => typeof item === "string" && item).join("\n") : "";
    showError(conflict ? t("settings.conflict") : (errors || err?.message || fallback), conflict);
  }

  async function adopt(answer) {
    settings = answer;
    // Saving succeeded even if the dialog was closed while the request ran.
    try { await onSaved?.(answer); } catch { /* The caller owns refresh failures. */ }
    layout(answer.mode === "developer");
    renderAll();
  }

  /** Save `change` over the settings as loaded; `text` says what changed, `back` is the change that undoes it. */
  async function commit(change, text, back = null) {
    if (saving || busy || !settings) return false;
    if (!settings.chat_backend) {
      showError(t("settings.chooseFirst"));
      renderAll();
      return false;
    }
    const currentGeneration = generation;
    saving = true;
    enable(true);
    showError("");
    saved("", null);
    try {
      const body = {chat_backend: settings.chat_backend, embedding_backend: settings.embedding_backend || "none",
        workspace: settings.workspace, sections: {}, revision: settings.revision, ...change};
      const answer = await api("/settings", {method: "PUT", body});
      if (currentGeneration !== generation) {
        // Saving succeeded even if the dialog was closed while the request ran.
        try { await onSaved?.(answer); } catch { /* The caller owns refresh failures. */ }
        return true;
      }
      await adopt(answer);
      saved(back ? t("settings.saved", {what: text}) : text, back);
      return true;
    } catch (err) {
      if (currentGeneration !== generation) return false;
      failure(err, t("settings.saveFailed"));
      renderAll();
      return false;
    } finally {
      if (currentGeneration === generation) {
        saving = false;
        enable(!busy);
      }
    }
  }

  async function undoLast() {
    const back = undo;
    if (!back || saving || busy) return;
    saved("", null);
    await back();
  }
  $("settings-undo").addEventListener("click", undoLast);

  function settingCommit(field, value, text, previous, previousText) {
    return commit({[field]: value}, text, () => commit({[field]: previous}, t("settings.undone", {what: previousText})));
  }

  // --- what each control does -------------------------------------------------------------------------------

  $("s-across-chats").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings) { event.target.checked = !on; return; }
    if (settings.across_chats === undefined) return; // an older gateway saves it with the connection
    settingCommit("across_chats", on, t(on ? "settings.across.on" : "settings.across.off"), !on, t(on ? "settings.across.off" : "settings.across.on"));
  });
  $("s-web-fetch").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings || settings.web_fetch === undefined) { event.target.checked = !on; return; }
    const words = (value) => t(value ? "settings.webFetch.on" : "settings.webFetch.off");
    settingCommit("web_fetch", on, words(on), !on, words(!on));
  });
  $("s-web-search").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings || settings.web_search === undefined) { event.target.checked = !on; return; }
    const words = (value) => t(value ? "settings.webSearch.on" : "settings.webSearch.off");
    settingCommit("web_search", on, words(on), !on, words(!on));
  });
  $("s-app-windows").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings || settings.app_windows === undefined) { event.target.checked = !on; return; }
    const words = (value) => t(value ? "settings.appWindows.on" : "settings.appWindows.off");
    settingCommit("app_windows", on, words(on), !on, words(!on));
  });
  $("s-follow-up").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings || settings.follow_up === undefined) { event.target.checked = !on; return; }
    const words = (value) => t(value ? "settings.followUp.on" : "settings.followUp.off");
    settingCommit("follow_up", on, words(on), !on, words(!on));
  });
  $("s-reflection").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings || settings.reflection === undefined) { event.target.checked = !on; return; }
    const words = (value) => t(value ? "settings.reflection.on" : "settings.reflection.off");
    settingCommit("reflection", on, words(on), !on, words(!on));
  });
  for (const name of RECALL) {
    $(`s-recall-${name}`).addEventListener("change", (event) => {
      if (!event.target.checked || !settings) return;
      const before = settings.memory_retriever || "auto";
      if (saving) { recall(); return; }
      settingCommit("memory_retriever", name, t("settings.recallSaved", {how: recallWords(name)}), before, t("settings.recallSaved", {how: recallWords(before)}));
    });
  }
  $("s-developer").addEventListener("change", (event) => {
    const on = event.target.checked;
    if (saving || !settings) { event.target.checked = !on; return; }
    const mode = on ? "developer" : "default";
    const other = on ? "default" : "developer";
    commit({mode}, t(on ? "settings.developerMode.on" : "settings.developerMode.off"),
      () => commit({mode: other}, t(on ? "settings.developerMode.undoneOff" : "settings.developerMode.undoneOn")));
  });

  async function setFolder(path) {
    const before = settings?.workspace;
    if (!before) return;
    if (path === before) { folderEdit(false); return; }
    const done = await commit({workspace: path}, t("settings.folder.saved", {path}),
      () => commit({workspace: before}, t("settings.folder.undone", {path: before})));
    if (done) folderEdit(false);
  }
  function folderEdit(open) {
    $("s-folder-edit").hidden = !open;
    $("s-folder-change").setAttribute("aria-expanded", String(open));
    if (open) {
      $("s-workspace").value = settings?.workspace || "";
      $("s-workspace").removeAttribute?.("aria-invalid");
      $("s-workspace").focus?.();
    }
  }
  $("s-folder-change").addEventListener("click", async () => {
    const handler = folders();
    if (!handler) return folderEdit($("s-folder-change").getAttribute("aria-expanded") !== "true");
    const picked = await handler.postMessage({action: "choose"}).catch(() => null);
    if (typeof picked === "string" && picked.startsWith("/")) await setFolder(picked);
    return undefined;
  });
  $("s-folder-cancel").addEventListener("click", () => folderEdit(false));
  async function usePath() {
    const text = $("s-workspace").value.trim();
    if (!/^(\/|~\/)\S/.test(text)) {
      $("s-workspace").setAttribute("aria-invalid", "true");
      $("s-folder-hint").textContent = t("settings.folder.notFull");
      return;
    }
    $("s-folder-hint").textContent = t("settings.folder.hint");
    await setFolder(text);
  }
  $("s-folder-use").addEventListener("click", usePath);
  $("s-workspace").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    event.preventDefault();
    usePath();
  });

  async function loadCheckin() {
    const currentGeneration = generation;
    try {
      const page = await api("/chores");
      if (currentGeneration !== generation) return;
      checkin = page.checkin ?? false;
    } catch {
      if (currentGeneration !== generation) return;
      checkin = checkin || false;
    }
    scheduled();
  }
  async function setCheckin(on) {
    const currentGeneration = generation;
    saving = true;
    showError("");
    saved("", null);
    enable(true);
    try {
      const done = await api("/checkin", {method: "PUT", body: {on}});
      if (currentGeneration !== generation) return false;
      checkin = done.checkin ?? {...checkin, on};
      return true;
    } catch (err) {
      if (currentGeneration !== generation) return false;
      failure(err, t("settings.checkin.failed"));
      return false;
    } finally {
      if (currentGeneration === generation) {
        saving = false;
        enable(!busy);
        scheduled();
      }
    }
  }
  $("s-checkin").addEventListener("change", async (event) => {
    const on = event.target.checked;
    if (saving || !checkin) { event.target.checked = !on; return; }
    if (await setCheckin(on)) {
      saved(t(on ? "settings.checkin.savedOn" : "settings.checkin.savedOff"), async () => {
        if (await setCheckin(!on)) saved(t(on ? "settings.checkin.undoneOff" : "settings.checkin.undoneOn"), null);
      });
    }
  });

  for (const name of THEMES) {
    $(`settings-theme-${name}`).addEventListener("change", (event) => {
      if (!event.target.checked) return;
      const before = theme;
      storeTheme(name);
      if (dialog.open && before !== name) {
        saved(t(`settings.theme.saved.${name}`), () => { storeTheme(before); saved(t(`settings.theme.undone.${before}`), null); });
      }
    });
  }

  async function setPlace(mode) {
    const handler = desktop();
    if (!handler) return false;
    const answer = await handler.postMessage({action: "set", mode}).catch(() => null);
    if (!DESKTOP.includes(answer)) {
      showError(t("settings.desktop.refused"));
      appearance();
      return false;
    }
    place = answer;
    appearance();
    return true;
  }
  for (const name of DESKTOP) {
    $(`s-desktop-${name}`).addEventListener("change", async (event) => {
      if (!event.target.checked || place === null) return;
      const before = place;
      showError("");
      if (await setPlace(name)) {
        saved(t(`settings.desktop.saved.${name}`), async () => {
          if (await setPlace(before)) saved(t(`settings.desktop.undone.${before}`), null);
        });
      }
    });
  }
  async function readPlace() {
    const handler = desktop();
    const answer = handler ? await handler.postMessage({action: "get"}).catch(() => null) : null;
    place = DESKTOP.includes(answer) ? answer : null;
    appearance();
  }

  for (const name of BROWSER_DRIVERS) {
    $(`s-browser-${name}`).addEventListener("change", (event) => {
      if (!event.target.checked || !settings) return;
      const before = settings.browser_driver || "off";
      if (saving || settings.browser_driver === undefined) { folder(); return; }
      if (name === "chrome" && settings.chrome_installed === false) {
        // Chrome is not there to start: the choice stays, and the line under it says why.
        folder();
        showError(t("settings.browser.chromeMissing"));
        return;
      }
      // From one browser to the other: each keeps its own sign-ins, and an open chat switches at its next message.
      const switched = (driver, from) => driver !== "off" && from !== "off" && driver !== from;
      const words = (driver, from) => t(switched(driver, from) ? "settings.browser.savedSwitch" : "settings.browser.saved",
        {browser: t(`settings.browser.${BROWSER_WORDS[driver]}`)});
      settingCommit("browser_driver", name, words(name, before), before, words(before, name));
    });
  }

  for (const name of APPROVAL_MODES) {
    $(`s-approvals-${name}`).addEventListener("change", (event) => {
      if (!event.target.checked || !settings) return;
      const before = settings.approvals || "ask";
      if (saving || settings.approvals === undefined) { folder(); return; }
      const words = (mode) => t("settings.approvals.saved", {mode: t(`auto.mode.${mode}`)});
      settingCommit("approvals", name, words(name), before, words(before));
    });
  }

  // A new language is saved like any setting, then the page loads again in it.
  for (const name of LANGUAGE_SETTINGS) {
    $(`s-language-${name}`).addEventListener("change", async (event) => {
      if (!event.target.checked || !settings) return;
      if (saving || settings.language === undefined) { appearance(); return; }
      if (await commit({language: name}, t("settings.language.saved"))) reloadPage();
    });
  }

  $("s-open-memory").addEventListener("click", () => openTab("memory"));
  $("s-open-scheduled").addEventListener("click", () => openTab("scheduled"));
  $("s-change").addEventListener("click", () => {
    const open = $("s-change").getAttribute("aria-expanded") !== "true";
    // The form starts from the saved connection, and closing it drops what was typed.
    restoreDrafts();
    editor(open);
    if (open) {
      $("s-editor").scrollIntoView?.({block: "nearest"});
      $("s-adapter").focus?.({preventScroll: true});
    }
  });
  $("s-editor-cancel").addEventListener("click", () => {
    restoreDrafts();
    editor(false);
    $("s-change").focus?.();
  });
  // Cancel puts the saved connection back in the form.
  function restoreDrafts() {
    if (!settings) return;
    drafts(settings);
    invalidateTest();
    renderChat();
    renderEmbedding();
    renderOffer();
    enable(!busy);
  }

  // A key typed and not saved does not stay in the page when the dialog closes or opens again.
  function forgetKey() {
    $("s-key-input").value = "";
    if (removeArmed) clearTimeout(removeArmed);
    removeArmed = null;
    replacing = false;
  }

  // The address of a free-form service is needed: left empty it would be OpenAI's own.
  function addressProblem() {
    const item = preset();
    return item?.free_address && !section(item.id).base_url?.trim() ? t("settings.hosted.needAddress") : "";
  }

  // --- open, load, test, save the connection ----------------------------------------------------------------

  async function open(name) {
    const currentGeneration = ++generation;
    discoveryVersion += 1;
    keyVersion += 1;
    keychainVersion += 1;
    forgetKey();
    settings = null;
    busy = true;
    saving = false;
    tested = null;
    checkin = null;
    helper = null;
    helperAsked = false;
    permissions.reset();
    invalidateTest();
    enable(false);
    showError("");
    saved("", null);
    status.textContent = t("settings.loading");
    modelStatus.textContent = "";
    keySays("");
    envStatus.textContent = "";
    dialog.dataset.ready = "false";
    layout(developer());
    // A named section opens as itself; otherwise a phone starts on the list.
    select(typeof name === "string" ? name : current, {show: true});
    if (narrow() && typeof name !== "string") view("list");
    if (!dialog.open) dialog.showModal();
    $(`settings-tab-${current}`).focus?.();
    readPlace();
    try {
      const loaded = await api("/settings");
      if (currentGeneration !== generation || !dialog.open) return;
      settings = loaded;
      drafts(loaded);
      layout(loaded.mode === "developer");
      busy = false;
      renderChat();
      renderEmbedding();
      renderOffer();
      renderAll();
      editor(!loaded.chat_backend);
      enable(true);
      // "not set (run 'melo setup')" is advice for the terminal: here the empty select says it.
      const errors = (loaded.errors || []).filter((item) => typeof item === "string" && item && !terminalAdvice(item));
      status.textContent = !loaded.chat_backend ? t("settings.chooseToContinue") : errors.length ? t("settings.review") : "";
      showError(errors.join("\n"));
      dialog.dataset.ready = "true";
      loadCheckin();
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      busy = false;
      status.textContent = "";
      showError(err.message || t("settings.loadFailed"), true);
    }
  }

  dialog.addEventListener("close", () => {
    generation += 1;
    discoveryVersion += 1;
    keyVersion += 1;
    keychainVersion += 1;
    forgetKey();
    saving = false;
    invalidateTest();
  });
  reload.addEventListener("click", () => open(current));
  chat.addEventListener("change", () => { invalidateTest(); renderChat(); enable(!busy); });
  embedding.addEventListener("change", () => { invalidateTest(); renderEmbedding(); renderOffer(); });
  $("s-embed-use").addEventListener("change", () => {
    invalidateTest();
    if ($("s-embed-use").checked) {
      // The embedding model is served beside the chat model: same backend, same address.
      if (!offered || !settings.choices.embedding_backends.some((choice) => choice.key === chat.value)) return renderOffer();
      embedding.value = chat.value;
      Object.assign(section(chat.value), {embedding_url: $("s-chat-url").value, embedding_model: offered});
    } else {
      embedding.value = "none";
    }
    renderEmbedding();
    return renderOffer();
  });
  for (const [id, field] of [["s-chat-url", "chat_url"], ["s-chat-model", "chat_model"],
    ["s-key-env", "api_key_env"], ["s-base-url", "base_url"]]) {
    $(id).addEventListener("input", () => {
      section(chat.value)[field] = $(id).value;
      invalidateTest();
      if (id === "s-key-env") checkKey();
      if (id === "s-chat-model" && isCloud()) $("s-thinking").checked = thinks(preset(), section(chat.value));
      if (id === "s-base-url") $("s-address-text").textContent = t("settings.hosted.addressIs", {address: $(id).value});
      if (id === "s-chat-url") {
        discoveryVersion += 1;
        $("s-model-options").replaceChildren();
        modelStatus.textContent = words.discover;
        discover.disabled = busy || isCloud();
      }
    });
  }
  $("s-thinking").addEventListener("change", (event) => {
    if (!isCloud()) return;
    section(chat.value).thinking = event.target.checked ? "on" : "off";
    invalidateTest();
  });

  // The key is sent once, over this page's own session with the gateway, and leaves the field at once.
  async function saveKey() {
    const item = preset();
    const key = $("s-key-input").value.trim();
    if (!item || busy || !settings) return;
    if (!key) {
      keySays(t("settings.hosted.keyEmpty"));
      return;
    }
    const currentGeneration = generation;
    $("s-key-save").disabled = true;
    keySays(t("settings.hosted.keySaving"));
    try {
      const result = await api("/settings/key", {method: "PUT", body: {preset: item.id, key}});
      if (currentGeneration !== generation || !dialog.open) return;
      $("s-key-input").value = "";
      const data = section(item.id);
      data.api_key = result.api_key || item.key_ref;
      // A key pasted on purpose is the one to use: a variable named before it would win over it.
      const cleared = !!data.api_key_env;
      if (cleared) { data.api_key_env = ""; $("s-key-env").value = ""; envStatus.textContent = ""; }
      keychain.set(item.id, "saved");
      replacing = false;
      invalidateTest();
      renderKey();
      // The saved row already shows "Saved in the keychain", so a plain save is only announced, not shown twice.
      keySays(t(cleared ? "settings.hosted.keySavedCleared" : "settings.hosted.keySaved"), !cleared);
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      keySays(err.message || t("settings.hosted.keyFailed"));
    } finally {
      if (currentGeneration === generation && dialog.open) $("s-key-save").disabled = busy || !settings;
    }
  }
  $("s-key-save").addEventListener("click", saveKey);
  $("s-key-input").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    event.preventDefault();
    saveKey();
  });
  $("s-key-replace").addEventListener("click", () => {
    replacing = true;
    keySays("");
    renderKey();
    $("s-key-input").focus?.();
  });
  // Removing a key cannot be undone (a service may not show it again), so it takes two clicks.
  $("s-key-remove").addEventListener("click", async () => {
    const item = preset();
    if (!item || busy || !settings) return;
    if (!removeArmed) {
      removeArmed = setTimeout(() => { removeArmed = null; renderKey(); }, REMOVE_WAIT_MS);
      renderKey();
      return;
    }
    clearTimeout(removeArmed);
    removeArmed = null;
    const currentGeneration = generation;
    try {
      await api(`/settings/key?${new URLSearchParams({preset: item.id})}`, {method: "DELETE"});
      if (currentGeneration !== generation || !dialog.open) return;
      section(item.id).api_key = "";
      keychain.set(item.id, "missing");
      invalidateTest();
      keySays(t("settings.hosted.keyRemoved"));
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      keySays(err.message || t("settings.hosted.keyFailed"));
    }
    renderKey();
  });

  for (const [id, field] of [["s-embed-url", "embedding_url"], ["s-embed-model", "embedding_model"]]) {
    $(id).addEventListener("input", () => { section(embedding.value)[field] = $(id).value; invalidateTest(); });
  }

  discover.addEventListener("click", async () => {
    if (busy || !settings || isCloud()) return;
    const version = ++discoveryVersion;
    const currentGeneration = generation;
    const params = new URLSearchParams({backend: chat.value, url: $("s-chat-url").value});
    if (!$("s-chat-url").value) {
      modelStatus.textContent = words.needUrl;
      return;
    }
    discover.disabled = true;
    modelStatus.textContent = t("settings.discovering");
    try {
      const result = await api(`/models?${params}`);
      if (version !== discoveryVersion || currentGeneration !== generation || !dialog.open) return;
      $("s-model-options").replaceChildren();
      for (const name of result.models) {
        const option = document.createElement("option");
        option.value = name;
        $("s-model-options").append(option);
      }
      modelStatus.textContent = t(result.models.length ? "settings.discovered" : "settings.noModels");
      offered = result.suggested_embedding || "";
      renderOffer();
    } catch (err) {
      if (version !== discoveryVersion || currentGeneration !== generation || !dialog.open) return;
      modelStatus.textContent = err.message || t("settings.discoverFailed");
    } finally {
      if (version === discoveryVersion && currentGeneration === generation && dialog.open) discover.disabled = busy || isCloud();
    }
  });

  testButton.addEventListener("click", async () => {
    if (busy || !settings || testButton.disabled) return;
    const missing = addressProblem();
    if (missing) {
      testStatus.textContent = missing;
      testStatus.dataset.state = "error";
      return;
    }
    const version = ++testVersion;
    const currentGeneration = generation;
    testButton.disabled = true;
    testStatus.textContent = t("settings.testing");
    testStatus.dataset.state = "testing";
    const both = embedding.value !== "none" && !!embedding.value;
    const firstLine = (text) => (typeof text === "string" ? text.split(/\r?\n/)[0] : "") || t("settings.test.unknown");
    try {
      const result = await api("/settings/test", {method: "POST", body: candidate()});
      if (version !== testVersion || currentGeneration !== generation || !dialog.open) return;
      // A hosted service answers in a sentence of its own, and lists the models it has.
      const hosted = typeof result.result === "string";
      if (hosted && Array.isArray(result.models) && result.models.length) {
        fillModels(preset()?.models || [], result.models);
        modelStatus.textContent = t("settings.hosted.modelsListed", {n: result.models.length});
      }
      // The saved connection's rows show a test only when the form is that connection: a service tried
      // in the form says so beside the button and leaves the saved one's state alone.
      const mine = draftIsSaved();
      if (result.ok === true) {
        tested = mine ? {at: clock(), ok: true} : null;
        testStatus.textContent = hosted ? result.message : t(both ? "settings.test.both" : "settings.test.chat");
        testStatus.dataset.state = "success";
      } else {
        tested = mine ? {at: clock(), ok: false, failed: result.failed === "embedding" ? "embedding" : "chat"} : null;
        testStatus.textContent = hosted ? firstLine(result.error) : t("settings.test.failed", {problem: firstLine(result.error)});
        testStatus.dataset.state = "error";
      }
    } catch (err) {
      if (version !== testVersion || currentGeneration !== generation || !dialog.open) return;
      testStatus.textContent = t("settings.test.failed", {problem: firstLine(err.message)});
      testStatus.dataset.state = "error";
    } finally {
      if (version === testVersion && currentGeneration === generation && dialog.open) {
        testButton.disabled = busy || !settings;
        model();
      }
    }
  });

  save.addEventListener("click", async () => {
    if (busy || saving || !settings || !chat.value) return;
    const missing = addressProblem();
    if (missing) {
      showError(missing);
      return;
    }
    const currentGeneration = generation;
    const before = settings;
    const across = settings.across_chats === undefined ? {} : {across_chats: $("s-across-chats").checked};
    const body = {...candidate(), revision: settings.revision, mode: $("s-developer").checked ? "developer" : "default", ...across};
    busy = true;
    invalidateTest();
    enable(false);
    showError("");
    saved("", null);
    status.textContent = t("settings.savingConnection");
    try {
      const answer = await api("/settings", {method: "PUT", body});
      if (currentGeneration !== generation || !dialog.open) {
        try { await onSaved?.(answer); } catch { /* The caller owns refresh failures. */ }
        return;
      }
      busy = false;
      await adopt(answer);
      drafts(answer);
      renderChat();
      renderEmbedding();
      renderOffer();
      editor(false);
      status.textContent = "";
      const back = before.chat_backend
        ? () => commit(savedConnection(before), t("settings.connection.undone")) : null;
      saved(t(back ? "settings.connection.saved" : "settings.connection.savedTest"), back);
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      status.textContent = "";
      failure(err, t("settings.connection.failed"));
    } finally {
      if (currentGeneration === generation && dialog.open) {
        busy = false;
        enable(true);
      }
    }
  });

  return open;
}
