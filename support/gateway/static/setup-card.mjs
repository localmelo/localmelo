/**
 * The welcome's setup card: connect the model server that already runs on this Mac.
 *
 * While the page shows it (`show(true)`), it asks `GET /api/setup/detect` and draws what it
 * found: a server with a model to use, a server that needs an API key, or nothing. "Use this
 * model" saves the choice as the settings dialog does (`PUT /api/settings` with the revision
 * just read), runs the connection test, and calls `connected()` so the page reads its health
 * again. Hidden, the card asks nothing. It never picks a model for the user: with several
 * models and none known to be loaded, the user chooses from the list.
 */
const QUICKSTART_URL = "https://localmelo.github.io/localmelo/en/quickstart.html";

const host = (url) => {
  try {
    return new URL(url).host;
  } catch {
    return url || "";
  }
};
const firstLine = (error) => String(error.message).split(/\r?\n/)[0];

export function setupModelCard({root, api, openSettings, connected}) {
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

  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const button = (className, label, onClick) => {
    const node = el("button", className, label);
    node.type = "button";
    node.addEventListener("click", onClick);
    return node;
  };

  /** The words of the card's state, and its main action. */
  function copy() {
    const who = data?.name || "a model server";
    const where = host(data?.url);
    if (phase === "checking") return {title: "Looking for a model server on this Mac…"};
    if (phase === "connecting") {
      return {title: `Connecting to ${who}…`, text: "Checking that the model answers. One that is not loaded yet can take a minute."};
    }
    if (phase === "failed") {
      const title = data ? `Could not connect to ${who}` : "Could not look for a model server";
      return {title, text: failure, label: "Try again", run: data ? use : detect};
    }
    if (data.state === "found") {
      return {title: `Found ${who} on this Mac`, text: `Running at ${where}.`, label: "Use this model", run: use};
    }
    if (data.state === "needs_key") {
      const text = `The server at ${where} asks for a key, and localmelo does not send keys to servers on this Mac. If it is oMLX, turn on “Allow inference without API key” in its settings.`;
      return {title: "A model server needs an API key", text, label: "Check again", run: detect};
    }
    const answered = data.url !== "";
    return {
      title: answered ? `Found ${who}, but no model is ready` : "No model server is running on this Mac",
      text: answered ? `It answered at ${where}: ${data.problem}.` : "Start oMLX or Ollama, then check again.",
      label: "Check again",
      run: detect,
      guide: true,
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
    if (data.loaded.includes(id)) row.append(el("span", "setup-loaded", "Loaded"));
    return row;
  }

  function render() {
    refocus = root.contains(doc.activeElement);
    const {title, text, label, run, guide} = copy();
    const settled = phase === "ready" || phase === "failed";
    const found = settled && data?.state === "found";
    const list = found && phase === "ready" && (choosing || !data.suggested);
    const heading = el("h2", "", title);
    heading.id = "setup-title";
    const parts = [heading];
    if (text) parts.push(el("p", "", text));
    const main = label ? button("primary-button", label, run) : null;
    if (list) {
      const group = el("fieldset", "setup-models");
      group.append(el("legend", "", "Pick the model to use"));
      group.append(...data.models.map((id) => modelRow(id, (value) => {
        picked = value;
        main.disabled = false;
      })));
      parts.push(group);
      if (data.loaded.length < data.models.length) {
        parts.push(el("p", "", "A model that is not loaded yet can take minutes and a lot of memory to load."));
      }
      main.disabled = !picked;
    } else if (found && phase === "ready") {
      parts.push(modelRow(data.suggested));
    }
    const actions = el("div", "setup-actions");
    if (main) actions.append(main);
    if (found && !list && data.models.length > 1) actions.append(button("link-button", "Choose another…", choose));
    if (phase === "failed" || (phase === "ready" && !found)) actions.append(button("link-button", "Enter a server address…", openSettings));
    if (guide) {
      const link = el("a", "link-button", "Read the quickstart");
      Object.assign(link, {href: QUICKSTART_URL, target: "_blank", rel: "noopener noreferrer"});
      actions.append(link);
    }
    if (actions.children.length) parts.push(actions);
    root.replaceChildren(...parts);
    root.setAttribute("aria-busy", String(phase === "checking" || phase === "connecting"));
    if (refocus) root.focus(); // The control that was clicked is gone: the next Tab reaches the first one.
  }

  function choose() {
    choosing = true;
    phase = "ready";
    render();
  }

  async function detect() {
    const mine = ++version;
    phase = "checking";
    data = null;
    picked = "";
    choosing = false;
    render();
    try {
      const found = await api("/setup/detect");
      if (mine !== version) return;
      data = found;
      picked = found.suggested || "";
      phase = "ready";
    } catch (error) {
      if (mine !== version) return;
      failure = firstLine(error);
      phase = "failed";
    }
    render();
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
      if (test.ok !== true) throw new Error(test.error || "The model did not answer.");
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
      if (visible) {
        detect();
      } else {
        phase = "idle";
        data = null;
      }
    },
    /** Whether a connection is being made or has just failed: the page keeps the card up. */
    holding: () => phase === "connecting" || phase === "failed",
  };
}
