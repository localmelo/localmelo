/**
 * What memory across chats did for a message, shown under it (the `turn.memory` event).
 *
 * "Used 2 memories ▾" opens the notes and earlier chats the turn recalled, each with where it came
 * from. "Remembered: you prefer short answers · Undo" names a note the message made melo keep; Undo
 * deletes it (`DELETE /api/memory/records/{id}?undo=true`). "Not remembered: …" says why a
 * statement was not kept. "From your other chats (last 30 min) ▾" opens the lines a new chat's
 * first message carried about the other chats (`digest`), each as the model read it; the chat's
 * name opens that chat when the list has it. It is context, not a note: no Undo. Nothing here
 * fetches: `workspace.js` passes the undo and the opening of a chat.
 */
import { language, t } from "./i18n.mjs";
import { maker } from "./dom.mjs";

function list(value) {
  return Array.isArray(value) ? value.filter((item) => item !== null && typeof item === "object") : [];
}

/** The other chats' digest of the event (`{minutes, items: [{session_id, turn, line}]}`), or null. */
function readDigest(value) {
  if (!value || typeof value !== "object") return null;
  const minutes = Math.round(Number(value.minutes));
  const items = list(value.items)
    .map((item) => ({ sessionId: String(item.session_id ?? ""), turn: Number(item.turn) || 0, line: String(item.line ?? "") }))
    .filter((item) => item.line);
  return minutes > 0 && items.length ? { minutes, items } : null;
}

/** The event's data, kept to what the page can use; null when it says nothing. */
export function readMemory(data) {
  const used = list(data?.used)
    .map((item) => ({ text: String(item.text ?? ""), source: String(item.source ?? "") }))
    .filter((item) => item.text);
  const remembered = list(data?.remembered)
    .map((item) => ({ id: Number(item.id) || 0, text: String(item.text ?? ""), undone: false }))
    .filter((item) => item.id > 0 && item.text);
  const refused = typeof data?.refused === "string" ? data.refused : "";
  const digest = readDigest(data?.digest);
  return used.length || remembered.length || refused || digest ? { used, remembered, refused, digest } : null;
}

/** A digest line as its two parts: the chat and time it names, and what was said there. */
export function digestParts(line) {
  const cut = line.indexOf(" · ");
  return cut > 0 ? { where: line.slice(0, cut), said: line.slice(cut) } : { where: "", said: line };
}

export function usedLabel(count) {
  return t("memory.used", { n: count });
}

/** A note inside an English sentence: "Remembered: you prefer…", a name keeps its capital. */
export function inline(text) {
  return language() === "en" && /^Your?\b/.test(text) ? text.charAt(0).toLowerCase() + text.slice(1) : text;
}

/**
 * The nodes under a message: `{doc, memory, open, toggle, undo, digestOpen, toggleDigest, openChat}`;
 * [] for none. *openChat*, when given, takes a chat's id and returns a function that opens it, or
 * null when the page cannot (a terminal chat, a chat deleted since).
 */
export function memoryNodes({ doc, memory, open, toggle, undo, digestOpen = false, toggleDigest = () => {}, openChat = null }) {
  if (!memory) return [];
  const {node: el} = maker(doc);
  const action = (text, onClick, expanded) => {
    const node = el("button", "", text);
    node.type = "button";
    if (expanded !== undefined) node.setAttribute("aria-expanded", String(expanded));
    node.addEventListener("click", onClick);
    return node;
  };
  const box = el("div", "memory-notes");
  if (memory.used.length) {
    box.append(action(`${usedLabel(memory.used.length)} ${open ? "▴" : "▾"}`, toggle, open));
    if (open) {
      const items = el("ul", "memory-list");
      for (const item of memory.used) {
        const row = el("li", "", item.text);
        row.append(el("span", "memory-source", ` (${item.source})`));
        items.append(row);
      }
      box.append(items);
    }
  }
  if (memory.digest) {
    const label = t("memory.fromOtherChats", { n: memory.digest.minutes });
    box.append(action(`${label} ${digestOpen ? "▴" : "▾"}`, toggleDigest, digestOpen));
    if (digestOpen) {
      const items = el("ul", "memory-list memory-digest");
      for (const item of memory.digest.items) {
        const { where, said } = digestParts(item.line);
        const row = el("li");
        const open = where ? openChat?.(item.sessionId) : null;
        if (open) row.append(action(where, open));
        else if (where) row.append(el("span", "memory-source", where));
        row.append(el("span", "", said));
        items.append(row);
      }
      box.append(items);
    }
  }
  for (const note of memory.remembered) {
    const line = el("p", "", t(note.undone ? "memory.forgotten" : "memory.remembered", { note: inline(note.text) }));
    if (!note.undone) line.append(el("span", "", " · "), action(t("memory.undo"), () => undo(note)));
    box.append(line);
  }
  if (memory.refused) box.append(el("p", "", t("memory.notRemembered", { reason: memory.refused })));
  return [box];
}
