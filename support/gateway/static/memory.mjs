/**
 * What memory across chats did for a message, shown under it (the `turn.memory` event).
 *
 * "Used 2 memories ▾" opens the notes and earlier chats the turn recalled, each with where it came
 * from. "Remembered: you prefer short answers · Undo" names a note the message made melo keep; Undo
 * deletes it (`DELETE /api/memory/records/{id}?undo=true`). "Not remembered: …" says why a
 * statement was not kept. Nothing here fetches: `workspace.js` passes the undo.
 */

function list(value) {
  return Array.isArray(value) ? value.filter((item) => item !== null && typeof item === "object") : [];
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
  return used.length || remembered.length || refused ? { used, remembered, refused } : null;
}

export function usedLabel(count) {
  return `Used ${count} ${count === 1 ? "memory" : "memories"}`;
}

/** A note inside a sentence: "Remembered: you prefer…", a name keeps its capital. */
export function inline(text) {
  return /^Your?\b/.test(text) ? text.charAt(0).toLowerCase() + text.slice(1) : text;
}

/** The nodes under a message: `{doc, memory, open, toggle, undo}`; [] for none. */
export function memoryNodes({ doc, memory, open, toggle, undo }) {
  if (!memory) return [];
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
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
  for (const note of memory.remembered) {
    const line = el("p", "", `${note.undone ? "Forgotten" : "Remembered"}: ${inline(note.text)}`);
    if (!note.undone) line.append(el("span", "", " · "), action("Undo", () => undo(note)));
    box.append(line);
  }
  if (memory.refused) box.append(el("p", "", `Not remembered: ${memory.refused}`));
  return [box];
}
