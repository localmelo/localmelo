import { t } from "./i18n.mjs";
import { formatBytes } from "./turns.mjs";

/**
 * Pictures and text files in the composer: what a drop, a paste or the picker gives, the strip of
 * thumbnails above the box, and what a send takes from it.
 *
 * A picture goes to the gateway as soon as it is added (`POST /api/uploads` reads, converts and
 * scales it, and answers with an id and whether the chat model reads pictures); a text file is read
 * here and travels in the message. Both wait in the strip, each with a remove button. The numbers
 * mirror the gateway's (`picture_input.py`, `sessions/attachments.py`); the gateway judges again.
 * While melo works, the strip's pictures go with the words the person adds to the turn (a note);
 * its text files cannot, and `host.refuse` says so.
 */

/** Pictures one message may carry: a vision model reads about a thousand tokens per picture. */
export const MAX_PICTURES = 4;
export const MAX_FILES = 8;
export const MAX_FILE_BYTES = 256 * 1024;
export const MAX_TOTAL_BYTES = 512 * 1024;
/** What a picture may weigh on the way in; the gateway scales what it keeps. */
export const MAX_PICTURE_BYTES = 20 * 1024 * 1024;

const PICTURE_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp", "image/heic", "image/heif"]);
const PICTURE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp", "heic", "heif"]);
const TEXT_TYPES = new Set([
  "application/json",
  "application/xml",
  "application/x-yaml",
  "application/yaml",
  "application/toml",
  "application/x-sh",
  "application/javascript",
  "application/sql",
]);
const TEXT_EXTENSIONS = new Set([
  "txt", "md", "markdown", "rst", "log", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
  "py", "pyi", "js", "mjs", "cjs", "ts", "tsx", "jsx", "html", "htm", "css", "scss", "xml", "svg", "sh", "bash", "zsh", "fish",
  "swift", "rs", "go", "java", "kt", "c", "h", "cc", "cpp", "hpp", "m", "mm", "rb", "php", "pl", "lua", "r", "sql", "tex", "bib",
  "diff", "patch", "gitignore", "dockerfile", "makefile",
]);

function extensionOf(name) {
  const dot = String(name).lastIndexOf(".");
  return dot < 0 ? String(name).toLowerCase() : String(name).slice(dot + 1).toLowerCase();
}

/** `picture`, `text` or `other` for a file by its type and name (the gateway reads the bytes again). */
export function kindOf(file) {
  const type = String(file.type ?? "").toLowerCase();
  const extension = extensionOf(file.name ?? "");
  if (PICTURE_TYPES.has(type) || (!type.startsWith("text/") && PICTURE_EXTENSIONS.has(extension))) return "picture";
  if (type.startsWith("text/") || TEXT_TYPES.has(type) || TEXT_EXTENSIONS.has(extension)) return "text";
  return "other";
}

/** The Content-Type an upload is sent with: the picture's own, else a generic one the gateway sniffs. */
export function uploadType(file) {
  const type = String(file.type ?? "").toLowerCase();
  return PICTURE_TYPES.has(type) ? type : "application/octet-stream";
}

const ENTITIES = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'" };

/**
 * A message's words and the text files it carries. The gateway writes the files after the words as
 * `<attachment name chars>` blocks (`compose_user_text`); a text that does not end in whole blocks
 * has no files. The words that stood in for none (`[2 pictures]`) are the caller's to hide.
 */
export function splitAttachments(userText) {
  const text = String(userText ?? "");
  const start = text.indexOf('\n\n<attachment name="');
  if (start < 0) return { words: text, files: [] };
  const block = /\n\n<attachment name="([^"\n]{1,200})" chars="(\d+)"(?: cut="(\d+)")?>\n[\s\S]*?\n<\/attachment>/y;
  const files = [];
  let at = start;
  while (at < text.length) {
    block.lastIndex = at;
    const found = block.exec(text);
    if (found === null) return { words: text, files: [] };
    files.push({ name: found[1].replace(/&(?:amp|lt|gt|quot|#x27);/g, (entity) => ENTITIES[entity]), chars: Number(found[2]) });
    at = block.lastIndex;
  }
  return { words: text.slice(0, start), files };
}

/** Whether *words* are only what the gateway writes for a message sent with no words. */
export function isStandIn(words) {
  return /^(?:\[\d+ pictures?\])?(?: ?\[attached: [^\]]*\])?$/.test(String(words).trim()) && String(words).trim() !== "";
}

let counter = 0;

/**
 * The composer's media. *host* gives: `doc`, `tray` (the strip), `warning` (the text-only line),
 * `fileInput`, `upload(file)` (resolves to the gateway's answer, throws an Error with a sentence),
 * `discard(uploadId)`, `notify(message)`, `changed()` (the strip changed), `openSettings()` and
 * `open(name, url, bytes)` (a picture at full size). Optional: `refuse(file, kind)`, a sentence when
 * that file cannot be added now (a text file while melo works), else nothing.
 */
export function setupMedia(host) {
  const { doc, tray, warning, fileInput } = host;
  let items = [];
  // A turn was refused for its pictures: the model of the settings is text only, whatever an item says.
  let refused = false;

  const node = (tag, className, text) => {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  };

  const pictures = () => items.filter((item) => item.kind === "picture");
  const files = () => items.filter((item) => item.kind === "file");
  // A tile removed by the person (or one whose upload failed) stays in a shelved list until that list is
  // next used; `live` is what counts.
  const live = (list) => list.filter((item) => !item.removed);

  function render() {
    const textOnly = pictures().some((item) => item.modelReads === false) || (refused && pictures().length > 0);
    warning.hidden = !textOnly;
    if (textOnly) {
      const link = node("button", "link-button", t("composer.textOnlyLink"));
      link.type = "button";
      link.addEventListener("click", () => host.openSettings());
      warning.replaceChildren(node("span", "", t("composer.textOnly")), link);
    }
    tray.hidden = items.length === 0;
    tray.replaceChildren(...items.map(tile));
  }

  function tile(item) {
    const box = node("div", `media-item media-${item.kind}`);
    box.dataset.state = item.state;
    box.dataset.mediaId = item.id;
    if (item.kind === "picture") {
      const frame = node("button", "media-frame");
      frame.type = "button";
      frame.setAttribute("aria-label", t("composer.openPicture", { name: item.name }));
      if (item.state === "ready") {
        const image = node("img", "media-thumb");
        image.alt = "";
        image.src = item.url;
        frame.append(image);
        frame.addEventListener("click", () => host.open(item.name, item.url, item.bytes));
      } else {
        frame.disabled = true;
        frame.append(node("span", "media-wait", t("composer.adding")));
      }
      box.append(frame);
    } else {
      const chip = node("span", "media-chip");
      chip.append(node("span", "media-chip-name", item.name), node("span", "media-chip-size", formatBytes(item.bytes)));
      chip.title = item.name;
      box.append(chip);
    }
    const remove = node("button", "media-remove", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", t("composer.removeMedia", { name: item.name }));
    remove.addEventListener("click", () => drop(item.id));
    box.append(remove);
    return box;
  }

  function drop(id) {
    const gone = items.find((item) => item.id === id);
    if (!gone) return;
    gone.removed = true;
    items = live(items);
    if (gone.uploadId) host.discard(gone.uploadId);
    render();
    host.changed();
  }

  async function addPicture(file) {
    if (pictures().length >= MAX_PICTURES) {
      host.notify(t("composer.tooManyPictures", { n: MAX_PICTURES }));
      return;
    }
    if (file.size > MAX_PICTURE_BYTES) {
      host.notify(t("composer.pictureTooLarge", { name: file.name, mb: MAX_PICTURE_BYTES >> 20 }));
      return;
    }
    counter += 1;
    const item = { id: `m${counter}`, kind: "picture", name: file.name || t("composer.pictureName"), state: "adding", modelReads: null };
    items.push(item);
    render();
    host.changed();
    try {
      const body = await host.upload(file);
      if (item.removed) {
        // Removed while it was on its way: the staged copy is not wanted.
        host.discard(body.id);
        return;
      }
      Object.assign(item, {
        state: "ready",
        uploadId: body.id,
        url: `/api/uploads/${encodeURIComponent(body.id)}`,
        bytes: body.size_bytes,
        width: body.width,
        height: body.height,
        modelReads: body.model_reads ?? null,
      });
      // The gateway's answer is the newest word on the model.
      refused = refused && body.model_reads === false;
    } catch (error) {
      item.removed = true;
      items = live(items);
      host.notify(t("composer.pictureFailed", { name: item.name, problem: error.message }));
    }
    render();
    host.changed();
  }

  /** Whether *file* fits beside the text files already in the strip; says why not. */
  function fitsAsText(file) {
    if (files().length >= MAX_FILES) {
      host.notify(t("composer.tooManyFiles", { n: MAX_FILES }));
      return false;
    }
    if (file.size > MAX_FILE_BYTES) {
      host.notify(t("composer.fileTooLarge", { name: file.name, kb: MAX_FILE_BYTES / 1024 }));
      return false;
    }
    if (files().reduce((sum, item) => sum + item.bytes, 0) + file.size > MAX_TOTAL_BYTES) {
      host.notify(t("composer.filesTooLarge", { kb: MAX_TOTAL_BYTES / 1024 }));
      return false;
    }
    return true;
  }

  async function addText(file) {
    if (!fitsAsText(file)) return;
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    } catch {
      host.notify(t("composer.notText", { name: file.name }));
      return;
    }
    if (text.includes("\0")) {
      host.notify(t("composer.notText", { name: file.name }));
      return;
    }
    // Other files were read meanwhile: the limits hold for the sum.
    if (!fitsAsText(file)) return;
    counter += 1;
    items.push({ id: `m${counter}`, kind: "file", name: file.name, state: "ready", text, bytes: file.size });
    render();
    host.changed();
  }

  /** Add what a drop, a paste or the picker gave. Pictures and text files wait in the strip; other files are told "not yet". */
  async function add(list) {
    const found = [...(list ?? [])];
    const work = [];
    for (const file of found) {
      const kind = kindOf(file);
      const why = host.refuse?.(file, kind);
      if (why) {
        host.notify(why);
        continue;
      }
      if (kind === "picture") work.push(addPicture(file));
      else if (kind === "text") work.push(addText(file));
      else host.notify(t("composer.notSupported", { name: file.name || t("composer.thatFile") }));
    }
    await Promise.all(work);
  }

  fileInput.addEventListener("change", () => {
    const chosen = [...(fileInput.files ?? [])];
    fileInput.value = "";
    add(chosen);
  });

  return {
    add,
    /** Open the system's file picker (the paperclip, `/attach`). */
    pick: () => fileInput.click(),
    remove: drop,
    render,
    items: () => live(items).map((item) => ({ ...item })),
    /** Whether anything waits in the strip. */
    has: () => items.length > 0,
    /** Whether a picture is still on its way to the gateway. */
    busy: () => items.some((item) => item.state !== "ready"),
    hasPictures: () => pictures().length > 0,
    /** Whether a text file waits in the strip. */
    hasFiles: () => files().length > 0,
    /** A turn was refused for its pictures: warn about the pictures in the strip from now on. */
    refusedPictures() {
      refused = true;
      render();
    },
    /** What a send carries: the picture ids and the text files. The strip is empty after. */
    take() {
      const taken = items.filter((item) => item.state === "ready");
      items = items.filter((item) => item.state !== "ready");
      render();
      host.changed();
      return taken;
    },
    /** Put back what a send did not take (a message that was not sent). */
    restore(back) {
      items = [...live(back), ...items];
      render();
      host.changed();
    },
    /**
     * Show *next* in the strip and return what it showed: the strip belongs to the chat that is open,
     * as its draft does. With `discard`, what it showed is forgotten (its chat was closed).
     */
    swap(next = [], { discard = false } = {}) {
      const outgoing = live(items);
      if (discard) {
        for (const item of outgoing) {
          item.removed = true;
          if (item.uploadId) host.discard(item.uploadId);
        }
      }
      items = live(next);
      render();
      host.changed();
      return discard ? [] : outgoing;
    },
  };
}

/** The `images` and `attachments` of a turn request for the *taken* items; a request has only the ones it carries. */
export function requestOf(taken) {
  const images = taken.filter((item) => item.kind === "picture").map((item) => item.uploadId);
  const attachments = taken
    .filter((item) => item.kind === "file")
    .map((item) => ({ name: item.name, text: item.text, chars: item.text.length }));
  return { ...(images.length ? { images } : {}), ...(attachments.length ? { attachments } : {}) };
}

/**
 * Strip items for pictures the gateway staged again (`unread_notes[].images` of a Stop): the same
 * items an upload makes, so they go back into the strip and are sent with the next message.
 */
export function pictureItems(staged) {
  return (staged ?? []).map((body) => {
    counter += 1;
    return {
      id: `m${counter}`,
      kind: "picture",
      name: body.name || t("composer.pictureName"),
      state: "ready",
      uploadId: body.id,
      url: `/api/uploads/${encodeURIComponent(body.id)}`,
      bytes: body.size_bytes,
      width: body.width,
      height: body.height,
      modelReads: body.model_reads ?? null,
    };
  });
}

/** The pictures of a turn (`turn.started`'s `images`) as shown: where each is served, and its size. */
export function turnPictures(sessionId, images) {
  return (images ?? []).map((image, index) => ({
    name: t("composer.pictureNumber", { n: index + 1 }),
    url: `/api/sessions/${encodeURIComponent(sessionId)}/blobs/${encodeURIComponent(image.sha256)}`,
    bytes: Number(image.size_bytes) || 0,
  }));
}
