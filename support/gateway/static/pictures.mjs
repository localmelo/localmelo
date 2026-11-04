import { t } from "./i18n.mjs";
import { formatBytes } from "./turns.mjs";

/**
 * Pictures in a turn: a row of previews, and a light box for one at full size.
 *
 * Two rows use it. The pictures a turn made or named are the `turn.pictures` event's list
 * (`{path, size, mtime, state}`, paths relative to the chat's folder); each preview is an `<img>` of
 * `GET /api/sessions/{id}/pictures?path=…`, which serves a picture only from inside that folder. The
 * pictures melo took in the turn's steps (a page it looked at, another app's window) are the steps'
 * images (`takenPictures`), kept with the chat; each is `GET /api/sessions/{id}/blobs/<sha256>`
 * (`shotsNode`). At most `PICTURES_SHOWN` show; the rest wait behind `+N more`. A picture that is
 * gone says so in place of its image.
 */
export const PICTURES_SHOWN = 4;

/** The file name of a picture: the last part of its path. */
export function pictureName(picture) {
  return picture.path.slice(picture.path.lastIndexOf("/") + 1);
}

/** Where the picture is served; its time keeps a rewritten file from showing the old one. */
export function pictureUrl(sessionId, picture) {
  const version = Math.round(picture.mtime * 1000);
  return `/api/sessions/${encodeURIComponent(sessionId)}/pictures?path=${encodeURIComponent(picture.path)}&v=${version}`;
}

/** `{shown, more}`: the pictures to draw, and how many wait behind `+N more`. */
export function pictureRow(pictures, expanded = false) {
  if (expanded || pictures.length <= PICTURES_SHOWN) return { shown: pictures, more: 0 };
  return { shown: pictures.slice(0, PICTURES_SHOWN), more: pictures.length - PICTURES_SHOWN };
}

/** What stands in for a picture the gateway would not serve, by its HTTP status. */
export function failureNote(status) {
  if (status === 404) return t("pictures.gone");
  if (status === 413) return t("pictures.tooLarge");
  if (status === 415) return t("pictures.notPicture");
  if (status === 403) return t("pictures.outside");
  return t("pictures.cannotShow");
}

/** The size line of a preview: `84 KB`, with `updated` for a file the turn changed. */
export function pictureFacts(picture) {
  return [formatBytes(picture.size), picture.state === "modified" ? t("pictures.updated") : ""].filter(Boolean).join(" · ");
}

// The kinds of picture a step's image may be for the row: what the gateway serves inline.
const SHOT_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
const SHA256 = /^[0-9a-f]{64}$/;

/** Where a picture a step took is served: the chat's own copy. */
export function shotUrl(sessionId, sha256) {
  return `/api/sessions/${encodeURIComponent(sessionId)}/blobs/${encodeURIComponent(sha256)}`;
}

/**
 * The pictures the turn's steps took (`{sha256, mime_type, size, tool}`), each once, in the order
 * taken: a page melo looked at with `browser_screenshot`, another app's window, any step's image.
 * Only the chat's own copies count (a sha256 and a picture kind the gateway serves).
 */
export function takenPictures(turn) {
  const seen = new Set();
  const found = [];
  for (const item of Array.isArray(turn?.items) ? turn.items : []) {
    if (item?.kind !== "tool" || !Array.isArray(item.images)) continue;
    for (const image of item.images) {
      const sha256 = String(image?.sha256 ?? "");
      if (!SHA256.test(sha256) || seen.has(sha256) || !SHOT_TYPES.has(image.mime_type)) continue;
      seen.add(sha256);
      found.push({ sha256, mime_type: image.mime_type, size: Number(image.size_bytes) || 0, tool: String(item.name ?? "") });
    }
  }
  return found;
}

/**
 * The row of the pictures melo took in a turn (`takenPictures`), or null when it took none: the
 * same previews and light box as the turn's own pictures, named `Picture 1`, `Picture 2`… in the
 * order taken.
 */
export function shotsNode({ doc, shots, sessionId, expanded = false, open, toggle, status }) {
  const pictures = shots.map((shot, index) => ({
    path: t("pictures.shot", { n: index + 1 }), size: shot.size, mtime: 0, state: "", sha256: shot.sha256,
  }));
  const row = picturesNode({
    doc, pictures, sessionId, expanded, open, toggle, status,
    urlOf: (picture) => shotUrl(sessionId, picture.sha256),
    label: (count) => t(count === 1 ? "pictures.takenLabel.one" : "pictures.takenLabel.many"),
    failure: () => t("pictures.cannotShow"),
  });
  if (row) row.className = "pictures picture-shots";
  return row;
}

/**
 * The row for one turn, or null when it has no pictures. `open(picture, url)` shows one at full
 * size; `toggle()` flips `expanded`; `status(url)` resolves the HTTP status of a picture that
 * failed to load (default: a plain fetch). `urlOf(picture)`, `label(count)` and `failure(status)`
 * are where a picture is served, the row's name and what stands in for one that failed; by default
 * the folder's.
 */
export function picturesNode({
  doc, pictures, sessionId, expanded = false, open, toggle, status,
  urlOf = (picture) => pictureUrl(sessionId, picture),
  label = (count) => t(count === 1 ? "pictures.rowLabel.one" : "pictures.rowLabel.many"),
  failure = failureNote,
}) {
  if (pictures.length === 0) return null;
  const ask = status ?? (async (url) => (await fetch(url, { credentials: "same-origin" })).status);
  const node = (tag, className, text) => {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  };
  const row = node("div", "pictures");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", label(pictures.length));
  const grid = node("div", "picture-grid");
  row.append(grid);
  const { shown, more } = pictureRow(pictures, expanded);
  for (const picture of shown) {
    const url = urlOf(picture);
    const tile = node("button", "picture");
    tile.type = "button";
    const stage = node("span", "picture-stage");
    const image = node("img", "picture-thumb");
    image.alt = "";
    image.loading = "lazy";
    image.src = url;
    stage.append(image);
    const name = node("span", "picture-name", pictureName(picture));
    name.title = picture.path;
    const facts = node("span", "picture-facts", pictureFacts(picture));
    tile.append(stage, name, facts);
    tile.addEventListener("click", () => open(picture, url));
    image.addEventListener("error", async () => {
      let code = 0;
      try {
        code = await ask(url);
      } catch {
        // The gateway itself did not answer: the generic note below.
      }
      tile.disabled = true;
      stage.replaceChildren(node("span", "picture-missing", failure(code)));
      facts.textContent = "";
    });
    grid.append(tile);
  }
  if (pictures.length > PICTURES_SHOWN) {
    const label = more > 0 ? t("pictures.more", { n: more }) : t("pictures.fewer");
    const toggler = node("button", "link-button picture-more", label);
    toggler.type = "button";
    toggler.setAttribute("aria-expanded", more > 0 ? "false" : "true");
    if (more > 0) toggler.setAttribute("aria-label", t("pictures.moreLabel", { n: more }));
    toggler.addEventListener("click", toggle);
    row.append(toggler);
  }
  return row;
}

/**
 * The light box: fills a `<dialog>` with one image, its name and a close button. Escape, the close
 * button and a click outside the picture close it. Returns `open(picture, url)`.
 */
export function setupPictureBox({ dialog }) {
  const doc = dialog.ownerDocument;
  const node = (tag, className, text) => {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
  };
  const image = node("img");
  image.alt = "";
  const caption = node("p", "picture-box-caption");
  const close = node("button", "outline-button", t("common.close"));
  close.type = "button";
  const bar = node("div", "picture-box-bar");
  bar.append(caption, close);
  dialog.replaceChildren(image, bar);
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  return (picture, url) => {
    image.className = picture.path.toLowerCase().endsWith(".svg") ? "vector" : "";
    image.alt = pictureName(picture);
    image.src = url;
    caption.textContent = [pictureName(picture), pictureFacts(picture)].filter(Boolean).join(" · ");
    dialog.showModal();
  };
}
