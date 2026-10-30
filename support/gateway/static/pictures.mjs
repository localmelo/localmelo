import { formatBytes } from "./turns.mjs";

/**
 * Pictures under a turn's answer: a row of previews, and a light box for one at full size.
 *
 * The pictures are the `turn.pictures` event's list (`{path, size, mtime, state}`, paths relative
 * to the chat's folder); each preview is an `<img>` of `GET /api/sessions/{id}/pictures?path=…`,
 * which serves a picture only from inside that folder. At most `PICTURES_SHOWN` show; the rest
 * wait behind `+N more`. A picture the folder no longer has says so in place of its image.
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
  if (status === 404) return "No longer in the folder";
  if (status === 413) return "Too large to preview";
  if (status === 415) return "Not a picture file";
  if (status === 403) return "Outside the chat's folder";
  return "Cannot be shown";
}

/** The size line of a preview: `84 KB`, with `updated` for a file the turn changed. */
export function pictureFacts(picture) {
  return [formatBytes(picture.size), picture.state === "modified" ? "updated" : ""].filter(Boolean).join(" · ");
}

/**
 * The row for one turn, or null when it has no pictures. `open(picture, url)` shows one at full
 * size; `toggle()` flips `expanded`; `status(url)` resolves the HTTP status of a picture that
 * failed to load (default: a plain fetch).
 */
export function picturesNode({ doc, pictures, sessionId, expanded = false, open, toggle, status }) {
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
  row.setAttribute("aria-label", pictures.length === 1 ? "Picture from this turn" : "Pictures from this turn");
  const grid = node("div", "picture-grid");
  row.append(grid);
  const { shown, more } = pictureRow(pictures, expanded);
  for (const picture of shown) {
    const url = pictureUrl(sessionId, picture);
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
      stage.replaceChildren(node("span", "picture-missing", failureNote(code)));
      facts.textContent = "";
    });
    grid.append(tile);
  }
  if (pictures.length > PICTURES_SHOWN) {
    const label = more > 0 ? `+${more} more` : "Show fewer";
    const toggler = node("button", "link-button picture-more", label);
    toggler.type = "button";
    toggler.setAttribute("aria-expanded", more > 0 ? "false" : "true");
    if (more > 0) toggler.setAttribute("aria-label", `Show ${more} more ${more === 1 ? "picture" : "pictures"}`);
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
  const close = node("button", "outline-button", "Close");
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
