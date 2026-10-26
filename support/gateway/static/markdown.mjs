/** A small, dependency-free Markdown renderer. Raw HTML is always text. */

export function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character],
  );
}

/** Return the normalized http(s) URL, or null for anything that must stay text. */
function safeUrl(destination) {
  const value =
    destination.startsWith("<") && destination.endsWith(">")
      ? destination.slice(1, -1)
      : destination;
  if (!/^https?:\/\//i.test(value) || /[\s\u0000-\u001f\u007f"'<>]/.test(value)) return null;
  if (/&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]*);/i.test(value)) return null;
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

const isSpace = (character) => character === undefined || /\s/.test(character);
const isWordCharacter = (character) => character !== undefined && /[\p{L}\p{N}]/u.test(character);
const isPunctuation = (character) => character !== undefined && /[\p{P}\p{S}]/u.test(character);
const CJK = /[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Hangul}]/u;
const isCjk = (character) => character !== undefined && CJK.test(character);
// The whole character before or at `position`, so a letter outside the BMP counts as one.
const characterBefore = (text, position) =>
  text.codePointAt(position - 2) > 0xffff
    ? String.fromCodePoint(text.codePointAt(position - 2))
    : text[position - 1];
const characterAt = (text, position) =>
  position < text.length ? String.fromCodePoint(text.codePointAt(position)) : undefined;

// inline() answers its searches from tables that are built for a text on first use, each in one
// pass: a forward search from every opener would make long answers quadratic.

/** For each position, where the run of `character` from it ends. */
function runEnds(text, character) {
  const ends = new Int32Array(text.length + 1);
  ends[text.length] = text.length;
  for (let position = text.length - 1; position >= 0; position--) {
    ends[position] = text[position] === character ? ends[position + 1] : position;
  }
  return ends;
}

/**
 * For each position, the end of the code span that the backticks from it open, or -1. A span
 * closes at the next run of exactly as many backticks, so one backward pass that remembers the
 * nearest run of each length answers every position.
 */
function codeSpanEnds(text, runs) {
  const ends = new Int32Array(text.length + 1).fill(-1);
  const nearest = new Map();
  for (let start = text.length - 1; start >= 0; start--) {
    if (text[start] !== "`" || text[start - 1] === "`") continue;
    const end = runs[start];
    for (let position = start; position < end; position++) {
      const close = nearest.get(end - position);
      if (close !== undefined) ends[position] = close + end - position;
    }
    nearest.set(end - start, start);
  }
  return ends;
}

/** Where the token at `position` ends: an escape, a code span or backtick run, or a character. */
function tokenEnd(text, position, tables) {
  if (text[position] === "\\") return Math.min(position + 2, text.length);
  if (text[position] !== "`") return position + 1;
  const runs = (tables["`"] ??= runEnds(text, "`"));
  const span = (tables.spans ??= codeSpanEnds(text, runs))[position];
  return span === -1 ? runs[position] : span;
}

/** The code span opened by the backticks at `start`; `code` is null when none closes. */
function codeSpanAt(text, start, tables) {
  const end = tokenEnd(text, start, tables);
  if (tables.spans[start] === -1) return { code: null, end };
  const length = tables["`"][start] - start;
  let code = text.slice(start + length, end - length).replace(/\n/g, " ");
  if (/^ .* $/.test(code) && /[^ ]/.test(code)) code = code.slice(1, -1);
  return { code, end };
}

/** For each position, the first "]" from it outside escapes and code spans, or -1. */
function bracketCloses(text, tables) {
  const closes = new Int32Array(text.length + 2).fill(-1);
  for (let position = text.length - 1; position >= 0; position--) {
    closes[position] = text[position] === "]" ? position : closes[tokenEnd(text, position, tables)];
  }
  return closes;
}

/** For each "(" outside an escape, the position of the ")" that closes it, or -1. */
function parenCloses(text) {
  const closes = new Int32Array(text.length).fill(-1);
  const open = [];
  for (let position = 0; position < text.length; position++) {
    if (text[position] === "\\") position++;
    else if (text[position] === "(") open.push(position);
    else if (text[position] === ")" && open.length) closes[open.pop()] = position;
  }
  return closes;
}

const OPENS = 1;
const CLOSES = 2;
const NESTS = 4;
const MAY_NEST = 8;
const LEFT_FLANKING = 16;

/**
 * Where emphasis opened with `character` closes, as `closes[length][start]` for content from
 * `start` (-1 when it does not close), and what each run of the character can do, as bits of
 * `kinds` decided by the characters around the whole run. A run can open before a non-space
 * and close after one; "_" does neither inside a word. A run that can close and is at least as
 * long as the delimiter closes it. A run that can only open starts nested emphasis, which the
 * search skips with its own closer, so `*a **b** c*` closes at the last "*". So does a run after
 * punctuation and before a letter, as CommonMark has it (`*x (*y*) z*`), when that nested
 * emphasis closes at a run that CommonMark would not let open. Otherwise, and always before a
 * Chinese, Japanese or Korean character, that run closes, since such text has no spaces:
 * `**注意：**请…`, `**注意：**Python … 改用**setuptools**`. Content that starts with the
 * character starts with nested emphasis too, and does not close when that does not. Filled from
 * the end, so every lookup is already known.
 */
function emphasisTables(text, character, tables) {
  const runs = (tables[character] ??= runEnds(text, character));
  const kinds = new Uint8Array(text.length + 1);
  for (let start = 0; start < text.length; start = Math.max(start + 1, runs[start])) {
    if (text[start] !== character) continue;
    const before = characterBefore(text, start);
    const after = characterAt(text, runs[start]);
    const word = character === "_";
    const opens = !isSpace(after) && !(word && isWordCharacter(before));
    const closes = !isSpace(before) && !(word && isWordCharacter(after));
    const mayNest = opens && closes && isPunctuation(before) && !isPunctuation(after);
    const left =
      !isSpace(after) && (!isPunctuation(after) || isSpace(before) || isPunctuation(before));
    kinds.fill(
      (opens ? OPENS : 0) |
        (closes ? CLOSES : 0) |
        (opens && !closes ? NESTS : 0) |
        (mayNest && !isCjk(after) ? MAY_NEST : 0) |
        (left ? LEFT_FLANKING : 0),
      start,
      runs[start],
    );
  }
  const table = () => new Int32Array(text.length + 2).fill(-1);
  const scan = [null, table(), table()];
  const closes = [null, table(), table()];
  // Where a search continues past nested emphasis opened at `position`, or -1. With `strict`,
  // only when it closes at a run that cannot open by CommonMark's rule.
  const pastNested = (position, strict = false) => {
    const length = runs[position] - position >= 2 ? 2 : 1;
    const close = closes[length][position + length];
    if (close === -1 || (strict && kinds[close] & LEFT_FLANKING)) return -1;
    return close + length;
  };
  for (let position = text.length - 1; position >= 0; position--) {
    const end = runs[position];
    const kind = kinds[position];
    if (text[position] === character) {
      const nested =
        kind & NESTS ? pastNested(position) : kind & MAY_NEST ? pastNested(position, true) : -1;
      for (const length of [1, 2]) {
        if (nested !== -1) scan[length][position] = scan[length][nested];
        else if (kind & CLOSES && end - position >= length) scan[length][position] = position;
        else scan[length][position] = scan[length][end];
      }
    } else {
      const next = tokenEnd(text, position, tables);
      for (const length of [1, 2]) scan[length][position] = scan[length][next];
    }
    const start = text[position] === character && kind & OPENS ? pastNested(position) : position;
    for (const length of [1, 2]) closes[length][position] = start === -1 ? -1 : scan[length][start];
  }
  return { kinds, closes };
}

function linkAt(text, start, tables) {
  const labelEnd = (tables.brackets ??= bracketCloses(text, tables))[start + 1];
  if (labelEnd === -1 || text[labelEnd + 1] !== "(") return null;
  const end = (tables.parens ??= parenCloses(text))[labelEnd + 1];
  if (end === -1) return null;
  return {
    label: text.slice(start + 1, labelEnd),
    url: safeUrl(text.slice(labelEnd + 2, end)),
    end: end + 1,
  };
}

// Full-width punctuation (U+3000-303F, U+FF00-FFEF) ends a URL, as in "详见 https://x.y/。".
const BARE_URL = /https?:\/\/[^\s<>"'`\u3000-\u303f\uff00-\uffef]{1,2112}/iy;
const MAX_BARE_URL = 2048;

/** Match a bare http(s) URL at `start`, without trailing punctuation (as GFM autolinks). */
function bareUrlAt(text, start) {
  const before = text[start - 1];
  if (start > 0 && !/[\s(*_~\u3000-\u303f\uff00-\uffef]/.test(before) && !isCjk(before))
    return null;
  BARE_URL.lastIndex = start;
  const match = BARE_URL.exec(text);
  // A quote or angle bracket after the URL keeps it text, and so does a URL that runs past
  // the pattern's cap, which bounds the work at each start.
  const next = text[start + (match ? match[0].length : 0)];
  if (!match || (next !== undefined && /[^\s`\u3000-\u303f\uff00-\uffef]/.test(next))) return null;
  // Text with no space before the URL has none after it either ("请访问https://x.y/docs查看").
  let raw = match[0];
  if (start > 0 && isCjk(before) && raw.search(CJK) !== -1) raw = raw.slice(0, raw.search(CJK));
  // Count unmatched closers once, so trimming a long run of them stays linear.
  const unmatched = { ")": 0, "]": 0 };
  for (const character of raw) {
    if (character === "(") unmatched[")"]--;
    else if (character === "[") unmatched["]"]--;
    else if (character === ")" || character === "]") unmatched[character]++;
  }
  let length = raw.length;
  for (;;) {
    const last = raw[length - 1];
    if (/[.,:;!?*_~]/.test(last)) length--;
    else if ((last === ")" || last === "]") && unmatched[last] > 0) {
      unmatched[last]--;
      length--;
    } else break;
  }
  const source = raw.slice(0, length);
  const url = source.length <= MAX_BARE_URL && safeUrl(source);
  return url ? { source, url, end: start + source.length } : null;
}

function anchor(url, labelHtml) {
  return (
    '<a href="' +
    escapeHtml(url) +
    '" target="_blank" rel="noopener noreferrer">' +
    labelHtml +
    "</a>"
  );
}

function inline(text, depth = 0, allowLinks = true) {
  if (depth > 20) return escapeHtml(text);
  const tables = {};
  let html = "";
  for (let index = 0; index < text.length; ) {
    const character = text[index];
    if (character === "\\" && /[!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~]/.test(text[index + 1] || "")) {
      html += escapeHtml(text[index + 1]);
      index += 2;
      continue;
    }
    if (character === "`") {
      const span = codeSpanAt(text, index, tables);
      html +=
        span.code === null
          ? text.slice(index, span.end)
          : "<code>" + escapeHtml(span.code) + "</code>";
      index = span.end;
      continue;
    }
    if (allowLinks && character === "[") {
      const link = linkAt(text, index, tables);
      if (link) {
        html += link.url
          ? anchor(link.url, inline(link.label, depth + 1, false))
          : escapeHtml(text.slice(index, link.end));
        index = link.end;
        continue;
      }
    }
    if (allowLinks && (character === "h" || character === "H")) {
      const bare = bareUrlAt(text, index);
      if (bare) {
        html += anchor(bare.url, escapeHtml(bare.source));
        index = bare.end;
        continue;
      }
    }
    if (character === "*" || character === "_") {
      const length = text[index + 1] === character ? 2 : 1;
      const start = index + length;
      const emphasis = (tables["emphasis" + character] ??= emphasisTables(text, character, tables));
      const end = emphasis.kinds[index] & OPENS ? emphasis.closes[length][start] : -1;
      if (end > start) {
        const tag = length === 2 ? "strong" : "em";
        html +=
          "<" +
          tag +
          ">" +
          inline(text.slice(start, end), depth + 1, allowLinks) +
          "</" +
          tag +
          ">";
        index = end + length;
        continue;
      }
    }
    html += character === "\n" ? "<br>\n" : escapeHtml(character);
    index++;
  }
  return html;
}

function tableCells(line) {
  let source = line.trim();
  if (source.startsWith("|")) source = source.slice(1);
  if (source.endsWith("|") && !source.endsWith("\\|")) source = source.slice(0, -1);
  const cells = [];
  let current = "";
  let codeDelimiter = "";
  for (let index = 0; index < source.length; index++) {
    const character = source[index];
    if (character === "\\" && source[index + 1] === "|") {
      // GFM drops the backslash even inside code, where inline() would show it.
      current += codeDelimiter ? "|" : "\\|";
      index++;
      continue;
    }
    if (character === "`") {
      const delimiter = source.slice(index).match(/^`+/)[0];
      if (!codeDelimiter) codeDelimiter = delimiter;
      else if (codeDelimiter === delimiter) codeDelimiter = "";
      current += delimiter;
      index += delimiter.length - 1;
      continue;
    }
    if (character === "|" && !codeDelimiter) {
      cells.push(current.trim());
      current = "";
    } else current += character;
  }
  cells.push(current.trim());
  return cells;
}

function tableAlignment(line) {
  if (!line || !line.includes("|")) return null;
  const cells = tableCells(line);
  return cells.every((cell) => /^:?-{3,}:?$/.test(cell))
    ? cells.map((cell) =>
        cell.endsWith(":") ? (cell.startsWith(":") ? "center" : "right") : "left",
      )
    : null;
}

const fence = (line) => /^ {0,3}(`{3,}|~{3,})([^\n]*)$/.exec(line);
const heading = (line) => /^ {0,3}(#{1,6})(?:[ \t]+(.*)|$)/.exec(line);
const listItem = (line) => /^( {0,3})([-+*]|\d{1,9}[.)])\s+(.+)$/s.exec(line);
const rule = (line) => /^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(line);

function blocks(lines, depth = 0) {
  if (depth > 20) return "<p>" + escapeHtml(lines.join("\n")) + "</p>";
  let html = "";
  for (let index = 0; index < lines.length; ) {
    const line = lines[index];
    if (!line.trim()) {
      index++;
      continue;
    }
    const opening = fence(line);
    if (opening) {
      const delimiter = opening[1];
      const language = opening[2].trim().split(/\s+/)[0];
      const close = new RegExp("^ {0,3}" + delimiter[0] + "{" + delimiter.length + ",}\\s*$");
      const content = [];
      index++;
      while (index < lines.length && !close.test(lines[index])) content.push(lines[index++]);
      const closed = index < lines.length;
      if (closed) index++;
      const className = /^[\w.+-]+$/.test(language)
        ? ' class="language-' + escapeHtml(language) + '"'
        : "";
      html +=
        "<pre><code" +
        className +
        ">" +
        escapeHtml(content.join("\n") + (closed && content.length ? "\n" : "")) +
        "</code></pre>";
      continue;
    }
    const title = heading(line);
    if (title) {
      html +=
        "<h" +
        title[1].length +
        ">" +
        inline((title[2] || "").replace(/[ \t]#+[ \t]*$/, "").trim()) +
        "</h" +
        title[1].length +
        ">";
      index++;
      continue;
    }
    if (rule(line)) {
      html += "<hr>";
      index++;
      continue;
    }
    if (/^ {0,3}>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^ {0,3}>/.test(lines[index]))
        quote.push(lines[index++].replace(/^ {0,3}> ?/, ""));
      html += "<blockquote>" + blocks(quote, depth + 1) + "</blockquote>";
      continue;
    }
    const alignment = tableAlignment(lines[index + 1]);
    const headers = tableCells(line);
    if (alignment && line.includes("|") && headers.length === alignment.length) {
      html +=
        '<div class="table-scroll"><table><thead><tr>' +
        headers
          .map(
            (cell, column) =>
              '<th class="align-' + alignment[column] + '">' + inline(cell) + "</th>",
          )
          .join("") +
        "</tr></thead><tbody>";
      index += 2;
      // Cap the empty cells that pad short rows, as markdown-it does: a wide header over many
      // short rows would otherwise grow a small answer into hundreds of megabytes of HTML.
      let padding = 0;
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
        const cells = tableCells(lines[index]);
        padding += Math.max(0, alignment.length - cells.length);
        if (padding > 65536) break;
        index++;
        html +=
          "<tr>" +
          alignment
            .map(
              (align, column) =>
                '<td class="align-' + align + '">' + inline(cells[column] || "") + "</td>",
            )
            .join("") +
          "</tr>";
      }
      html += "</tbody></table></div>";
      continue;
    }
    const firstItem = listItem(line);
    if (firstItem) {
      const ordered = /^\d/.test(firstItem[2]);
      const tag = ordered ? "ol" : "ul";
      const start = ordered ? Number.parseInt(firstItem[2], 10) : 1;
      html += "<" + tag + (ordered && start !== 1 ? ' start="' + start + '"' : "") + ">";
      while (index < lines.length) {
        // Blank lines between items keep the list going.
        let next = index;
        while (next < lines.length && !lines[next].trim()) next++;
        const item = next < lines.length && !rule(lines[next]) && listItem(lines[next]);
        if (!item || item[1].length !== firstItem[1].length || /^\d/.test(item[2]) !== ordered)
          break;
        index = next;
        const indent = lines[index].indexOf(item[3], item[1].length + item[2].length);
        const prefix = " ".repeat(indent);
        const content = [item[3]];
        index++;
        // So do blank lines inside an item, up to its last line indented under the marker.
        for (next = index; next < lines.length; next++) {
          if (!lines[next].trim()) continue;
          if (!lines[next].startsWith(prefix)) break;
          while (index <= next) content.push(lines[index++].slice(indent));
        }
        html += "<li>" + blocks(content, depth + 1) + "</li>";
      }
      html += "</" + tag + ">";
      continue;
    }
    const paragraph = [line];
    index++;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !fence(lines[index]) &&
      !heading(lines[index]) &&
      !rule(lines[index]) &&
      !listItem(lines[index]) &&
      !/^ {0,3}>/.test(lines[index]) &&
      !tableAlignment(lines[index + 1])
    )
      paragraph.push(lines[index++]);
    html += "<p>" + inline(paragraph.join("\n")) + "</p>";
  }
  return html;
}

export function renderMarkdown(text) {
  return blocks(
    String(text ?? "")
      .replace(/\r\n?/g, "\n")
      .split("\n"),
  );
}
