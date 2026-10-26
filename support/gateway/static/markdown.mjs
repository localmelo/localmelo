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

function closingDelimiter(text, delimiter, start) {
  for (let position = start; position < text.length; position++) {
    if (text[position] === "\\") {
      position++;
      continue;
    }
    if (text[position] === "`") {
      const code = codeSpanAt(text, position);
      if (code) {
        position = code.end - 1;
        continue;
      }
    }
    if (text.startsWith(delimiter, position)) return position;
  }
  return -1;
}

/**
 * Find where an emphasis opened just before `start` closes. A run of the delimiter
 * character preceded by whitespace opens nested emphasis; a later run first closes those
 * nested openers, and only what is left of it can close this one.
 *
 * `tables` caches the tables built for this `text`: per delimiter character, the run ends
 * from delimiterRunEnds, so a long run of the character is crossed in one step; per
 * delimiter, the table from emphasisBlockers, so an opener without a closer returns at once
 * instead of scanning to the end of the paragraph.
 */
function closingEmphasis(text, delimiter, start, tables) {
  const character = delimiter[0];
  const runEnds = (tables.runEnds[character] ??= delimiterRunEnds(text, character));
  const blocked = (tables.blockers[delimiter] ??= emphasisBlockers(text, delimiter, runEnds));
  let nested = 0;
  for (let position = start; position < text.length; position++) {
    if (position > start && nested >= blocked[position]) return -1;
    if (text[position] === "\\") {
      position++;
      continue;
    }
    if (text[position] === "`") {
      const code = codeSpanAt(text, position);
      if (code) {
        position = code.end - 1;
        continue;
      }
    }
    if (text[position] !== character) continue;
    const runEnd = runEnds[position];
    if (position > start && !/\s/.test(text[position - 1])) {
      const closed = Math.min(nested, runEnd - position);
      nested -= closed;
      if (runEnd - position - closed >= delimiter.length) return position + closed;
    } else if (runEnd < text.length && !/\s/.test(text[runEnd])) {
      nested += runEnd - position;
    }
    position = runEnd - 1;
  }
  return -1;
}

/**
 * For each position, the fewest pending nested openers with which closingEmphasis, scanning
 * on from that position, finds no closer. It is built once per text and delimiter, from the
 * end, and follows closingEmphasis's steps for a position after `start`.
 */
function emphasisBlockers(text, delimiter, runEnds) {
  const character = delimiter[0];
  const blocked = new Array(text.length + 2).fill(0);
  for (let position = text.length - 1; position >= 0; position--) {
    let next = position + 1;
    if (text[position] === "\\") next = position + 2;
    else if (text[position] === "`") {
      const code = codeSpanAt(text, position);
      if (code) next = code.end;
    } else if (text[position] === character) {
      const runEnd = runEnds[position];
      const run = runEnd - position;
      const after = blocked[runEnd];
      if (!/\s/.test(text[position - 1])) {
        // A run that can close: it closes unless nested openers leave less than a delimiter.
        blocked[position] = after === 0 ? run - delimiter.length + 1 : run + after;
      } else if (runEnd < text.length && !/\s/.test(text[runEnd])) {
        blocked[position] = Math.max(0, after - run);
      } else blocked[position] = after;
      continue;
    }
    blocked[position] = blocked[next];
  }
  return blocked;
}

/** For each position, where the run of `character` from it ends (itself when it is another). */
function delimiterRunEnds(text, character) {
  const ends = new Array(text.length + 1);
  ends[text.length] = text.length;
  for (let position = text.length - 1; position >= 0; position--) {
    ends[position] = text[position] === character ? ends[position + 1] : position;
  }
  return ends;
}

const BARE_URL = /https?:\/\/[^\s<>"'`]+/iy;

/** Match a bare http(s) URL at `start`, without trailing punctuation (as GFM autolinks). */
function bareUrlAt(text, start) {
  if (start > 0 && !/[\s(*_~]/.test(text[start - 1])) return null;
  BARE_URL.lastIndex = start;
  const match = BARE_URL.exec(text);
  if (!match || /["'<>]/.test(text[start + match[0].length] || "")) return null;
  // Count unmatched closers once, so trimming a long run of them stays linear.
  const unmatched = { ")": 0, "]": 0 };
  for (const character of match[0]) {
    if (character === "(") unmatched[")"]--;
    else if (character === "[") unmatched["]"]--;
    else if (character === ")" || character === "]") unmatched[character]++;
  }
  let length = match[0].length;
  for (;;) {
    const last = match[0][length - 1];
    if (/[.,:;!?*_~]/.test(last)) length--;
    else if ((last === ")" || last === "]") && unmatched[last] > 0) {
      unmatched[last]--;
      length--;
    } else break;
  }
  const source = match[0].slice(0, length);
  const url = safeUrl(source);
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

function codeSpanAt(text, start) {
  const delimiter = text.slice(start).match(/^`+/)[0];
  let end = text.indexOf(delimiter, start + delimiter.length);
  while (end !== -1 && (text[end - 1] === "`" || text[end + delimiter.length] === "`")) {
    end = text.indexOf(delimiter, end + delimiter.length);
  }
  if (end === -1) return null;
  let code = text.slice(start + delimiter.length, end).replace(/\n/g, " ");
  if (/^ .* $/.test(code) && /[^ ]/.test(code)) code = code.slice(1, -1);
  return { code, end: end + delimiter.length };
}

function linkAt(text, start) {
  const labelEnd = closingDelimiter(text, "]", start + 1);
  if (labelEnd === -1 || text[labelEnd + 1] !== "(") return null;
  let depth = 1;
  let end = labelEnd + 2;
  for (; end < text.length; end++) {
    if (text[end] === "\\") {
      end++;
      continue;
    }
    if (text[end] === "(") depth++;
    if (text[end] === ")" && --depth === 0) break;
  }
  if (depth !== 0) return null;
  return {
    label: text.slice(start + 1, labelEnd),
    url: safeUrl(text.slice(labelEnd + 2, end)),
    end: end + 1,
  };
}

function inline(text, depth = 0, allowLinks = true) {
  if (depth > 20) return escapeHtml(text);
  const tables = { runEnds: {}, blockers: {} };
  let html = "";
  for (let index = 0; index < text.length; ) {
    const character = text[index];
    if (character === "\\" && /[!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~]/.test(text[index + 1] || "")) {
      html += escapeHtml(text[index + 1]);
      index += 2;
      continue;
    }
    if (character === "`") {
      const delimiter = text.slice(index).match(/^`+/)[0];
      const span = codeSpanAt(text, index);
      if (span) {
        html += "<code>" + escapeHtml(span.code) + "</code>";
        index = span.end;
        continue;
      }
      html += delimiter;
      index += delimiter.length;
      continue;
    }
    if (allowLinks && character === "[") {
      const link = linkAt(text, index);
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
      const delimiter = text[index + 1] === character ? character.repeat(2) : character;
      // An opener followed by whitespace never opens, so it needs no search for a closer;
      // otherwise each spaced "*" before a real closer would scan all the way to it.
      const opens = !/\s/.test(text[index + delimiter.length]);
      const end = opens ? closingEmphasis(text, delimiter, index + delimiter.length, tables) : -1;
      const intraword = character === "_" && /[\p{L}\p{N}]/u.test(text[index - 1] || "");
      if (opens && !intraword && end > index + delimiter.length && !/\s/.test(text[end - 1])) {
        const tag = delimiter.length === 2 ? "strong" : "em";
        html +=
          "<" +
          tag +
          ">" +
          inline(text.slice(index + delimiter.length, end), depth + 1, allowLinks) +
          "</" +
          tag +
          ">";
        index = end + delimiter.length;
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
      current += "\\|";
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
const listItem = (line) => /^( {0,3})([-+*]|\d{1,9}[.)])\s+(.+)$/.exec(line);
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
        inline((title[2] || "").replace(/[ \t]+#+[ \t]*$/, "").trim()) +
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
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
        const cells = tableCells(lines[index++]);
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
        const item = listItem(lines[index]);
        if (!item || item[1].length !== firstItem[1].length || /^\d/.test(item[2]) !== ordered)
          break;
        const indent = lines[index].indexOf(item[3], item[1].length + item[2].length);
        const content = [item[3]];
        index++;
        while (index < lines.length && lines[index].startsWith(" ".repeat(indent)))
          content.push(lines[index++].slice(indent));
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
