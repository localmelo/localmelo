/**
 * The console's words in the reader's language: English or Simplified Chinese.
 *
 * One catalog per language, `i18n/<language>.json`, which the server
 * (`support/gateway/i18n.py`) reads too. Keys are flat and grouped by screen
 * (`running.more`); a value is a string with `{name}` parameters, or, for a
 * count, `{one, other}` picked by the parameter `n`. A key the catalog lacks
 * reads in English.
 *
 * The page's language: the `[console] language` setting the server wrote on
 * `<html data-language>` when it names a language, else the first of the
 * reader's languages the console has (the macOS app's `localmeloLanguages`,
 * else `navigator.languages`). The page keeps it in a cookie, so the server
 * writes its own words in it too, and sets `<html lang>`. Elements with
 * `data-i18n="key"` get their text, and `data-i18n-placeholder`,
 * `data-i18n-title`, `data-i18n-aria-label` or `data-i18n-data-prompt` that
 * attribute, from the catalog; each `{code}` in the words stands for the
 * element's next own `<code>`, which stays. The macOS app is told the
 * language each time the page starts (`localmeloLanguage`), so its menus
 * follow without a restart.
 *
 * For another screen's code, the whole API is `t(key, params)` for words,
 * `saidText(said)` for words the gateway stored with a record,
 * `formatTime(date, kind)` for a time or a date, and `language()`; a new key
 * goes into both catalogs (see the Chinese style guide's interface terms for
 * the words to use).
 */

export const LANGUAGES = ["en", "zh-Hans"];
const COOKIE = "localmelo_language";
const ATTRIBUTES = ["placeholder", "title", "aria-label", "data-prompt"];
const TRADITIONAL = new Set(["hant", "tw", "hk", "mo"]);
const catalogs = {};
let current = "en";

/** The console language a BCP 47 tag asks for, or null. */
function languageOf(tag) {
  const parts = String(tag).trim().replace(/_/g, "-").toLowerCase().split("-");
  if (parts[0] === "en") return "en";
  if (parts[0] !== "zh") return null;
  if (parts.slice(1).includes("hans")) return "zh-Hans";
  return parts.slice(1).some((part) => TRADITIONAL.has(part)) ? null : "zh-Hans";
}

/** The language to show: `setting` when it names one, else the first preferred one the console has. */
export function pickLanguage(setting, preferred = []) {
  if (LANGUAGES.includes(setting)) return setting;
  for (const tag of preferred ?? []) {
    const found = languageOf(tag);
    if (found !== null) return found;
  }
  return "en";
}

/** `one` or `other`: English has both, Chinese only `other`. */
export function pluralForm(language, n) {
  return language === "en" && n === 1 ? "one" : "other";
}

/** `entry` with its parameters filled in; a count picks its plural form first. */
export function formatEntry(entry, params = {}, language = "en") {
  const text = typeof entry === "string" ? entry : entry[pluralForm(language, params.n)] ?? entry.other;
  return text.replace(/\{(\w+)\}/g, (whole, name) => (Object.hasOwn(params, name) ? String(params[name]) : whole));
}

/** The words of `key` in the page's language, falling back to English, then to the key. */
export function t(key, params = {}) {
  const own = catalogs[current]?.[key];
  if (own !== undefined) return formatEntry(own, params, current);
  const english = catalogs.en?.[key];
  return english === undefined ? key : formatEntry(english, params, "en");
}

/**
 * The words the gateway stored with a record (a card, an echo) as `said: {key, params}`, in the page's
 * language; a parameter may itself be a `said`, or a time of day `{clock: minutes after midnight}`, which
 * is written as this page writes a clock. Null when there is none, so the record's `line` reads.
 */
export function saidText(said) {
  if (!said || typeof said !== "object" || typeof said.key !== "string") return null;
  const params = {};
  for (const [name, value] of Object.entries(said.params ?? {})) {
    if (value && typeof value === "object" && typeof value.key === "string") params[name] = saidText(value);
    else if (value && typeof value === "object" && Number.isFinite(value.clock)) {
      params[name] = formatTime(new Date(2000, 0, 1, Math.floor(value.clock / 60), value.clock % 60), "clock");
    } else params[name] = value;
  }
  return t(said.key, params);
}

/** The page's language. */
export function language() {
  return current;
}

// What each kind of time shows. Chinese writes the hour on 12 hours with its day part ("上午 2:06"), as the
// page's English does; the English hour follows the computer (see `timeLocale`).
const TIME_KINDS = {
  clock: {hour: "numeric", minute: "2-digit"},
  weekday: {weekday: "long"},
  day: {month: "short", day: "numeric"},
  dayYear: {year: "numeric", month: "short", day: "numeric"},
  full: {year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit"},
};
const formats = new Map();

/**
 * The locale a time is written in: Simplified Chinese for the Chinese page; for the English page the
 * first English language the computer prefers (so a 24-hour computer keeps its clock), else plain `en`.
 */
export function timeLocale(lang, preferred = globalThis.localmeloLanguages ?? globalThis.navigator?.languages ?? []) {
  if (lang === "zh-Hans") return "zh-Hans";
  const own = [...preferred ?? []].find((tag) => /^en(-|_|$)/i.test(String(tag)));
  try {
    return own ? Intl.DateTimeFormat.supportedLocalesOf([String(own).replace(/_/g, "-")])[0] ?? "en" : "en";
  } catch {
    return "en";
  }
}

/**
 * A time or a date as the page's language writes it: `clock` ("2:06 AM", "上午 2:06"), `weekday`
 * ("Friday", "星期五"), `day` ("Oct 31", "10月31日"), `dayYear` ("Oct 31, 2026", "2026年10月31日") or `full`
 * (the day and the clock, with seconds). `date` is a Date; an invalid one reads as an empty string.
 */
export function formatTime(date, kind = "clock", {language: lang = current, preferred} = {}) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return "";
  const locale = timeLocale(lang, preferred);
  const id = `${locale}|${kind}`;
  if (!formats.has(id)) {
    const options = TIME_KINDS[kind] ?? TIME_KINDS.clock;
    formats.set(id, new Intl.DateTimeFormat(locale, locale.startsWith("zh") && options.hour ? {...options, hour12: true} : options));
  }
  const parts = formats.get(id).formatToParts(date);
  // Chinese puts a space between the day part and the figures that follow, as the style guide does for digits.
  // Newer ICU writes a narrow no-break space before "AM"; a plain space is what people copy and search.
  return parts.map((part, index) => (part.type === "dayPeriod" && parts[index + 1]?.type === "hour" ? `${part.value} ` : part.value)).join("").replace(/\u202f/g, " ");
}

async function readCatalog(name) {
  const url = new URL(`./i18n/${name}.json`, import.meta.url);
  if (url.protocol === "file:") {
    const {readFile} = await import("node:fs/promises");
    return JSON.parse(await readFile(url, "utf8"));
  }
  const response = await fetch(url, {credentials: "same-origin"});
  if (!response.ok) throw new Error(`No catalog for ${name} (${response.status})`);
  return response.json();
}

/** Switch to `next`, reading its catalog first; an unknown language is English. */
export async function useLanguage(next) {
  const wanted = LANGUAGES.includes(next) ? next : "en";
  if (!catalogs[wanted]) {
    try {
      catalogs[wanted] = await readCatalog(wanted);
    } catch {
      // English stays: every key has its English words.
      current = "en";
      return current;
    }
  }
  current = wanted;
  return current;
}

/** Fill the text and attributes of `root`'s `data-i18n` elements. */
export function translate(root) {
  for (const node of root.querySelectorAll?.("[data-i18n]") ?? []) {
    const text = t(node.dataset.i18n);
    const codes = node.querySelectorAll ? [...node.querySelectorAll("code")] : [node.querySelector?.("code")].filter(Boolean);
    if (codes.length && text.includes("{code}")) {
      // Each {code} takes the element's next <code>; one past the last stays as written.
      const parts = text.split("{code}");
      node.replaceChildren(...parts.flatMap((part, index) => (index < parts.length - 1 ? [part, codes[index] ?? "{code}"] : [part])));
    } else {
      node.textContent = text;
    }
  }
  for (const name of ATTRIBUTES) {
    for (const node of root.querySelectorAll?.(`[data-i18n-${name}]`) ?? []) {
      node.setAttribute(name, t(node.getAttribute(`data-i18n-${name}`)));
    }
  }
}

/** Choose the page's language, tell the server by cookie, and translate the page. */
async function start(doc) {
  const html = doc.documentElement;
  const preferred = globalThis.localmeloLanguages ?? globalThis.navigator?.languages ?? [];
  const chosen = await useLanguage(pickLanguage(html.dataset.language || "auto", preferred));
  html.lang = chosen;
  try {
    doc.cookie = `${COOKIE}=${chosen}; Path=/; SameSite=Strict; Max-Age=31536000`;
  } catch {
    // Without the cookie the server writes its own words in English.
  }
  translate(doc);
  delete html.dataset.i18nPending;
  // The macOS app writes its menus, pill and pet in the page's language too; elsewhere nobody listens.
  try {
    globalThis.webkit?.messageHandlers?.localmeloLanguage?.postMessage({language: chosen});
  } catch {
    // Without the app the page needs no one to follow it.
  }
}

// Without its English catalog the page still runs, showing keys: a failed read must not stop every module.
catalogs.en = await readCatalog("en").catch(() => ({}));
if (globalThis.document?.documentElement?.dataset && typeof globalThis.fetch === "function") {
  await start(globalThis.document);
}
