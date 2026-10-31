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
 * `data-i18n-title` or `data-i18n-aria-label` that attribute, from the catalog;
 * a `{code}` in the words stands for the element's own `<code>`, which stays.
 */

export const LANGUAGES = ["en", "zh-Hans"];
const COOKIE = "localmelo_language";
const ATTRIBUTES = ["placeholder", "title", "aria-label"];
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

/** The page's language. */
export function language() {
  return current;
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
    const code = node.querySelector?.("code");
    if (code && text.includes("{code}")) {
      const [before, ...after] = text.split("{code}");
      node.replaceChildren(before, code, after.join("{code}"));
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
}

// Without its English catalog the page still runs, showing keys: a failed read must not stop every module.
catalogs.en = await readCatalog("en").catch(() => ({}));
if (globalThis.document?.documentElement?.dataset && typeof globalThis.fetch === "function") {
  await start(globalThis.document);
}
