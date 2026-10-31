/**
 * The scheduler's six levels as people read them: a P-code and what it means.
 *
 * One table for every place the console names a level (the panel's menu, the main chat's chips and
 * lists, the Scheduled tab, the task page, the chat's level menu), so a translation changes it once.
 * The gateway keeps the same table in `front_rules.py`; its words (Now, Today, …) are what a person may
 * type and what the API's `level_word` still answers, never what the console shows.
 */
export const LEVELS = [
  {code: 'P0', meaning: 'runs first; others wait'},
  {code: 'P1', meaning: 'today'},
  {code: 'P2', meaning: 'normal'},
  {code: 'P3', meaning: 'when there is time'},
  {code: 'P4', meaning: 'only when the computer is idle'},
  {code: 'P5', meaning: 'on hold until you start it'},
];

const NORMAL = 2;

/** The level behind a number: P2 (normal) for anything that is not one. */
export function levelOf(level) {
  return LEVELS[Number.isInteger(level) && level >= 0 && level < LEVELS.length ? level : NORMAL];
}

/** `P1`. */
export function levelCode(level) {
  return levelOf(level).code;
}

/** `today`. */
export function levelMeaning(level) {
  return levelOf(level).meaning;
}

/** `P1 · today`: for a menu option, where the meaning has room. */
export function levelLabel(level) {
  const {code, meaning} = levelOf(level);
  return `${code} · ${meaning}`;
}
