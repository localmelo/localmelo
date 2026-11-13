/**
 * How often the page asks for its lists (the chats, and the status the header chip, the panel and the model chip show).
 *
 * The page asks every `LIST_BASE_MS` while anything runs, waits for the person or changes. While refreshes keep bringing
 * nothing new it asks less and less often, up to `LIST_CEILING_MS`, and goes back to the base interval at once when
 * something changes, when the person acts on the page or comes back to it, and while the server does not answer.
 * A page nobody can see asks nothing.
 *
 * Everything that depends on the refresh, and how late it may show at the ceiling:
 * - the sidebar's rows (a title, a "working" or "waiting" mark, a dot) when another place changed them: a terminal, a
 *   schedule, another window: 16 s, and nobody waits for them with the window idle;
 * - the header chip's count and the panel's lists of running work: work that starts without the person, 16 s; work
 *   the person starts here, and anything that runs, asks at the base interval;
 * - the model chip (the model server stopped or came back, a switch of models): 16 s; the server's own check of the
 *   model server runs only every 30 s;
 * - the offline notice: a server that stopped shows after 16 s at most (the chat's own live connection says so sooner);
 *   while the server does not answer the page asks at the base interval, so the notice goes soon after it is back;
 * - the header's mood (done or an error for `MOOD_WINDOW_S` after a run ends): the page asks at the base interval
 *   until the window is over, so the mood ends about when it always did;
 * - the day's lists (the sidebar's day groups and times, the Activity tab's day): a new day is a change, and the
 *   Activity tab's list is read again every 30 s and when its read failed, as before (`panel.keep`).
 *
 * What a reading says that the page derives from a clock is part of the comparison (the mood's window, the day);
 * what only a command or a dialog reads on demand (the uptime, the age of the model server's last check) is not: the
 * newest reading is kept as it came, drawn or not.
 */

import { showsEnd } from "./panel.mjs";

export const LIST_BASE_MS = 4000;
export const LIST_CEILING_MS = 16000;

/**
 * The time to the next refresh, from the last one's. A refresh that brought something new, failed, found anything
 * running or waiting for the person, or came after the person acted asks again at the base interval; otherwise the
 * interval doubles, up to the ceiling.
 */
export function nextListDelay(current, { changed = false, failed = false, busy = false, acted = false } = {}) {
  if (changed || failed || busy || acted) return LIST_BASE_MS;
  return Math.min(Math.max(current, LIST_BASE_MS) * 2, LIST_CEILING_MS);
}

/**
 * Whether anything runs or waits for the person in what a refresh read: a chat's turn, a task, a run, an approval, a
 * question, a task held for the person or queued, a page the person holds, or a run that ended so lately that the
 * header still shows how.
 */
export function listBusy(list, health) {
  const rows = [list?.main, ...(list?.sessions ?? [])].filter(Boolean);
  if (rows.some((row) => row.busy || row.held || row.pending_approvals > 0 || row.waiting_tasks > 0 || row.waiting_in_side > 0)) return true;
  const status = health?.status;
  const running = Object.values(status?.running ?? {}).reduce((total, count) => total + (Number(count) || 0), 0);
  return running > 0 || (status?.waiting?.length ?? 0) > 0 || (status?.control?.length ?? 0) > 0
    || (status?.tasks?.queued ?? 0) > 0 || (status?.tasks?.held ?? 0) > 0 || showsEnd(status?.recent);
}

// The fields of the health that tick by themselves and that the page shows only on demand: the seconds since the server
// started and since the model server was last asked. The age of the newest finished run ticks too, but the header
// derives its mood from it: the comparison keeps whether that mood is on (`showsEnd`) and leaves the age out.
const CLOCKS = new Set(["uptime_s", "checked_s_ago"]);

/** What a body says, as text: equal text is nothing new. The clocks are left out, the mood's window is kept. */
export function fingerprint(body) {
  return JSON.stringify(body ?? null, (key, value) => {
    if (CLOCKS.has(key)) return undefined;
    if (key === "recent" && value) {
      const kept = { ...value, shown: showsEnd(value) };
      delete kept.ended_s_ago;
      return kept;
    }
    return value;
  });
}
