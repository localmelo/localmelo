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
 *   while the server does not answer the page asks at the base interval, so the notice goes soon after it is back.
 */

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
 * question, a task held for the person or queued, a page the person holds.
 */
export function listBusy(list, health) {
  const rows = [list?.main, ...(list?.sessions ?? [])].filter(Boolean);
  if (rows.some((row) => row.busy || row.held || row.pending_approvals > 0 || row.waiting_tasks > 0 || row.waiting_in_side > 0)) return true;
  const status = health?.status;
  const running = Object.values(status?.running ?? {}).reduce((total, count) => total + (Number(count) || 0), 0);
  return running > 0 || (status?.waiting?.length ?? 0) > 0 || (status?.control?.length ?? 0) > 0
    || (status?.tasks?.queued ?? 0) > 0 || (status?.tasks?.held ?? 0) > 0;
}

// The fields of the health that tick by themselves: the seconds since the server started, since the newest run ended
// and since the model server was last asked. Two reads of an unchanged server differ in these and nothing else.
const CLOCKS = new Set(["uptime_s", "ended_s_ago", "checked_s_ago"]);

/** What a body says, as text: equal text is nothing new. The clocks are left out. */
export function fingerprint(body) {
  return JSON.stringify(body ?? null, (key, value) => (CLOCKS.has(key) ? undefined : value));
}
