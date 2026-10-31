/**
 * The panel's Scheduled tab (scheduled-tab.mjs) in a DOM double: the check-in's row and switch, each chore's
 * title, time, next run and last result, its switch, Run now, Edit time and Delete (which asks first), a
 * chore that changed under the person (409), a side chat's own list, and the empty and failed states.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {CHECKIN, EMPTY, lateWords, rowLines, setupScheduledTab, whenWords} from '../../support/gateway/static/scheduled-tab.mjs';

class Element {
  constructor(tag) {
    Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, textContent: '',
      className: '', hidden: false, disabled: false, value: '', parent: null});
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  dispatch(type) { for (const fn of this.handlers[type] ?? []) fn({target: this}); }
  click() { if (!this.disabled) this.dispatch('click'); }
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
  findAll(test) { return [...(test(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test))]; }
  button(label) { return this.findAll((node) => node.tagName === 'BUTTON' && (node.text() === label || node.attributes['aria-label'] === label))[0] ?? null; }
}
const doc = {createElement: (tag) => new Element(tag)};

const settle = () => new Promise((resolve) => setImmediate(resolve));
// 2026-10-28 07:00 local time, a Wednesday.
const NOW = new Date(2026, 9, 28, 7, 0, 0).getTime();
const at = (day, hour, minute = 0) => new Date(2026, 9, day, hour, minute, 0).getTime() / 1000;
const chore = (id, extra = {}) => ({id, title: `Chore ${id}`, request: `Do ${id}`, schedule: {kind: 'weekdays', at: '08:30'}, when: 'every weekday at 8:30',
  level: 2, level_word: 'Normal', enabled: true, revision: 1, front_id: 'main', next_run: at(28, 8, 30), last_run: null, last: null, ...extra});
const checkin = (extra = {}) => ({on: true, every_min: 30, next_at: at(28, 7, 30), last_at: null, ...extra});

function harness({chores = [], checkinState = checkin(), chat = '', developer = false, fail = {}, scope = 'all'} = {}) {
  const h = {calls: [], chores, checkin: checkinState, chat, scope};
  const api = async (path, options = {}) => {
    const method = options.method ?? 'GET';
    h.calls.push([method, path, options.body]);
    const failure = fail[`${method} ${path}`];
    if (failure) throw Object.assign(new Error(failure.message ?? failure), failure.code ? {code: failure.code, status: 409} : {status: 500});
    if (method === 'GET') return {chores: h.chores, checkin: h.checkin, scope: h.scope};
    if (path === '/checkin') { h.checkin = {...h.checkin, on: options.body.on, next_at: options.body.on ? at(28, 7, 30) : null}; return {checkin: h.checkin}; }
    const id = path.split('/')[2].split('?')[0];
    const found = h.chores.find((item) => item.id === id);
    if (method === 'DELETE') return {deleted: id};
    if (path.endsWith('/run')) return {chore: {...found, last_run: NOW / 1000, last: {state: 'queued', line: '', task_id: 't1'}}, task: {}};
    const change = {...options.body};
    delete change.revision;
    return {chore: {...found, ...change, revision: found.revision + 1, when: change.schedule ? 'changed' : found.when}};
  };
  h.tab = setupScheduledTab({doc, api, chat: () => h.chat, developer: () => developer, clock: () => NOW});
  h.text = () => h.tab.root.text();
  h.rows = () => h.tab.root.findAll((node) => node.className === 'sched-row');
  h.button = (label) => h.tab.root.button(label);
  h.switches = () => h.tab.root.findAll((node) => node.attributes.role === 'switch');
  h.open = async (index = 0) => { h.rows()[index].findAll((node) => node.className === 'sched-head')[0].click(); await settle(); };
  return h;
}

test('the words: when something runs, and how the last run went', () => {
  const now = NOW / 1000;
  assert.match(whenWords(at(28, 8, 30), now), /^today at 8:30\s?AM$/i);
  assert.match(whenWords(at(29, 8, 30), now), /^tomorrow at 8:30\s?AM$/i);
  assert.match(whenWords(at(30, 21, 5), now), /^Friday at 9:05\s?PM$/i);
  assert.match(whenWords(at(35, 8, 30), now), /^Nov 4 at 8:30\s?AM$/, 'a week or more ahead is a date');
  assert.equal(whenWords(null, now), '');
  assert.match(rowLines(chore('a'), now)[0], /^Next today at 8:30\s?AM$/i);
  assert.deepEqual(rowLines(chore('a', {enabled: false}), now), ['Off']);
  assert.deepEqual(rowLines(chore('a', {last: {state: 'done', line: 'Three meetings today.'}}), now).slice(1), ['Last: Done · Three meetings today.']);
  assert.deepEqual(rowLines(chore('a', {last: {state: 'failed', line: 'Stopped before it finished'}}), now).slice(1), ['Last: Did not finish · Stopped before it finished']);
  assert.deepEqual(rowLines(chore('a', {last: {state: 'running', line: ''}}), now).slice(1), ['Last: Running now']);
});

test('with no chores the tab says how to add one, under the check-in with its switch and next time', async () => {
  const h = harness();
  assert.match(h.text(), /Reading the schedule/);
  await h.tab.show();
  assert.ok(h.text().includes('Check-in'));
  assert.ok(h.text().includes(CHECKIN));
  assert.match(h.text(), /Next check-in today at 7:30\s?AM/i);
  assert.ok(h.text().includes(EMPTY));
  assert.equal(EMPTY, "Nothing scheduled. Ask in the main chat: 'every weekday at 8:30, brief me on my calendar' or 'remind me tomorrow at 9 to call mom'.");
  assert.deepEqual(h.switches().map((item) => [item.attributes['aria-label'], item.attributes['aria-checked']]), [['Check-in', 'true']]);
});

test('the check-in switch turns it off and on, and says so in words', async () => {
  const h = harness();
  await h.tab.show();
  h.switches()[0].click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PUT', '/checkin', {on: false}]);
  assert.equal(h.switches()[0].attributes['aria-checked'], 'false');
  assert.ok(h.text().includes('Off') && !h.text().includes('Next check-in'));
  h.switches()[0].click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PUT', '/checkin', {on: true}]);
  assert.match(h.text(), /Next check-in/);
});

test('each chore shows its title, when it runs, its next run and its last result', async () => {
  const h = harness({chores: [chore('a', {title: 'Calendar brief', last: {state: 'done', line: 'Three meetings today.'}}),
    chore('b', {title: 'Check disk', when: 'every 2 hours', enabled: false, next_run: null})]});
  await h.tab.show();
  const [first, second] = h.rows();
  assert.match(first.text(), /^Calendar brief every weekday at 8:30 Next today at 8:30\s?AM Last: Done · Three meetings today\./i);
  assert.equal(second.text(), 'Check disk every 2 hours Off');
  assert.deepEqual(h.switches().map((item) => item.attributes['aria-checked']), ['true', 'true', 'false']);
  assert.equal(second.dataset.enabled, 'false');
  assert.ok(!h.text().includes(EMPTY));
});

test('a chore switch names the revision it read, and the row takes the answer', async () => {
  const h = harness({chores: [chore('a', {revision: 4})]});
  await h.tab.show();
  h.tab.root.findAll((node) => node.attributes['aria-label'] === 'Chore a on')[0].click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PATCH', '/chores/a', {revision: 4, enabled: false}]);
  assert.equal(h.tab.state.chores[0].revision, 5);
  assert.equal(h.tab.state.chores[0].enabled, false);
});

test('a chore that changed meanwhile reloads the list and says so', async () => {
  const h = harness({chores: [chore('a')], fail: {'PATCH /chores/a': {message: 'This chore changed.', code: 'stale_revision'}}});
  await h.tab.show();
  h.chores = [chore('a', {revision: 7, title: 'Renamed elsewhere'})];
  h.tab.root.findAll((node) => node.attributes['aria-label'] === 'Chore a on')[0].click();
  await settle();
  assert.ok(h.text().includes('This chore changed. The list is up to date again, try once more.'));
  assert.ok(h.text().includes('Renamed elsewhere'));
  assert.equal(h.tab.state.busy, false);
});

test('a row opens to what it runs, its priority and the three actions', async () => {
  const h = harness({chores: [chore('a', {level: 1, level_word: 'Today'})]});
  await h.tab.show();
  await h.open();
  assert.ok(h.text().includes('Do a'));
  assert.ok(h.text().includes('Priority: P1 · today'), 'the code and its meaning, not the API\'s level_word');
  for (const label of ['Run now: Chore a', 'Edit time: Chore a', 'Delete: Chore a']) assert.ok(h.button(label), label);
  assert.equal(h.tab.root.findAll((node) => node.attributes['aria-expanded'] === 'true').length, 1);
  await h.open();
  assert.ok(!h.text().includes('Priority'), 'it closes again');
  const dev = harness({chores: [chore('a', {level: 3, level_word: 'Whenever'})], developer: true});
  await dev.tab.show();
  await dev.open();
  assert.ok(dev.text().includes('Priority: P3 · when there is time'), 'developer mode reads the same');
});

test('Run now starts the chore and says where the result will be', async () => {
  const h = harness({chores: [chore('a')]});
  await h.tab.show();
  await h.open();
  h.button('Run now: Chore a').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['POST', '/chores/a/run', {}]);
  assert.ok(h.text().includes('Started: Chore a. Its result shows in Activity and in the main chat.'));
  assert.ok(h.text().includes('Last: Starting'));
});

test('Edit time changes the schedule: a time of day, or hours between runs', async () => {
  const h = harness({chores: [chore('a', {revision: 2})]});
  await h.tab.show();
  await h.open();
  h.button('Edit time: Chore a').click();
  const form = h.tab.root.findAll((node) => node.className === 'sched-form')[0];
  const [kind, field] = form.findAll((node) => node.className === 'sched-input');
  assert.deepEqual([kind.tagName, field.type, field.value], ['SELECT', 'time', '08:30']);
  assert.deepEqual(kind.children.map((option) => [option.value, option.selected]), [['daily', false], ['weekdays', true], ['hours', false]]);
  field.value = '07:15';
  field.dispatch('input');
  h.button('Save').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PATCH', '/chores/a', {revision: 2, schedule: {kind: 'weekdays', at: '07:15'}}]);
  assert.equal(h.tab.state.edit, null);

  h.button('Edit time: Chore a').click();
  const again = h.tab.root.findAll((node) => node.className === 'sched-input');
  again[0].value = 'hours';
  again[0].dispatch('change');
  const hours = h.tab.root.findAll((node) => node.className === 'sched-input')[1];
  assert.deepEqual([hours.type, hours.value], ['number', '4']);
  hours.value = '6';
  hours.dispatch('input');
  h.button('Save').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PATCH', '/chores/a', {revision: 3, schedule: {kind: 'hours', every: 6}}]);
});

test('Cancel leaves the time as it was, and a refusal stays in the tab with the form open', async () => {
  const h = harness({chores: [chore('a')], fail: {'PATCH /chores/a': 'every: 1 to 24 hours'}});
  await h.tab.show();
  await h.open();
  h.button('Edit time: Chore a').click();
  h.button('Cancel').click();
  assert.equal(h.tab.state.edit, null);
  assert.ok(!h.calls.some(([method]) => method === 'PATCH'));
  h.button('Edit time: Chore a').click();
  h.button('Save').click();
  await settle();
  assert.ok(h.text().includes('every: 1 to 24 hours'));
  assert.notEqual(h.tab.state.edit, null);
});

test('Delete asks first; Keep it leaves the chore, Delete removes it with its revision', async () => {
  const h = harness({chores: [chore('a', {revision: 3}), chore('b')]});
  await h.tab.show();
  await h.open();
  h.button('Delete: Chore a').click();
  assert.ok(h.text().includes('Delete this chore?'));
  h.button('Keep it').click();
  assert.ok(!h.calls.some(([method]) => method === 'DELETE'));
  h.button('Delete: Chore a').click();
  h.button('Delete').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/chores/a?revision=3', undefined]);
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['b']);
});

test("a side chat lists its own chores and says so; the main chat's list is unscoped", async () => {
  const main = harness({chores: [chore('a')]});
  await main.tab.show();
  assert.deepEqual(main.calls[0], ['GET', '/chores', undefined]);
  assert.ok(!main.text().includes('own chores'));
  const side = harness({chores: [chore('a')], chat: 'side one', scope: 'chat'});
  await side.tab.show();
  assert.deepEqual(side.calls[0], ['GET', '/chores?session_id=side%20one', undefined]);
  assert.ok(side.text().includes('This chat’s own chores. The main chat lists all of them.'));
});

test('an unreadable schedule says so; a quiet reload keeps what is shown and never interrupts an edit', async () => {
  const down = harness({fail: {'GET /chores': 'The session store is not available.'}});
  await down.tab.show();
  assert.ok(down.text().includes('The session store is not available.'));

  const h = harness({chores: [chore('a')]});
  await h.tab.show();
  h.chores = [chore('a', {title: 'Retitled'})];
  await h.tab.reload();
  assert.ok(h.text().includes('Retitled'));
  await h.open();
  h.button('Edit time: Retitled').click();
  const before = h.calls.length;
  assert.equal(h.tab.reload(), undefined);
  assert.equal(h.calls.length, before, 'an open edit is not read over');
});

// A reminder is a one-time chore: Friday 2026-10-30 09:00 (NOW is Wednesday the 28th).
const reminder = (id, extra = {}) => chore(id, {title: 'send the report', request: 'Reminder: send the report', schedule: {kind: 'once', at: '2026-10-30T09:00'},
  when: 'once, Fri Oct 30 at 9:00', level: 1, next_run: at(30, 9, 0), late_min: null, ...extra});

test('a one-time chore lists its time, and when it has run it is done, with how late', () => {
  const now = NOW / 1000;
  assert.match(rowLines(reminder('r'), now)[0], /^Next Friday at 9:00\s?AM$/i);
  // Done: switched off with no next time; the line says when it ran, not "Off".
  const ran = reminder('r', {enabled: false, next_run: null, last_run: at(30, 9, 1), last: {state: 'done', line: 'Time to send the report.'}});
  const lines = rowLines(ran, at(30, 12, 0));
  assert.match(lines[0], /^Reminded today at 9:01\s?AM$/i);
  assert.deepEqual(lines.slice(1), ['Last: Done · Time to send the report.']);
  // A reminder that ran after the computer slept says so.
  const late = rowLines({...ran, last_run: at(30, 11, 0), late_min: 120}, at(30, 12, 0));
  assert.deepEqual(late.slice(1), ['ran 2 h late', 'Last: Done · Time to send the report.']);
  assert.deepEqual([35, 60, 90, 120, 47 * 60, 48 * 60, 3 * 24 * 60].map(lateWords),
    ['ran 35 min late', 'ran 1 h late', 'ran 2 h late', 'ran 2 h late', 'ran 47 h late', 'ran 2 days late', 'ran 3 days late']);
  // A one-time chore the person switched off before it ran is just off.
  assert.deepEqual(rowLines(reminder('r', {enabled: false, next_run: null}), now), ['Off']);
});

test('a reminder row shows its words and time, and opens to its request, with Edit time as a date and time', async () => {
  const h = harness({chores: [reminder('r', {revision: 3})]});
  await h.tab.show();
  assert.ok(h.text().includes('send the report') && h.text().includes('once, Fri Oct 30 at 9:00'));
  await h.open();
  assert.ok(h.text().includes('Reminder: send the report'));
  h.button('Edit time: send the report').click();
  const form = h.tab.root.findAll((node) => node.className === 'sched-form')[0];
  const inputs = form.findAll((node) => node.className === 'sched-input');
  // No "how often" list for a one-time chore: only the date and time.
  assert.equal(inputs.length, 1);
  assert.deepEqual([inputs[0].tagName, inputs[0].type, inputs[0].value, inputs[0].attributes['aria-label']], ['INPUT', 'datetime-local', '2026-10-30T09:00', 'Date and time']);
  inputs[0].value = '2026-11-02T08:15';
  inputs[0].dispatch('input');
  h.button('Save').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PATCH', '/chores/r', {revision: 3, schedule: {kind: 'once', at: '2026-11-02T08:15'}}]);
  assert.equal(h.tab.state.edit, null);
});

test('a done reminder shows when it ran, and a refused switch-on stays in the tab', async () => {
  const done = reminder('r', {enabled: false, next_run: null, last_run: at(28, 6, 59), late_min: 35,
    last: {state: 'done', line: 'Time to send the report.'}});
  const h = harness({chores: [done], fail: {'PATCH /chores/r': 'That time has passed. Choose a new time.'}});
  await h.tab.show();
  assert.ok(h.text().includes('Reminded today at 6:59'));
  assert.ok(h.text().includes('ran 35 min late'));
  h.switches()[1].click();
  await settle();
  assert.ok(h.text().includes('That time has passed. Choose a new time.'));
});
