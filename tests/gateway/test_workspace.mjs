/**
 * The workspace page's turns: the reducer (`turns.mjs`) on its own, and the
 * page's `/api` flows in a DOM double. Browser layout is verified separately
 * by the screenshots; nothing here measures it.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import * as turns from '../../support/gateway/static/turns.mjs';
import {renderMarkdown} from '../../support/gateway/static/markdown.mjs';
import {workerHistory} from '../../support/gateway/static/worker-history.mjs';
import {setupTasks} from '../../support/gateway/static/tasks.mjs';
import {setupStatusRow} from '../../support/gateway/static/status-row.mjs';

const {
  applyEvent, activity, approvalTag, cleanOutput, deniedCount, failedCount,
  formatBytes, formatDuration, formatPast, formatTokens, liveFacts, newTurn,
  newView, reflectionLine, runElapsed, splitThinking, statusText, toolElapsed,
  toolState, workSummary,
} = turns;

// ── Turns: one frame at a time ──

function started(overrides = {}) {
  const view = newView('a');
  applyEvent(view, {
    id: 1, turn: 1, ts: 100, type: 'turn.started',
    data: {turn: 1, turn_id: 'turn-1', user_text: 'Do the thing', client_request_id: 'key-1'},
  });
  const turn = view.turns[0];
  Object.assign(turn, overrides);
  return {view, turn};
}

/** The events of one finished turn, as the store gives them. */
function fullTurn() {
  const {view} = started();
  let id = 1;
  const push = (type, data, extra = {}) => {
    applyEvent(view, {id: ++id, turn: 1, ts: 100 + id, type, data, ...extra});
  };
  push('run.started', {query: 'Do the thing', max_steps: 30, workspace_root: '/tmp', approval: true}, {run_id: 'run-1'});
  push('step.started', {step: 1, attempt: 1});
  return {view, push, turn: view.turns[0]};
}

test('a frame at or below the cursor is not applied twice', () => {
  const {view, turn} = started();
  const frame = {id: 7, turn: 1, ts: 110, type: 'turn.finished', data: {turn: 1, turn_id: 'turn-1', status: 'completed', cause: 'run', error: ''}};
  applyEvent(view, frame);
  assert.equal(turn.status, 'completed');
  applyEvent(view, {...frame, data: {...frame.data, status: 'failed'}});
  assert.equal(view.lastEventId, 7);
  assert.equal(turn.status, 'completed', 'The same id is the same event');
  applyEvent(view, {...frame, data: {...frame.data, status: 'failed'}}, {ignoreCursor: true});
  assert.equal(turn.status, 'failed', 'A stored turn loaded for history ignores the cursor');
});

test('a turn starts with its user text, and a finished skeleton keeps its status', () => {
  const {view, turn} = started({status: 'completed', finishedAt: 150});
  assert.equal(turn.userText, 'Do the thing');
  assert.equal(turn.clientRequestId, 'key-1');
  applyEvent(view, {
    id: 2, turn: 1, ts: 120, type: 'turn.started',
    data: {turn: 1, turn_id: 'turn-1', user_text: 'Do the thing', client_request_id: 'key-1'},
  });
  assert.equal(turn.status, 'completed', 'A replayed start does not reopen a finished turn');
  assert.equal(turn.startedAt, 100, 'A replayed start does not move the clock either');
});

test('a reply keeps its thinking, its text and the calls that never started', () => {
  const {view, turn, push} = fullTurn();
  push('assistant.message', {
    step: 1,
    content: '<think>let me look</think>I will read the file.',
    thinking: '',
    tool_calls: [{id: 'call-1', name: 'file_read', arguments: {path: 'a.py'}, preview: 'a.py'}],
    usage: {prompt_tokens: 120, completion_tokens: 30},
    duration_s: 1.8,
  });
  assert.deepEqual(turn.items.map((item) => item.kind), ['thinking', 'assistant_text']);
  assert.equal(turn.items[0].text, 'let me look');
  assert.equal(turn.items[0].durationS, 1.8);
  assert.equal(turn.items[1].text, 'I will read the file.');
  assert.equal(turn.outputTokens, 30);
  assert.equal(view.contextTokens, 120);
  assert.equal(turn.unstartedCalls.get('call-1').name, 'file_read');

  push('step.started', {step: 2, attempt: 1});
  const last = turn.items[turn.items.length - 1];
  assert.equal(last.kind, 'tool');
  assert.equal(last.status, 'not_run', 'A listed call with no tool.started reads not run');
  assert.equal(last.preview, 'a.py');
  assert.equal(turn.unstartedCalls.size, 0);
});

test('a held reply is flushed before the next step, or by the run end', () => {
  const {turn, push} = fullTurn();
  push('assistant.message', {step: 1, content: 'First, the plan.', tool_calls: []});
  assert.equal(turn.heldText, 'First, the plan.');
  assert.equal(turn.items.filter((item) => item.kind === 'assistant_text').length, 0);
  push('step.started', {step: 2, attempt: 1});
  assert.deepEqual(turn.items.map((item) => item.text), ['First, the plan.']);
  push('assistant.message', {step: 2, content: 'The same words.', tool_calls: []});
  push('run.finished', {result: {status: 'completed', result: 'The same words.', duration_s: 9, n_tool: 0, steps: 2}});
  assert.equal(turn.items.filter((item) => item.kind === 'assistant_text').length, 1, 'The answer is not printed twice');
  assert.equal(turn.answer, 'The same words.');
});

test('a tool call runs, asks for approval and finishes', () => {
  const {view, turn, push} = fullTurn();
  push('assistant.message', {step: 1, content: '', tool_calls: [{id: 'call-1', name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls'}]});
  push('tool.started', {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls'});
  let card = turn.items.find((item) => item.kind === 'tool');
  assert.equal(card.status, 'running');
  assert.equal(toolState(card, {now: 110}).text, 'Running · 0:05 of 10:00', 'A shell call states its limit');

  push('approval.requested', {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls', workspace_root: '/tmp'});
  assert.equal(card.status, 'waiting_approval');
  assert.equal(card.pendingRequestId, 'req-1');
  assert.equal(view.pending.size, 1);
  assert.equal(toolState(card).text, 'Needs approval');

  push('approval.resolved', {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', outcome: 'allowed', origin: 'user', reason: '', remember: true});
  assert.equal(card.status, 'running');
  assert.equal(view.pending.size, 0);
  assert.deepEqual(view.grants, ['shell_exec']);
  assert.ok(view.grantedAt.shell_exec > 0);

  push('tool.finished', {step: 1, call_id: 'call-1', name: 'shell_exec', status: 'success', error_category: 'none', output: 'a.py\n', output_chars: 5, error: '', duration_s: 4, images: [], approval: {request_id: 'req-1', allowed: true, reason: '', remember: true, origin: 'user'}, exit_code: 2});
  assert.equal(card.status, 'success');
  assert.equal(card.output, 'a.py\n');
  assert.deepEqual(toolState(card), {state: 'success', mark: '[ok]', text: '4.0s · exit 2 · allowed for the session'});
});

test('an approval the turn withdrew leaves the call not run', () => {
  const {turn, push} = fullTurn();
  push('assistant.message', {step: 1, content: '', tool_calls: [{id: 'call-1', name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls'}]});
  push('tool.started', {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls'});
  push('approval.requested', {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', arguments: {command: 'ls'}, preview: 'ls', workspace_root: '/tmp'});
  push('approval.resolved', {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', outcome: 'withdrawn', origin: 'withdrawn', reason: '', remember: false});
  const card = turn.items.find((item) => item.kind === 'tool');
  assert.equal(card.status, 'not_run', 'A withdrawn approval never ran the call');
  assert.equal(toolState(card).text, 'not run');
  push('turn.finished', {turn: 1, turn_id: 'turn-1', status: 'cancelled', error: '', cause: 'user_stop'});
  assert.equal(card.status, 'not_run', 'The turn\'s end does not turn it into an interruption');
});

test('a denied approval blocks the call and says why', () => {
  const {view, turn, push} = fullTurn();
  push('tool.started', {step: 1, call_id: 'call-1', name: 'file_write', arguments: {path: 'a'}, preview: 'a'});
  push('approval.requested', {request_id: 'req-1', call_id: 'call-1', tool_name: 'file_write', arguments: {path: 'a'}, preview: 'a', workspace_root: '/tmp'});
  push('approval.resolved', {request_id: 'req-1', call_id: 'call-1', tool_name: 'file_write', outcome: 'denied', origin: 'user', reason: 'wrong file\npick another', remember: false});
  const card = turn.items.find((item) => item.kind === 'tool');
  assert.equal(card.status, 'blocked');
  assert.deepEqual(view.grants, []);
  push('tool.finished', {step: 1, call_id: 'call-1', name: 'file_write', status: 'blocked', error_category: 'approval_denied', output: '', output_chars: 0, error: '', duration_s: 0, images: [], approval: {request_id: 'req-1', allowed: false, reason: 'wrong file pick another', remember: false, origin: 'user'}, exit_code: null});
  assert.deepEqual(toolState(card), {state: 'blocked', mark: '[-]', text: 'denied: wrong file pick another'});
  assert.equal(deniedCount(turn), 1);
});

test('a retried request counts down and replaces the notice of its step', () => {
  const {turn, push} = fullTurn();
  push('notice', {kind: 'retry', message: 'Model request failed; retrying in 2 s: boom', step: 1, attempt: 1, detail: {request_attempt: 1, delay_s: 2, status_code: 500}});
  assert.ok(turn.retryUntil > 100, 'The retry ends a delay after its event');
  const retryAt = turn.retryUntil;
  push('notice', {kind: 'retry', message: 'Model request failed; retrying in 4 s: boom', step: 1, attempt: 1, detail: {delay_s: 4}});
  const retries = turn.items.filter((item) => item.kind === 'notice' && item.noticeKind === 'retry');
  assert.equal(retries.length, 1);
  assert.match(retries[0].message, /retrying in 4 s/);
  assert.ok(turn.retryUntil > retryAt);
  push('notice', {kind: 'budget', message: 'The attempt used its step budget.', step: 1, attempt: 1, detail: {}});
  assert.equal(turn.items.filter((item) => item.kind === 'notice').length, 2);
});

test('the live row words the run as the terminal does', () => {
  const {turn, push} = fullTurn();
  push('run.progress', {level: 1, state: 'running', step: 4, attempt: 2, phase: 'Verify', cycle: 1, tests: {passed: 3, failed: 1, total: 4}, items: null, past: null, basis: {match: 'new'}});
  assert.deepEqual(liveFacts(turn), ['step 4', 'attempt 2', 'Verify (cycle 1)', 'tests 3/4']);
  assert.equal(activity(turn, {now: 200}), 'Thinking');
  push('reflection.started', {attempt: 1, failure_type: 'stuck'});
  assert.equal(activity(turn, {now: 200}), 'Reflecting');
  assert.match(reflectionLine(turn.items[turn.items.length - 1]), /^Reflecting after attempt 1/);
  push('reflection', {attempt: 1, action: 'continue', continue_run: true, rationale: 'keep going', summary: 'keep going', best_effort_result: null, requests: 1, readable: true, usage: {completion_tokens: 5}, duration_s: 2});
  assert.equal(reflectionLine(turn.items[turn.items.length - 1]), 'Reflected after attempt 1: continue · keep going (2.0s)');
  push('tool.started', {step: 4, call_id: 'call-1', name: 'pytest', arguments: {}, preview: 'pytest'});
  assert.equal(activity(turn, {now: 200}), 'Running pytest');
  assert.equal(activity(turn, {held: 'Resources', now: 200}), 'Waiting: Resources');
  assert.equal(activity(turn, {pending: 1, now: 200}), 'Waiting for your approval');
  assert.equal(activity(turn, {cancelling: true, now: 200}), 'Stopping');
  turn.retryUntil = 210;
  turn.items = turn.items.filter((item) => item.kind !== 'tool');
  assert.equal(activity(turn, {now: 200}), 'Retrying in 10s');
});

test('a reflection line reads as render.py writes it', () => {
  const item = {attempt: 1, status: 'done', action: 'continue', summary: 'a summary', rationale: 'First line.\nSecond line.', durationS: 2};
  assert.equal(reflectionLine(item), 'Reflected after attempt 1: continue · First line. (2.0s)');
  const onlyOne = {attempt: 2, status: 'done', action: 'done', summary: 'the summary', rationale: '', durationS: 12};
  assert.equal(reflectionLine(onlyOne), 'Reflected after attempt 2: done (12s)');
});

test('before the first run.progress the step is the turn\'s own', () => {
  const {turn, push} = fullTurn();
  push('step.started', {step: 2, attempt: 3});
  assert.deepEqual(liveFacts(turn), ['step 2/30', 'attempt 3']);
});

test('the run end fills the answer and flushes the calls that never started', () => {
  const {turn, push} = fullTurn();
  push('assistant.message', {step: 1, content: '<think>reading</think>Working on it.', tool_calls: [{id: 'call-2', name: 'file_read', arguments: {path: 'a'}, preview: 'a'}]});
  push('run.finished', {result: {status: 'completed', result: '<think>done</think>The answer.', duration_s: 20, n_tool: 1, steps: 1}});
  assert.equal(turn.answer, 'The answer.');
  assert.equal(turn.result.duration_s, 20);
  assert.deepEqual(turn.items.map((item) => [item.kind, item.status]), [
    ['thinking', undefined], ['assistant_text', undefined], ['tool', 'not_run'],
  ]);
  assert.equal(turn.heldText, '');
});

test('a finished turn interrupts the calls that were still running', () => {
  const {view, turn, push} = fullTurn();
  push('tool.started', {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {command: 'sleep 9'}, preview: 'sleep 9'});
  push('approval.requested', {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', arguments: {command: 'sleep 9'}, preview: 'sleep 9', workspace_root: '/tmp'});
  push('turn.finished', {turn: 1, turn_id: 'turn-1', status: 'cancelled', cause: 'shutdown', error: ''});
  assert.equal(turn.status, 'cancelled');
  assert.equal(turn.cause, 'shutdown');
  assert.equal(turn.runningHere, false);
  assert.ok(turn.finishedAt > 0);
  assert.equal(turn.items.find((item) => item.kind === 'tool').status, 'interrupted');
  assert.equal(view.pending.size, 0, 'A withdrawn approval leaves the card');
  assert.equal(toolState(turn.items.find((item) => item.kind === 'tool')).text, 'Interrupted: may have partly run');
});

test('an undone turn and a renamed session reach the page', () => {
  const {view, push} = fullTurn();
  push('session.title', {title: 'A better title'});
  const title = applyEvent(view, {id: 99, turn: 1, ts: 1, type: 'session.title', data: {title: 'Another title'}});
  assert.deepEqual(title, {title: 'Another title'});
  const updated = applyEvent(view, {id: 100, turn: 1, ts: 1, type: 'session.updated', data: {model: 'qwen3:8b', workspace: '/w', meta: {}}});
  assert.deepEqual(updated, {model: 'qwen3:8b', workspace: '/w'});
  applyEvent(view, {id: 101, turn: 1, ts: 2, type: 'turn.undone', data: {turn: 1, turn_id: 'turn-1'}});
  assert.equal(view.turns[0].undone, true);
});

// ── The terminal's words, mirrored ──

test('durations, tokens and sizes read as display.py reads them', () => {
  assert.equal(formatDuration(0.44), '0.4s');
  assert.equal(formatDuration(6.4), '6.4s');
  assert.equal(formatDuration(12), '12s');
  assert.equal(formatDuration(123), '2m03s');
  assert.equal(runElapsed(3725), '1h02m');
  assert.equal(toolElapsed(41), '0:41');
  assert.equal(toolElapsed(3723), '1:02:03');
  assert.equal(formatTokens(412), '412');
  assert.equal(formatTokens(3400), '3.4k');
  assert.equal(formatTokens(61050), '61k');
  assert.equal(formatTokens(1234567), '1.2M');
  assert.equal(formatBytes(412), '412 bytes');
  assert.equal(formatBytes(1234), '1.2 KB');
  assert.equal(formatBytes(312000), '312 KB');
  assert.equal(formatBytes(12345000), '12 MB');
});

test('a test count needs both numbers to be shown', () => {
  const {turn} = fullTurn();
  turn.progress = {step: 3, tests: {passed: 3}};
  assert.deepEqual(liveFacts(turn).filter((part) => part.startsWith('tests')), []);
  turn.progress = {step: 3, tests: '3/8'};
  assert.deepEqual(liveFacts(turn).filter((part) => part.startsWith('tests')), []);
  turn.progress = {step: 3, tests: {passed: 3, total: 8}};
  assert.deepEqual(liveFacts(turn).filter((part) => part.startsWith('tests')), ['tests 3/8']);
});

test('the past facts of a first turn follow the terminal', () => {
  const exact = {
    level: 1, basis: {match: 'exact'},
    past: {runs: 1, last_steps: 6, last_wall_s: 130},
    items: [{label: 'Read x', state: 'done'}, {label: 'Edit y', state: 'current'}],
  };
  assert.equal(formatPast(exact), 'last time 6 steps, 2m10s: Read x ✓ · Edit y ◐');
  const range = {
    level: 1, basis: {match: 'routine'},
    past: {runs: 3, steps_lo: 5, steps_hi: 8, wall_lo_s: 100, wall_hi_s: 170},
    items: [{label: 'Read x', state: 'done'}],
  };
  assert.equal(formatPast(range), 'past 5-8 steps, 1-3 min · last time: Read x ✓');
  const similar = {
    level: 1, basis: {match: 'similar', title: 'Refactor'},
    past: {runs: 1, last_steps: 3, last_wall_s: 40},
    items: [{label: 'Edit y', state: 'done'}],
  };
  assert.equal(formatPast(similar), 'like Refactor: 3 steps, 40s: Edit y ✓');
  assert.equal(formatPast({level: 1, basis: {match: 'similar'}, past: {runs: 2, steps_lo: 2, steps_hi: 3}, items: null}), 'like a similar task: 2-3 steps');
  assert.equal(formatPast({level: 0, past: {runs: 1, last_steps: 3}}), '', 'A first run has no past to show');
  assert.equal(formatPast({level: 1, basis: {match: 'exact'}, past: {runs: 1}, items: null}), '');
});

test('a status\'s words come from status_text', () => {
  assert.deepEqual(statusText('completed'), {word: 'Done', severity: 'ok', title: 'Done'});
  assert.deepEqual(statusText('cancelled', 'user_stop'), {word: 'Stopped', severity: 'stopped', title: 'Stopped by you'});
  assert.deepEqual(statusText('cancelled', 'shutdown').word, 'Interrupted');
  assert.deepEqual(statusText('crashed', 'crash').title, 'Interrupted: melo stopped unexpectedly');
  assert.deepEqual(statusText('provider_error').title, 'Could not get an answer from the model server');
  assert.deepEqual(statusText('brand_new'), {word: 'brand_new', severity: 'warn', title: 'brand_new'});
});

test('an approval tag follows the call\'s end', () => {
  const allow = {allowed: true, origin: 'user', remember: false, reason: ''};
  assert.deepEqual(approvalTag('success', 'none', allow), {short: 'allowed', long: 'Allowed once by you'});
  assert.equal(approvalTag('success', 'none', {...allow, remember: true}).short, 'allowed for the session');
  assert.equal(approvalTag('success', 'none', {...allow, origin: 'session'}).short, 'allowed by a session grant');
  assert.equal(approvalTag('success', 'none', {...allow, origin: 'policy'}).short, 'auto-allowed');
  assert.deepEqual(approvalTag('blocked', 'approval_denied', {allowed: false, origin: 'user', reason: 'not now'}), {short: 'denied: not now', long: 'Denied by you: not now'});
  assert.equal(approvalTag('blocked', 'approval_denied', {allowed: false, origin: 'policy', reason: 'rule'}).short, 'denied by policy: rule');
  assert.deepEqual(approvalTag('cancelled', 'cancelled', {allowed: false, origin: 'withdrawn'}), {short: 'not run', long: 'Not run: the turn was stopped'});
  assert.equal(approvalTag('cancelled', 'none', {origin: 'timeout'}).short, 'not answered in time');
  assert.equal(approvalTag('success', 'none', null), null, 'A call nobody asked about has no tag');
});

test('the work group\'s summary follows the turn\'s end', () => {
  const {turn, push} = fullTurn();
  assert.equal(workSummary(turn), 'Working');
  assert.equal(workSummary(turn, {held: 'Resources'}), 'Waiting: Resources');
  push('tool.started', {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {}, preview: 'ls'});
  assert.equal(workSummary(turn), 'Working · 1 tool call');
  push('tool.finished', {step: 1, call_id: 'call-1', name: 'shell_exec', status: 'error', error_category: 'runtime_error', output: '', output_chars: 0, error: 'bad', duration_s: 1, images: [], approval: null, exit_code: null});
  assert.equal(failedCount(turn), 1);
  push('run.finished', {result: {status: 'completed', result: 'done', duration_s: 14, n_tool: 1, steps: 1}});
  push('turn.finished', {turn: 1, turn_id: 'turn-1', status: 'completed', cause: 'run', error: ''});
  assert.equal(workSummary(turn), 'Worked for 14s · 1 tool call · 1 failed');

  const thought = newTurn({turn: 1, status: 'completed', started_at: 0, finished_at: 1.2, result: {duration_s: 1.2}});
  assert.equal(workSummary(thought), 'Thought for 1.2s');

  const stopped = newTurn({turn: 1, status: 'cancelled', cause: 'user_stop', started_at: 10, finished_at: 16, result: {duration_s: 6, n_tool: 2}});
  stopped.items = [{kind: 'tool', status: 'success'}, {kind: 'tool', status: 'success'}];
  assert.equal(workSummary(stopped), 'Stopped after 6.0s · 2 tool calls');
});

test('tool output is cleaned and thinking is split as the terminal does', () => {
  assert.equal(cleanOutput('\u001b[31mred\u001b[0m\r\nprogress 10%\rDone\u0007'), 'red\nDone');
  assert.deepEqual(splitThinking('<think>why</think>What'), ['why', 'What']);
  assert.deepEqual(splitThinking('<think>unfinished'), ['unfinished', '']);
  assert.deepEqual(splitThinking('before</think>after'), ['before', 'after']);
  assert.deepEqual(splitThinking('  plain  '), ['', 'plain']);
});

test('a tool card that was cancelled shows how it ended', () => {
  const withdrawn = {status: 'cancelled', errorCategory: 'cancelled', approval: {origin: 'withdrawn'}, durationS: 3, exitCode: null};
  assert.deepEqual(toolState(withdrawn), {state: 'cancelled', mark: '[ ]', text: 'not run'});
  const stopped = {status: 'cancelled', errorCategory: 'cancelled', approval: null, durationS: 3.5, exitCode: null};
  assert.deepEqual(toolState(stopped), {state: 'cancelled', mark: '[-]', text: 'stopped after 3.5s'});
  const held = {status: 'running', startedTs: 10, durationS: null, approval: null, exitCode: null};
  assert.equal(toolState(held, {held: 'Resources', now: 20}).text, 'Waiting: Resources');
});

// ── The page in a DOM double ──

const source = readFileSync(new URL('../../support/gateway/static/workspace.js', import.meta.url), 'utf8')
  .replace(/^import \{[^}]*\} from "\.\/turns\.mjs";\n/m, '')
  .replace(/^import \{ renderMarkdown \} from "\.\/markdown\.mjs";\n/m, '')
  .replace(/^import \{ setupSettings \} from "\.\/settings\.mjs";\n/m, '')
  .replace(/^import \{ setupDrawer \} from "\.\/drawer\.mjs";\n/m, '')
  .replace(/^import \{ workerHistory \} from "\.\/worker-history\.mjs";\n/m, '')
  .replace(/^import \{ setupTasks \} from "\.\/tasks\.mjs";\n/m, '')
  .replace(/^import \{ setupStatusRow \} from "\.\/status-row\.mjs";\n/m, '')
  .replace(/\ninit\(\);\s*$/, '\n');

function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

// The harness's document, for the element double made outside its closure.
let documentDouble = null;

function makeElement(tag, id = '') {
  const classes = new Set();
  const element = {
    tag, tagName: tag.toUpperCase(), id, value: '', textContent: '', innerHTML: '', title: '', type: '', src: '', alt: '',
    style: {}, dataset: {}, attributes: {}, children: [], listeners: {}, parentElement: null,
    get ownerDocument() { return documentDouble; },
    hidden: false, disabled: false, open: false, inert: false, tabIndex: 0,
    scrollHeight: 50, scrollTop: 0, clientHeight: 40,
    classList: {
      add(name) { classes.add(name); element.className = [...classes].join(' '); },
      remove(name) { classes.delete(name); element.className = [...classes].join(' '); },
      contains: (name) => classes.has(name),
      toggle(name, force) {
        const on = force === undefined ? !classes.has(name) : !!force;
        if (on) classes.add(name); else classes.delete(name);
        element.className = [...classes].join(' ');
        return on;
      },
    },
    setAttribute(name, value) { element.attributes[name] = String(value); },
    getAttribute(name) { return element.attributes[name] ?? null; },
    removeAttribute(name) { delete element.attributes[name]; },
    append(...items) { for (const item of items) { item.parentElement = element; element.children.push(item); } },
    prepend(...items) { for (const item of items) item.parentElement = element; element.children.unshift(...items); },
    replaceChildren(...items) { element.children = []; element.append(...items); },
    contains(node) { return node === element || element.children.some((child) => child.contains?.(node)); },
    addEventListener(type, handler) { (element.listeners[type] ||= []).push(handler); },
    removeEventListener() {},
    closest(selector) {
      if (selector === '.sidebar' && element.id === 'sidebar') return element;
      if (selector === '.work-group') {
        let parent = element.parentElement;
        while (parent && !parent.classList.contains('work-group')) parent = parent.parentElement;
        return parent;
      }
      return null;
    },
    focus() { documentDouble.activeElement = element; },
    setSelectionRange(start, end, direction) { element.selectionStart = start; element.selectionEnd = end; element.selectionDirection = direction; },
    click() { if (!element.disabled) return element.listeners.click?.[0]?.({target: element}); },
    close() { element.open = false; },
    showModal() { element.open = true; },
    querySelector(selector) {
      if (selector === 'h1') return findNode(element, (node) => node.tag === 'h1');
      if (selector === '[data-approval-once]') return findNode(element, (node) => 'approvalOnce' in node.dataset);
      if (selector === '[data-approval-deny]') return findNode(element, (node) => 'approvalDeny' in node.dataset);
      const focus = selector.match(/^\[data-approval-focus="(\w+)"\]$/);
      if (focus) return findNode(element, (node) => node.dataset.approvalFocus === focus[1]);
      return null;
    },
    scrollIntoView() {},
  };
  Object.defineProperty(element, 'className', {
    get() { return [...classes].join(' '); },
    set(value) {
      classes.clear();
      for (const name of String(value || '').split(/\s+/)) if (name) classes.add(name);
    },
  });
  return element;
}

function flatten(node) {
  const parts = [];
  if (node.textContent) parts.push(String(node.textContent));
  if (node.innerHTML) parts.push(String(node.innerHTML));
  for (const child of node.children ?? []) parts.push(flatten(child));
  return parts.join(' ').replace(/\s+/g, ' ').trim();
}

function findNode(node, predicate) {
  if (predicate(node)) return node;
  for (const child of node.children ?? []) {
    const found = findNode(child, predicate);
    if (found) return found;
  }
  return null;
}

class FakeEventSource {
  static instances = [];
  static reset() { FakeEventSource.instances = []; }
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.closed = false;
    FakeEventSource.instances.push(this);
  }
  close() { this.closed = true; this.readyState = 2; }
  emit(frame) { this.onmessage?.({data: typeof frame === 'string' ? frame : JSON.stringify(frame)}); }
  fail() { this.readyState = 2; this.onerror?.({}); }
}

/** The gateway's answers, by "METHOD /path"; every call is recorded. */
function gateway(routes) {
  const calls = [];
  const fetch = async (url, options = {}) => {
    const method = options.method ?? 'GET';
    const path = String(url).replace(/^\/api/, '');
    calls.push({method, path, headers: options.headers ?? {}, body: options.body ? JSON.parse(options.body) : undefined});
    const handler = routes[`${method} ${path}`] ?? routes[path];
    if (handler === undefined) throw new Error(`Unexpected request: ${method} ${path}`);
    let result = typeof handler === 'function' ? handler(options.body ? JSON.parse(options.body) : undefined) : handler;
    if (result instanceof Error) throw result;
    if (result?.then) result = await result;
    if (result instanceof Error) throw result;
    const status = result?.status ?? 200;
    const body = result?.body ?? result ?? {};
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    };
  };
  return {fetch, calls, callsTo: (path) => calls.filter((call) => call.path === path)};
}

function harness(routes, {mobile = false} = {}) {
  FakeEventSource.reset();
  const elements = new Map();
  const timers = [];
  const documentListeners = {};
  const windowListeners = {};
  const root = makeElement('html');
  const document = {
    hidden: false,
    activeElement: null,
    documentElement: root,
    addEventListener(type, handler) {
      (documentListeners[type] ||= []).push(handler);
    },
    querySelector: (selector) => {
      assert.ok(['.chat', '.composer-area'].includes(selector), 'The DOM double only supports the selectors the page uses');
      return element(selector === '.chat' ? 'chat' : 'composer-area');
    },
    querySelectorAll: () => [],
    createElement: (tag) => makeElement(tag),
  };
  function element(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id, id));
    return elements.get(id);
  }
  document.getElementById = (id) => {
    if (elements.has(id)) return elements.get(id);
    for (const node of elements.values()) {
      const found = findNode(node, (candidate) => candidate.id === id);
      if (found) return found;
    }
    return id.startsWith('approval-') && id !== 'approval-bar' && id !== 'approval-go' ? null : element(id);
  };
  documentDouble = document;
  const net = gateway(routes);
  const drawer = {
    opened: false, calls: [],
    open(tab) { this.opened = true; this.calls.push(tab); },
    close() { if (!this.opened) return false; this.opened = false; return true; },
    selectTab(tab) { this.calls.push(`select:${tab}`); }, sessionChanged() {},
  };
  const context = vm.createContext({
    ...turns,
    renderMarkdown,
    workerHistory,
    setupTasks,
    setupStatusRow,
    setupSettings: () => async () => {},
    setupDrawer: () => drawer,
    document,
    location: {hash: '', pathname: '/', search: ''},
    history: {replaceState(_state, _title, url) { context.location.hash = ''; context.replacedUrl = url; }},
    URLSearchParams,
    localStorage: storage(),
    sessionStorage: storage(),
    navigator: {clipboard: {writeText: async (text) => { context.copied = text; }}},
    matchMedia: () => ({matches: mobile, addEventListener() {}}),
    addEventListener(type, handler) {
      (windowListeners[type] ||= []).push(handler);
    },
    Option: class { constructor(label, value) { this.label = label; this.value = value; } },
    setTimeout: (handler, ms) => {
      timers.push({handler, ms, cleared: false});
      return timers.length;
    },
    clearTimeout: (id) => {
      if (timers[id - 1]) timers[id - 1].cleared = true;
    },
    setInterval: (handler, ms) => {
      timers.push({handler, ms, cleared: false, interval: true});
      return timers.length;
    },
    clearInterval: (id) => {
      if (timers[id - 1]) timers[id - 1].cleared = true;
    },
    fetch: net.fetch,
    EventSource: FakeEventSource,
  });
  vm.runInContext(source, context, {filename: 'workspace.js'});
  return {
    element,
    root,
    context,
    net,
    timers,
    run: (expression) => vm.runInContext(expression, context),
    state: () => vm.runInContext('S', context),
    text: (id) => flatten(element(id)),
    stream: () => FakeEventSource.instances[FakeEventSource.instances.length - 1],
    documentListeners,
    drawer,
    windowListeners,
    local: context.localStorage,
    session: context.sessionStorage,
  };
}

function session(id, extra = {}) {
  return {
    id, title: 'New chat', model: 'Qwen-27B', workspace: '/w', workspace_label: '~/w',
    busy: false, running_here: false, pending_approvals: 0, level: 2, held: '',
    updated_at: 1, created_at: 1, kind: 'chat', ...extra,
  };
}

const health = {
  ok: true, configured: true, model: {backend: 'mlc', backend_label: 'MLC', name: 'Qwen-27B', embedding: 'none'},
  sessions: {open: 0, running: 0, pending_approvals: 0, store: 'ok', store_error: '', resume_mode: 'pairs'},
};

function turnRow(extra = {}) {
  return {
    id: 'turn-1', turn: 1, run_id: 'run-1', user_text: 'Do the thing', status: 'completed',
    cause: 'run', answer: 'The answer.', undone: false, started_at: 10, finished_at: 20,
    client_request_id: 'key-1', running_here: false,
    result: {status: 'completed', result: 'The answer.', duration_s: 10, n_tool: 0, steps: 1},
    ...extra,
  };
}

function baseRoutes(extra = {}) {
  return {
    'POST /presence': {ok: true},
    'GET /health': health,
    'GET /sessions': {sessions: [session('a')], machine: null, has_more: false},
    'GET /sessions/a': {
      session: session('a'), pending_approvals: [], grants: [], granted_at: {},
      mcp_error: '', agent: 'not_loaded', tokens: 0,
    },
    'GET /sessions/a/turns?limit=10': {turns: [turnRow()], has_earlier: false, replay_from: 7},
    ...extra,
  };
}

test('locked startup does not load protected workspace data', async () => {
  const h = harness(baseRoutes({'GET /health': {status: 401}}));
  await h.run('init()');
  assert.deepEqual(h.net.calls.map((call) => call.path), ['/health']);
  assert.equal(h.element('app').hidden, true);
  assert.equal(h.element('app').inert, true);
  assert.equal(h.element('lock-screen').hidden, false);
  await h.run('api("/sessions").catch(() => {})');
  h.documentListeners.visibilitychange[0]();
  assert.equal(h.net.calls.length, 1);
});

test('login removes the fragment before exchange and health precedes workspace initialization', async () => {
  let h;
  h = harness(baseRoutes({'POST /auth/login': (body) => {
    assert.equal(h.context.location.hash, '');
    assert.equal(h.context.replacedUrl, '/?view=one');
    assert.equal(body.code, 'single-use-code');
    return {status: 204};
  }}));
  h.context.location.search = '?view=one';
  h.context.location.hash = '#login=single-use-code';
  await h.run('init()');
  assert.deepEqual(h.net.calls.slice(0, 3).map((call) => call.path), ['/auth/login', '/health', '/sessions']);
  assert.equal(h.element('app').hidden, false);
  assert.equal(h.element('lock-screen').hidden, true);
  assert.equal(h.local.getItem('single-use-code'), null);
  assert.equal(h.session.getItem('single-use-code'), null);
});

test('login sends the app launch identifier from the fragment', async () => {
  const instance = '0123456789abcdef0123456789abcdef';
  let h;
  h = harness(baseRoutes({'POST /auth/login': (body) => {
    assert.equal(body.code, 'single-use-code');
    assert.equal(h.context.location.hash, '');
    return {status: 204};
  }}));
  h.context.location.hash = `#login=single-use-code&instance=${instance}`;
  await h.run('init()');
  const [call] = h.net.callsTo('/auth/login');
  assert.equal(call.headers['X-LocalMelo-Instance'], instance);
});

test('invalid login never proceeds to health or displays the code', async () => {
  const h = harness(baseRoutes({'POST /auth/login': {status: 401}}));
  h.context.location.hash = '#login=secret-code';
  await h.run('init()');
  assert.deepEqual(h.net.calls.map((call) => call.path), ['/auth/login']);
  assert.equal(h.context.location.hash, '');
  assert.equal(h.element('app').hidden, true);
  assert.doesNotMatch(h.text('lock-status'), /secret-code/);
});

test('failed fragment removal and a rejected health check both fail closed', async () => {
  const failedRemoval = harness(baseRoutes());
  failedRemoval.context.location.hash = '#login=secret';
  failedRemoval.context.history.replaceState = () => { throw new Error('denied'); };
  await failedRemoval.run('init()');
  assert.equal(failedRemoval.net.calls.length, 0);
  assert.equal(failedRemoval.element('app').hidden, true);
  const rejected = harness(baseRoutes({'POST /auth/login': {status: 204}, 'GET /health': {status: 401}}));
  rejected.context.location.hash = '#login=secret';
  await rejected.run('init()');
  assert.deepEqual(rejected.net.calls.map((call) => call.path), ['/auth/login', '/health']);
  assert.equal(rejected.element('app').hidden, true);
});

test('a presence 401 locks and closes the stream', async () => {
  let expired = false;
  const h = harness(baseRoutes({'POST /presence': () => expired ? {status: 401} : {ok: true}}));
  await h.run('init()');
  const source = h.stream();
  expired = true;
  h.run('presence(true)');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(source.closed, true);
  assert.equal(h.element('app').hidden, true);
});

test('a 401 locks, preserves the draft, closes the stream and suppresses retries', async () => {
  const h = harness(baseRoutes({'GET /settings': {status: 401}}));
  await h.run('init()');
  const source = h.stream();
  h.element('input').value = 'unfinished draft';
  await h.run('api("/settings").catch(() => {})');
  assert.equal(source.closed, true);
  assert.equal(h.element('app').hidden, true);
  assert.equal(JSON.parse(h.session.getItem('localmelo-console-drafts')).a, 'unfinished draft');
  const count = h.net.calls.length;
  for (const timer of h.timers.filter((item) => item.interval)) await timer.handler();
  source.fail();
  assert.equal(h.net.calls.length, count);
  assert.equal(h.state().stream.timer, null);
});

test('a stream failure closes browser retries and a 401 on reconnect locks', async () => {
  let fail = false;
  const h = harness(baseRoutes({'GET /sessions/a': () => fail ? {status: 401} : {
    session: session('a'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0,
  }}));
  await h.run('init()');
  const source = h.stream();
  fail = true;
  source.fail();
  assert.equal(source.closed, true);
  const timer = h.timers.find((item) => !item.interval && item.ms === 1000 && !item.cleared);
  await timer.handler();
  assert.equal(h.element('app').hidden, true);
  assert.equal(FakeEventSource.instances.length, 1);
});

test('approval guard, denial keyboard and focus never allow by accident', async () => {
  const h = harness(baseRoutes({'POST /approvals/req1': {outcome: 'denied'}}));
  await h.run('init()');
  h.element('input').focus();
  h.context.request = {request_id: 'req1', tool_name: 'shell_exec', arguments: {command: 'make test'}, workspace_label: '~/w'};
  h.run('S.view.pending.set("req1", request)');
  const card = h.run('approvalCard(request)');
  const once = findNode(card, (node) => node.textContent === 'Allow once');
  const deny = findNode(card, (node) => node.textContent === 'Deny…');
  assert.equal(h.context.document.activeElement, h.element('input'));
  assert.equal(card.attributes.role, 'region');
  assert.equal(once.disabled, true);
  once.click();
  assert.equal(h.net.callsTo('/approvals/req1').length, 0);
  h.run('approvalStates.get("req1").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  assert.equal(deny.disabled, false);
  deny.click();
  const reason = findNode(card, (node) => node.className === 'deny-reason');
  assert.equal(h.context.document.activeElement, reason);
  reason.value = 'Needs review';
  reason.listeners.input[0]();
  const event = {key: 'Enter', isComposing: true, keyCode: 229, preventDefault() {}, stopPropagation() {}};
  reason.listeners.keydown[0](event);
  assert.equal(h.net.callsTo('/approvals/req1').length, 0);
  event.isComposing = false; event.keyCode = 13;
  reason.listeners.keydown[0](event);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(h.net.callsTo('/approvals/req1')[0].body, {choice: 'deny', reason: 'Needs review'});
});

test('approval buttons post only their explicit choices and Escape closes denial', async () => {
  for (const [choice, label] of [['once', 'Allow once'], ['session', 'Allow every shell_exec call in this session']]) {
    const h = harness(baseRoutes({'POST /approvals/req': {outcome: 'allowed'}}));
    await h.run('init()');
    h.context.request = {request_id: 'req', tool_name: 'shell_exec', arguments: {command: 'pwd'}, arguments_complete: true};
    h.run('S.view.pending.set("req", request)');
    const card = h.run('approvalCard(request)');
    const target = findNode(card, (node) => node.textContent === label);
    h.run('approvalStates.get("req").mountedAt -= 1000');
    h.timers.findLast((timer) => timer.ms <= 500).handler();
    await target.click();
    assert.deepEqual(h.net.callsTo('/approvals/req')[0].body, {choice});
  }
  const h = harness(baseRoutes());
  await h.run('init()');
  h.context.request = {request_id: 'deny', tool_name: 'python_exec', arguments: {code: 'print(1)'}};
  h.run('S.view.pending.set("deny", request)');
  const card = h.run('approvalCard(request)');
  h.run('approvalStates.get("deny").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  const deny = findNode(card, (node) => node.textContent === 'Deny…');
  deny.click();
  const reason = findNode(card, (node) => node.className === 'deny-reason');
  let stopped = false;
  reason.listeners.keydown[0]({key: 'Escape', isComposing: true, keyCode: 229, preventDefault() {}, stopPropagation() { stopped = true; }});
  assert.equal(h.context.document.activeElement, reason);
  assert.equal(stopped, false);
  reason.listeners.keydown[0]({key: 'Escape', isComposing: false, keyCode: 27, preventDefault() {}, stopPropagation() { stopped = true; }});
  assert.equal(stopped, true);
  assert.equal(h.context.document.activeElement, deny);
  assert.equal(h.net.callsTo('/approvals/deny').length, 0);
});

test('approval bar navigates to the card only on request', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'two', user_text: 'Run', client_request_id: 'k'}});
  h.stream().emit({id: 21, turn: 2, ts: 51, type: 'tool.started', data: {call_id: 'c', name: 'shell_exec', arguments: {command: 'pwd'}}});
  h.element('input').focus();
  h.stream().emit({id: 22, turn: 2, ts: 52, type: 'approval.requested', data: {request_id: 'go', call_id: 'c', tool_name: 'shell_exec', arguments: {command: 'pwd'}}});
  assert.equal(h.context.document.activeElement, h.element('input'));
  assert.equal(h.element('approval-bar').hidden, false);
  h.run('S.view.pending.get("go").arguments_complete = true; approvalStates.get("go").mountedAt -= 1000; messageSnapshot = ""; renderMessages(false)');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  h.element('approval-go').click();
  assert.equal(h.context.document.activeElement.textContent, 'Allow once');
});

test('approval guard survives redraw but resets on navigation', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.context.request = {request_id: 'guard', tool_name: 'shell_exec', arguments: {command: 'pwd'}, arguments_complete: true};
  h.run('S.view.pending.set("guard", request); approvalCard(request); approvalStates.get("guard").mountedAt -= 1000');
  const same = h.run('approvalCard(request)');
  assert.equal(findNode(same, (node) => node.textContent === 'Allow once').disabled, false);
  h.run('newChat(); approvalCard(request)');
  const fresh = h.run('approvalCard(request)');
  assert.equal(findNode(fresh, (node) => node.textContent === 'Allow once').disabled, true);
});

test('denial input keeps selection and defers redraw while composing', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'two', user_text: 'Run'}});
  h.stream().emit({id: 21, turn: 2, ts: 51, type: 'tool.started', data: {call_id: 'c', name: 'shell_exec', arguments: {command: 'pwd'}}});
  h.stream().emit({id: 22, turn: 2, ts: 52, type: 'approval.requested', data: {request_id: 'ime', call_id: 'c', tool_name: 'shell_exec', arguments: {command: 'pwd'}}});
  h.run('approvalStates.get("ime").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  let card = h.context.document.getElementById('approval-card-ime');
  findNode(card, (node) => node.textContent === 'Deny…').click();
  let reason = findNode(card, (node) => node.className === 'deny-reason');
  reason.value = 'abcdef'; reason.listeners.input[0](); reason.setSelectionRange(2, 4, 'forward');
  h.stream().emit({id: 23, type: 'session.updated', data: {model: 'changed'}});
  card = h.context.document.getElementById('approval-card-ime');
  reason = findNode(card, (node) => node.className === 'deny-reason');
  assert.equal(h.context.document.activeElement, reason);
  assert.deepEqual([reason.value, reason.selectionStart, reason.selectionEnd], ['abcdef', 2, 4]);
  reason.listeners.compositionstart[0]();
  reason.value = 'abc語def';
  h.stream().emit({id: 24, type: 'session.updated', data: {model: 'again'}});
  assert.equal(h.context.document.activeElement, reason);
  assert.equal(h.context.document.getElementById('approval-card-ime'), card);
  reason.listeners.compositionend[0]();
  const replacement = findNode(h.context.document.getElementById('approval-card-ime'), (node) => node.className === 'deny-reason');
  assert.equal(h.context.document.activeElement, replacement);
  assert.equal(replacement.value, 'abc語def');
});

test('workspace labels follow new events and old events keep their label', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().emit({id: 20, type: 'session.updated', data: {model: 'm2', workspace: '/home/me/project', workspace_label: '~/project'}});
  assert.equal(h.state().view.session.workspace_label, '~/project');
  assert.equal(h.state().sessions[0].workspace_label, '~/project');
  h.stream().emit({id: 21, type: 'session.updated', data: {model: 'm3', workspace: '/old'}});
  assert.equal(h.state().view.session.workspace_label, '~/project');
});

test('approval shows complete multi-line shell commands and generic arguments', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.context.request = {request_id: 'long', tool_name: 'shell_exec', arguments: {command: Array.from({length: 25}, (_, i) => `line-${i}`).join('\n')}};
  const shell = h.run('approvalCard(request)');
  assert.match(flatten(shell), /line-24/);
  assert.equal(findNode(shell, (node) => String(node.textContent).startsWith('Show all')), null);
  h.context.request = {request_id: 'custom-long', tool_name: 'custom', arguments: {values: Array.from({length: 25}, (_, i) => `value-${i}`)}};
  const other = h.run('approvalCard(request)');
  assert.match(flatten(other), /value-24/);
});

test('incomplete or unconfirmed arguments cannot be allowed, but can be denied', async () => {
  for (const complete of [false, undefined]) {
    const h = harness(baseRoutes({'POST /approvals/unsafe': {outcome: 'denied'}}));
    await h.run('init()');
    h.context.request = {request_id: 'unsafe', tool_name: 'shell_exec', arguments: {command: 'truncated…'}, arguments_complete: complete};
    h.run('S.view.pending.set("unsafe", request)');
    const card = h.run('approvalCard(request)');
    h.run('approvalStates.get("unsafe").mountedAt -= 1000');
    h.timers.findLast((timer) => timer.ms <= 500).handler();
    assert.equal(findNode(card, (node) => node.textContent === 'Allow once').disabled, true);
    assert.equal(findNode(card, (node) => node.textContent === 'Allow every shell_exec call in this session').disabled, true);
    await h.run('answerApproval(request, approvalStates.get("unsafe"), "once")');
    assert.equal(h.net.callsTo('/approvals/unsafe').length, 0);
    assert.match(flatten(card), /cannot safely be allowed|allow is disabled/);
    findNode(card, (node) => node.textContent === 'Deny…').click();
    await findNode(card, (node) => node.textContent === 'Deny').click();
    assert.equal(h.net.callsTo('/approvals/unsafe')[0].body.choice, 'deny');
  }
});

test('file approval can be denied while target details are unavailable', async () => {
  const h = harness(baseRoutes({'POST /approvals/missing': {outcome: 'denied'}}));
  await h.run('init()');
  h.context.request = {request_id: 'missing', tool_name: 'file_write', arguments: {path: 'x', content: 'y'}, target: null};
  h.run('S.view.pending.set("missing", request)');
  const card = h.run('approvalCard(request)');
  h.run('approvalStates.get("missing").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  assert.equal(findNode(card, (node) => node.textContent === 'Allow once').disabled, true);
  const deny = findNode(card, (node) => node.textContent === 'Deny…');
  assert.equal(deny.disabled, false);
  deny.click();
  await findNode(card, (node) => node.textContent === 'Deny').click();
  assert.equal(h.net.callsTo('/approvals/missing')[0].body.choice, 'deny');
});

for (const succeeds of [true, false]) {
  test(`approval detail refresh ${succeeds ? 'enables verified details' : 'fails closed'}`, async () => {
    let detail = {session: session('a'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0};
    const h = harness(baseRoutes({'GET /sessions/a': () => detail}));
    await h.run('init()');
    h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'two', user_text: 'Run'}});
    h.stream().emit({id: 21, turn: 2, ts: 51, type: 'tool.started', data: {call_id: 'c', name: 'file_write', arguments: {path: 'a', content: 'x'}}});
    h.stream().emit({id: 22, turn: 2, ts: 52, type: 'approval.requested', data: {request_id: 'refresh', call_id: 'c', tool_name: 'file_write', arguments: {path: 'a', content: 'x'}}});
    await new Promise((resolve) => setImmediate(resolve));
    h.run('approvalStates.get("refresh").mountedAt -= 1000');
    h.timers.findLast((timer) => timer.ms <= 500).handler();
    let card = h.context.document.getElementById('approval-card-refresh');
    assert.equal(findNode(card, (node) => node.textContent === 'Allow once').disabled, true);
    assert.equal(findNode(card, (node) => node.textContent === 'Deny…').disabled, false);
    detail = succeeds
      ? {pending_approvals: [{request_id: 'refresh', arguments_complete: true, target: {path_label: 'a', exists: true, size_bytes: 7}}]}
      : new Error('details unavailable');
    await findNode(card, (node) => node.textContent === 'Reload approval details').click();
    card = h.context.document.getElementById('approval-card-refresh');
    assert.equal(h.net.callsTo('/sessions/a').length, 3);
    assert.equal(findNode(card, (node) => node.textContent === 'Allow once').disabled, !succeeds);
    assert.equal(h.net.callsTo('/approvals/refresh').length, 0);
    if (succeeds) {
      assert.match(flatten(card), /overwrite a \(7 bytes\)/);
      assert.equal(findNode(card, (node) => node.textContent === 'Reload approval details'), null);
    } else {
      assert.match(h.element('notice').textContent, /Could not load approval details: details unavailable/);
      assert.equal(findNode(card, (node) => node.textContent === 'Deny…').disabled, false);
    }
  });
}

test('earlier paging preserves the scroll position on a short page', async () => {
  const earlier = turnRow({id: 'old', turn: 1, user_text: 'Earlier'});
  earlier.events = [];
  const h = harness(baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [turnRow({id: 'new', turn: 2})], has_earlier: true, replay_from: 10},
    'GET /sessions/a/turns?before=2&limit=10&events=true': {turns: [earlier], has_earlier: false},
  }));
  await h.run('init()');
  h.element('scroll').scrollTop = 0;
  await h.run('loadEarlier()');
  assert.equal(h.element('scroll').scrollTop, 0);
});

test('earlier paging adjusts scroll height and restores keyboard focus', async () => {
  for (const hasEarlier of [false, true]) {
    const older = turnRow({id: 'old', turn: 1});
    older.events = [];
    let resolvePage;
    const page = new Promise((resolve) => { resolvePage = resolve; });
    const h = harness(baseRoutes({
      'GET /sessions/a/turns?limit=10': {turns: [turnRow({id: 'new', turn: 2})], has_earlier: true, replay_from: 10},
      'GET /sessions/a/turns?before=2&limit=10&events=true': () => page,
    }));
    await h.run('init()');
    const messages = h.element('messages');
    const scroller = h.element('scroll');
    // Model actual layout: a turn adds 220px; the paging control adds 30px.
    Object.defineProperty(scroller, 'scrollHeight', {get: () => 300 + messages.children.reduce((n, child) => n + (child.classList.contains('turn') ? 220 : child.classList.contains('earlier-turns') ? 30 : 0), 0)});
    const replace = messages.replaceChildren;
    messages.replaceChildren = (...items) => {
      if (messages.contains(h.context.document.activeElement)) h.context.document.activeElement = null;
      replace(...items);
    };
    scroller.scrollTop = 130;
    const oldHeight = scroller.scrollHeight;
    const earlier = findNode(messages, (node) => node.classList.contains('earlier-turns'));
    earlier.focus();
    const pending = earlier.click();
    assert.equal(h.context.document.activeElement, scroller);
    resolvePage({turns: [older], has_earlier: hasEarlier});
    await pending;
    assert.equal(scroller.scrollTop, 130 + scroller.scrollHeight - oldHeight);
    const focus = h.context.document.activeElement;
    assert.equal(focus.classList.contains(hasEarlier ? 'earlier-turns' : 'turn'), true);
    if (!hasEarlier) assert.equal(focus.dataset.turnId, 'old');
  }
});

test('failed earlier paging restores the paging button focus', async () => {
  const h = harness(baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [turnRow({id: 'new', turn: 2})], has_earlier: true, replay_from: 10},
    'GET /sessions/a/turns?before=2&limit=10&events=true': new Error('network down'),
  }));
  await h.run('init()');
  const earlier = findNode(h.element('messages'), (node) => node.classList.contains('earlier-turns'));
  earlier.focus();
  await earlier.click();
  assert.equal(h.context.document.activeElement.classList.contains('earlier-turns'), true);
  assert.equal(h.context.document.activeElement.disabled, false);
  assert.match(h.element('notice').textContent, /network down/);
});

test('approval cards display each tool and confirm file target before enabling', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  for (const [name, args, target, expected] of [
    ['shell_exec', {command: 'echo hello'}, null, /command in ~\/w/],
    ['python_exec', {code: 'print(1)'}, null, /Python in ~\/w/],
    ['file_write', {path: 'a.txt', content: 'abc'}, {path_label: 'a.txt', exists: true, size_bytes: 7}, /overwrite a\.txt \(7 bytes\) with 3 chars/],
    ['file_write', {path: 'b.txt', content: 'abc'}, {path_label: 'b.txt', exists: false, size_bytes: 0}, /create b\.txt \(3 chars\)/],
    ['custom', {value: 1}, null, /custom wants to run/],
  ]) {
    h.context.request = {request_id: `req-${name}-${target?.exists}`, tool_name: name, arguments: args, workspace_label: '~/w', target};
    const card = h.run('approvalCard(request)');
    assert.match(flatten(card), expected);
  }
  h.context.request = {request_id: 'file-unchecked', tool_name: 'file_write', arguments: {path: 'x'}, target: null};
  const card = h.run('approvalCard(request)');
  h.run('approvalStates.get("file-unchecked").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  assert.equal(findNode(card, (node) => node.textContent === 'Allow once').disabled, true);
});

test('API errors retain validation details', async () => {
  const h = harness(baseRoutes({'PUT /settings': {status: 400, body: {error: {code: 'invalid_settings', message: 'first', details: {errors: ['first', 'second']}}}}}));
  await h.run('init()');
  await assert.rejects(h.run('api("/settings", {method: "PUT", body: {}})'), (error) => {
    assert.equal(error.code, 'invalid_settings');
    assert.deepEqual(Array.from(error.details.errors), ['first', 'second']);
    return true;
  });
});

test('title and independent banners follow active state without replacing notices', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.state().health = {...health, configured: false, sessions: {...health.sessions, store: 'error', store_error: 'database offline'}};
  h.state().view.pending.set('req', {});
  h.run('notify("Other error", {sticky: true}); S.stream.status = "paused"; S.stream.retryAt = 0; syncUI()');
  assert.match(h.context.document.title, /localmelo · New chat · 1/);
  assert.match(h.element('store-banner').textContent, /database offline/);
  assert.match(h.element('model-banner').textContent, /No model/);
  assert.equal(h.element('stream-banner').hidden, false);
  assert.equal(h.element('notice').textContent, 'Other error');
  await h.element('reconnect-btn').click();
  assert.equal(h.state().stream.status, 'live');
  h.run('syncUI()');
  assert.equal(h.element('stream-banner').hidden, true);
});

test('status banners do not rewrite unchanged live text', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.state().health = {...health, configured: false, sessions: {...health.sessions, store: 'error', store_error: 'offline'}};
  h.run('S.stream.status="paused"; S.stream.retryAt=0; syncUI()');
  const writes = {};
  for (const id of ['store-banner', 'model-banner', 'stream-banner-text']) {
    const node = h.element(id);
    let value = node.textContent;
    writes[id] = 0;
    Object.defineProperty(node, 'textContent', {get: () => value, set: (text) => { writes[id]++; value = text; }});
  }
  h.run('syncUI(); refreshStreamNotice()');
  assert.deepEqual(writes, {'store-banner': 0, 'model-banner': 0, 'stream-banner-text': 0});
  h.state().health.sessions.store_error = 'new error';
  h.run('syncUI()');
  assert.equal(writes['store-banner'], 1);
  assert.equal(writes['model-banner'], 0);
});

test('setup failure offers the details drawer tools action', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  const turn = turns.newTurn(turnRow({status: 'setup_error', result: {error: 'MCP failed'}}));
  h.context.turn = turn;
  const node = h.run('statusNode(turn)');
  assert.match(flatten(node), /MCP server failed to start/);
  const show = findNode(node, (item) => item.textContent === 'Show tools');
  assert.ok(show);
  show.click();
  assert.deepEqual(h.drawer.calls, ['tools']);
});

test('failure actions are limited to the newest eligible turn', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  const old = turns.newTurn(turnRow({turn: 1, status: 'timeout', result: {n_tool: 2}}));
  const latest = turns.newTurn(turnRow({id: 'two', turn: 2, status: 'budget_exhausted', result: {n_tool: 2, error: 'ran out'}}));
  h.state().view.turns = [old, latest];
  h.context.old = old; h.context.latest = latest;
  const oldText = flatten(h.run('statusNode(old)'));
  const latestText = flatten(h.run('statusNode(latest)'));
  assert.doesNotMatch(oldText, /Continue|Try again/);
  assert.match(latestText, /Continue/);
  assert.match(latestText, /Try again/);
  assert.match(latestText, /Copy error/);
  latest.result.n_tool = 0;
  assert.equal(findNode(h.run('statusNode(latest)'), (node) => node.tag === 'button' && node.textContent === 'Continue'), null);
  assert.equal(turns.statusLine('timeout', 'a\n b'), 'melo: run ended with status timeout: a b');
});

test('failure controls copy bounded errors, continue without losing drafts and retry by turn', async () => {
  const failed = turnRow({status: 'timeout', result: {n_tool: 1, error: 'line one\nline two'}});
  const routes = baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [failed], has_earlier: false, replay_from: 7},
    'POST /sessions/a/turns': {status: 202, body: {repeated: false}},
    'POST /sessions/a/retry': {status: 202, body: {}},
  });
  const h = harness(routes);
  await h.run('init()');
  h.element('input').value = 'My unsent draft';
  h.run('saveDraft()');
  const card = h.run('statusNode(S.view.turns[0])');
  await findNode(card, (node) => node.textContent === 'Copy error').click();
  assert.equal(h.context.copied, 'melo: run ended with status timeout: line one line two\nrun run-1');
  await findNode(card, (node) => node.textContent === 'Continue').click();
  assert.equal(h.net.callsTo('/sessions/a/turns')[0].body.text, 'Continue the task from where you stopped.');
  assert.equal(h.element('input').value, 'My unsent draft');
  delete h.state().outbox.a;
  await findNode(card, (node) => node.textContent === 'Try again').click();
  assert.deepEqual(h.net.callsTo('/sessions/a/retry')[0].body, {turn_id: 'turn-1'});
  failed.status = 'cancelled'; failed.cause = 'user_stop';
  const stopped = turns.newTurn(failed);
  h.context.stopped = stopped;
  assert.match(flatten(h.run('statusNode(stopped)')), /Stopped by you/);
  assert.doesNotMatch(flatten(h.run('statusNode(stopped)')), /Copy error|Details/);
});

test('earlier turns replay separately without disturbing current cursor or approvals', async () => {
  const earlier = turnRow({id: 'old', turn: 1, user_text: 'Earlier', answer: 'Old answer'});
  earlier.events = [{id: 2, turn: 1, ts: 11, type: 'assistant.message', data: {content: 'old work', tool_calls: [{id: 'old-call', name: 'read'}]}}];
  const h = harness(baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [turnRow({id: 'new', turn: 2})], has_earlier: true, replay_from: 10},
    'GET /sessions/a/turns?before=2&limit=10&events=true': {turns: [earlier], has_earlier: false},
  }));
  await h.run('init()');
  h.state().view.pending.set('current', {request_id: 'current'});
  const before = h.state().view.lastEventId;
  await h.run('loadEarlier()');
  assert.deepEqual(Array.from(h.state().view.turns, (turn) => turn.turn), [1, 2]);
  assert.equal(h.state().view.lastEventId, before);
  assert.equal(h.state().view.pending.has('current'), true);
  assert.match(h.text('messages'), /Earlier/);
  assert.equal(h.state().view.hasEarlier, false);
});

test('accepted retry clears only the matching restored draft when SSE wins', async () => {
  for (const draft of ['Retry me', 'New unsent text']) {
    let release;
    const waiting = new Promise((resolve) => { release = resolve; });
    const h = harness(baseRoutes({'POST /sessions/a/turns': () => waiting}));
    await h.run('init()');
    h.state().outbox.a = {clientRequestId: 'retry-key', text: 'Retry me', state: 'failed'};
    h.state().drafts.a = draft;
    h.element('input').value = draft;
    const card = h.run('outboxNode(S.outbox.a, "a")');
    findNode(card, (node) => node.textContent === 'Retry').click();
    h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'two', user_text: 'Retry me', client_request_id: 'retry-key'}});
    release({turn: {id: 'two'}});
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(h.state().outbox.a, undefined);
    assert.equal(h.state().drafts.a || '', draft === 'Retry me' ? '' : draft);
    assert.equal(h.element('input').value, draft === 'Retry me' ? '' : draft);
  }
});

test('a late outbox check cannot replace a newer message', async () => {
  let release;
  const waiting = new Promise((resolve) => { release = resolve; });
  const h = harness(baseRoutes({'GET /sessions/a/submissions/old-key': () => waiting}));
  await h.run('init()');
  h.state().outbox.a = {clientRequestId: 'old-key', text: 'Old', state: 'checking'};
  const check = h.run('checkSubmission("a", "old-key")');
  h.state().outbox.a = {clientRequestId: 'new-key', text: 'New', state: 'sending'};
  release({submission: {state: 'unknown'}});
  await check;
  assert.equal(h.state().outbox.a.clientRequestId, 'new-key');
  assert.equal(h.state().outbox.a.state, 'sending');
});

test('a late cancellation restores its own chat draft', async () => {
  let release;
  const waiting = new Promise((resolve) => { release = resolve; });
  const h = harness(baseRoutes({'POST /sessions/a/submissions/key/cancel': () => waiting}));
  await h.run('init()');
  h.state().outbox.a = {clientRequestId: 'key', text: 'Original', state: 'failed'};
  const cancel = h.run('cancelSend("a", S.outbox.a)');
  h.run('newChat()');
  h.element('input').value = 'Other chat';
  release({});
  await cancel;
  assert.equal(h.element('input').value, 'Other chat');
  assert.equal(h.state().drafts.a, 'Original');
});

test('a persisted in-flight outbox is reconciled with its original key', async () => {
  const h = harness(baseRoutes({'GET /sessions/a/submissions/key-keep': {submission: {state: 'unknown'}}}));
  h.session.setItem('localmelo-console-outbox', JSON.stringify({a: {clientRequestId: 'key-keep', text: 'Wait for me', state: 'sending', error: ''}}));
  await h.run('init()');
  assert.equal(h.net.callsTo('/sessions/a/submissions/key-keep').length, 1);
  assert.equal(h.state().outbox.a.clientRequestId, 'key-keep');
  assert.equal(h.state().outbox.a.state, 'failed');
  assert.equal(JSON.parse(h.session.getItem('localmelo-console-outbox')).a.clientRequestId, 'key-keep');
});

test('the page boots from /api and renders a session\'s turns', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  assert.deepEqual(h.net.calls.map((call) => `${call.method} ${call.path}`), [
    'GET /health', 'GET /sessions',
    'POST /presence', 'GET /sessions/a', 'GET /sessions/a/turns?limit=10',
    'GET /sessions/a/status',
  ]);
  assert.equal(h.state().activeId, 'a');
  const transcript = h.text('messages');
  assert.match(transcript, /Do the thing/);
  assert.match(transcript, /The answer\./);
  assert.match(transcript, /10s · 1 step/);
  assert.equal(h.element('machine-state').hidden, true);
  assert.equal(h.element('connection-label').textContent, 'Ready');
  assert.equal(h.element('header-title').textContent, 'New chat');
  assert.equal(h.stream().url, '/api/sessions/a/events?after=7');
  assert.equal(h.root.dataset.ready, '1', 'The first load marks the page ready');
});

test('the chat shows the server\'s status segments and refreshes them on a frame', async () => {
  let segments = ['cache 40%', 'ctx 1.0k/8.0k'];
  const h = harness(baseRoutes({'GET /sessions/a/status': () => ({session_id: 'a', segments})}));
  await h.run('init()');
  await new Promise((resolve) => setImmediate(resolve));
  const row = h.element('status-row');
  assert.equal(row.hidden, false);
  assert.deepEqual(row.children.map((child) => child.textContent), ['cache 40%', 'ctx 1.0k/8.0k']);
  assert.equal(h.net.callsTo('/sessions/a/status').length, 1);
  segments = ['cache --', 'step 2'];
  h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'two', user_text: 'Run'}});
  await new Promise((resolve) => setImmediate(resolve));
  // A frame within a second of the last fetch waits for its slot.
  assert.equal(h.net.callsTo('/sessions/a/status').length, 1);
});

test('the #/new route opens a new chat even when chats are saved', async () => {
  const h = harness(baseRoutes());
  h.context.location.hash = '#/new';
  h.local.setItem('localmelo-console-view', JSON.stringify({id: 'a'}));
  await h.run('init()');
  assert.equal(h.state().activeId, null);
  assert.equal(h.net.callsTo('/sessions/a').length, 0, 'A new chat opens no session');
  assert.equal(h.element('welcome').hidden, false);
  h.context.location.hash = '#/s/a';
  await h.run('init()');
  assert.equal(h.state().activeId, 'a', 'A session hash opens that chat');
});

test('a hash chat outside the loaded list is opened anyway', async () => {
  const h = harness(baseRoutes({'GET /sessions': {sessions: [], machine: null, has_more: false}}));
  h.context.location.hash = '#/s/a';
  await h.run('init()');
  assert.equal(h.state().activeId, 'a');
  assert.equal(h.net.callsTo('/sessions/a').length, 1, 'The hash names a chat, not a list row');
});

test('a hash chat the store does not have falls back to a new chat', async () => {
  const missing = new Error('Session not found.');
  missing.status = 404;
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [], machine: null, has_more: false},
    'GET /sessions/a': () => missing,
  }));
  h.context.location.hash = '#/s/a';
  await h.run('init()');
  assert.equal(h.state().activeId, null);
  assert.equal(h.context.location.hash, '#/new');
  assert.match(h.element('notice').textContent, /expired or the gateway restarted/);
});

test('Back and Forward move the chat with the hash', async () => {
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a'), session('b')], machine: null, has_more: false},
    'GET /sessions/b': {session: session('b'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0},
    'GET /sessions/b/turns?limit=10': {turns: [], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  assert.equal(h.state().activeId, 'a');
  const hashchange = h.windowListeners.hashchange[0];
  assert.ok(hashchange, 'The page listens for the hash');
  h.context.location.hash = '#/s/b';
  await hashchange();
  assert.equal(h.state().activeId, 'b');
  assert.equal(h.context.location.hash, '#/s/b');
  h.context.location.hash = '#/new';
  await hashchange();
  assert.equal(h.state().activeId, null);
  assert.equal(h.element('welcome').hidden, false);
});

test('the sidebar escapes a server title and model', async () => {
  const title = '<img src=x onerror=alert(1)>';
  const model = 'a&"b';
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a', {title, model})], machine: null, has_more: false},
    'GET /sessions/a': {
      session: session('a', {title, model}), pending_approvals: [], grants: [], granted_at: {},
      mcp_error: '', agent: 'not_loaded', tokens: 0,
    },
  }));
  await h.run('init()');
  const info = h.element('session-list').children[0].children[0];
  assert.match(info.innerHTML, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(info.innerHTML, /a&amp;&quot;b/);
  assert.doesNotMatch(info.innerHTML, /<img/);
});

test('a send in an open chat posts a keyed turn and waits for its event', async () => {
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': {status: 202, body: {turn_id: 'turn-2', turn: turnRow({id: 'turn-2', turn: 2, status: 'running', answer: null, finished_at: 0})}},
  }));
  await h.run('init()');
  h.element('input').value = 'Do it again';
  await h.run('send()');
  const [call] = h.net.callsTo('/sessions/a/turns');
  assert.equal(call.method, 'POST');
  assert.equal(call.body.text, 'Do it again');
  assert.match(call.body.client_request_id, /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/);
  assert.equal(h.state().outbox.a.state, 'sending');
  assert.match(h.text('messages'), /Sending…/);

  const key = call.body.client_request_id;
  h.stream().emit({
    id: 30, turn: 2, ts: 100, type: 'turn.started',
    data: {turn: 2, turn_id: 'turn-2', user_text: 'Do it again', client_request_id: key},
  });
  assert.equal(h.state().outbox.a, undefined);
  assert.doesNotMatch(h.text('messages'), /Sending…/);
  assert.match(h.text('messages'), /Do it again/);
});

test('a send in a new chat creates the chat with its first turn', async () => {
  const created = session('b', {title: 'Do the thing'});
  const h = harness({
    ...baseRoutes({'GET /sessions': {sessions: [], machine: null, has_more: false}}),
    'POST /sessions': {status: 201, body: {session: created, turn_id: 'turn-1', turn: turnRow()}},
    'GET /sessions/b': {session: created, pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'open', tokens: 0},
    'GET /sessions/b/turns?limit=10': {turns: [turnRow({client_request_id: 'first'})], has_earlier: false, replay_from: 0},
  });
  await h.run('init()');
  h.element('input').value = 'Do the thing';
  await h.run('send()');
  const call = h.net.callsTo('/sessions').find((entry) => entry.method === 'POST');
  assert.ok(call, 'Creating a chat is a POST');
  assert.match(call.body.turn.client_request_id, /^[A-Za-z0-9-]{1,128}$/);
  assert.equal(call.body.turn.text, 'Do the thing');
  assert.equal(h.state().activeId, 'b');
  assert.equal(h.context.location.hash, '#/s/b');
  assert.equal(h.element('input').value, '');
  assert.equal(h.stream().url, '/api/sessions/b/events?after=0');
  assert.match(h.text('messages'), /The answer\./);
});

test('a refused send stays as an unsent draft and is retried under the same key', async () => {
  let attempts = 0;
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => {
      attempts += 1;
      if (attempts === 1) return {status: 400, body: {error: {code: 'invalid_request', message: 'text: the message must not be blank'}}};
      return {status: 202, body: {turn_id: 'turn-2', turn: turnRow({id: 'turn-2', turn: 2})}};
    },
  }));
  await h.run('init()');
  h.element('input').value = 'Try me';
  await h.run('send()');
  assert.equal(h.state().outbox.a.state, 'failed');
  assert.match(h.text('messages'), /Not sent: text: the message must not be blank/);
  assert.equal(h.element('input').value, 'Try me', 'The refused text comes back to the composer');
  assert.equal(
    JSON.parse(h.session.getItem('localmelo-console-drafts')).a,
    'Try me',
    'The chat draft survives a reload, where the in-memory outbox does not',
  );
  const retry = flatten(h.element('messages')).includes('Retry');
  assert.equal(retry, true);
  const first = h.net.callsTo('/sessions/a/turns')[0].body.client_request_id;
  await h.run('submit("a", S.outbox.a.clientRequestId, S.outbox.a.text)');
  const second = h.net.callsTo('/sessions/a/turns')[1].body.client_request_id;
  assert.equal(second, first, 'A retry repeats the key, so one message makes one turn');
  assert.equal(h.element('input').value, '', 'An accepted retry is not a draft any more');
  assert.equal(h.state().drafts.a, undefined);
});

test('a send the store already accepted is not sent twice', async () => {
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': {status: 200, body: {turn_id: 'turn-2', turn: turnRow({id: 'turn-2', turn: 2}), repeated: true}},
    'GET /sessions/a/turns?limit=10': {turns: [turnRow(), turnRow({id: 'turn-2', turn: 2, user_text: 'Only once'})], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  h.element('input').value = 'Only once';
  await h.run('send()');
  assert.equal(h.state().outbox.a, undefined, 'A repeated key is the same message, already sent');
  assert.equal(h.net.callsTo('/sessions/a/turns?limit=10').length, 2, 'The view reloads to show the turn');
  assert.match(h.text('messages'), /Only once/);
});

test('a lost response asks the store before it offers a retry', async () => {
  const key = '11111111-1111-1111-1111-111111111111';
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => new Error('The connection dropped'),
    [`GET /sessions/a/submissions/${key}`]: {submission: {client_request_id: key, state: 'accepted', session_id: 'a', turn_id: 'turn-2', status: 'running', cause: '', error: '', created_at: 1, turn: null}},
  }));
  await h.run('init()');
  h.element('input').value = 'Did it land?';
  h.context.crypto = {randomUUID: () => key};
  await h.run('send()');
  assert.equal(h.state().outbox.a, undefined, 'An accepted message is not offered again');
  assert.equal(h.net.callsTo(`/sessions/a/submissions/${key}`).length, 1);
  assert.equal(h.net.callsTo('/sessions/a/turns?limit=10').length, 2, 'The view reloads to show the turn');
  assert.equal(FakeEventSource.instances.length, 2, 'The rebuilt view reopens the stream');
});

test('the page words every pair in the shared display vectors', () => {
  // tests/sessions/test_shared.py reads the same file: one table, both surfaces.
  const vectors = JSON.parse(readFileSync(new URL('../fixtures/display_vectors.json', import.meta.url), 'utf8'));
  for (const [seconds, text] of vectors.format_duration) {
    assert.equal(formatDuration(seconds), text, `format_duration(${seconds})`);
  }
  for (const [count, text] of vectors.format_tokens) {
    assert.equal(formatTokens(count), text, `format_tokens(${count})`);
  }
  for (const [input, text] of vectors.clean_output) {
    assert.equal(cleanOutput(input), text, `clean_output(${JSON.stringify(input)})`);
  }
  for (const [status, cause, word, severity, title] of vectors.status_text) {
    const read = statusText(status, cause);
    assert.deepEqual(
      {word: read.word, severity: read.severity, title: read.title},
      {word, severity, title}, `status_text(${status}, ${JSON.stringify(cause)})`,
    );
  }
  for (const [status, errorCategory, approval, short, long] of vectors.approval_tag) {
    const read = approvalTag(status, errorCategory, approval);
    const seen = read === null ? null : {short: read.short, long: read.long};
    const want = short === null || short === undefined ? null : {short, long};
    assert.deepEqual(seen, want, `approval_tag(${status}, ${errorCategory})`);
  }
});

test('sending the same text again keeps the key the store may already hold', async () => {
  const key = '33333333-3333-3333-3333-333333333333';
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => new Error('The connection dropped'),
    [`GET /sessions/a/submissions/${key}`]: {submission: {client_request_id: key, state: 'unknown', session_id: null, turn_id: null, status: null, cause: null, error: '', created_at: null, turn: null}},
  }));
  await h.run('init()');
  h.context.crypto = {randomUUID: () => key};
  h.element('input').value = 'Maybe landed';
  await h.run('send()');
  assert.equal(h.state().outbox.a.state, 'failed');
  h.element('input').value = 'Maybe landed';
  await h.run('send()');
  const keys = h.net.callsTo('/sessions/a/turns').map((call) => call.body.client_request_id);
  assert.equal(keys.length, 2);
  assert.equal(keys[1], keys[0], 'The same text asks the store about the key it already has');
});

test('cancel and keep draft puts the message back in the composer', async () => {
  const key = '44444444-4444-4444-4444-444444444444';
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => new Error('The connection dropped'),
    [`GET /sessions/a/submissions/${key}`]: {submission: {client_request_id: key, state: 'unknown', session_id: null, turn_id: null, status: null, cause: null, error: '', created_at: null, turn: null}},
    [`POST /sessions/a/submissions/${key}/cancel`]: {ok: true},
  }));
  await h.run('init()');
  h.context.crypto = {randomUUID: () => key};
  h.element('input').value = 'Keep me';
  await h.run('send()');
  assert.equal(h.element('input').value, 'Keep me', 'An unknown send keeps the text');
  h.element('input').value = '';
  await h.run('cancelSend("a", S.outbox.a)');
  assert.equal(h.state().outbox.a, undefined);
  assert.equal(h.element('input').value, 'Keep me', 'Cancel puts it back when the composer is empty');
});

test('a withdrawal the store made on its own keeps the message', async () => {
  const key = '55555555-5555-5555-5555-555555555555';
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => new Error('The connection dropped'),
    [`GET /sessions/a/submissions/${key}`]: {submission: {client_request_id: key, state: 'unknown', session_id: null, turn_id: null, status: null, cause: null, error: '', created_at: null, turn: null}},
    [`POST /sessions/a/submissions/${key}/cancel`]: {status: 409, body: {error: {code: 'submission_withdrawn', message: 'This message was cancelled before it started.'}}},
  }));
  await h.run('init()');
  h.context.crypto = {randomUUID: () => key};
  h.element('input').value = 'Too late';
  await h.run('send()');
  await h.run('cancelSend("a", S.outbox.a)');
  assert.equal(h.state().outbox.a, undefined);
  assert.equal(h.element('input').value, 'Too late');
});

test('a chat left before its load finishes keeps no stream', async () => {
  let release;
  const slow = new Promise((resolve) => {
    release = resolve;
  });
  const detail = {session: session('a'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0};
  const h = harness(baseRoutes({
    'GET /sessions/a': () => slow,
    'GET /sessions/a/turns?limit=10': () => slow,
    'GET /sessions/b': {session: session('b'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0},
    'GET /sessions/b/turns?limit=10': {turns: [], has_earlier: false, replay_from: 0},
  }));
  h.context.location.hash = '#/new';
  await h.run('init()');
  const first = h.run('openSession("a")');
  await h.run('openSession("b")');
  release(detail);
  await first;
  assert.equal(h.state().activeId, 'b');
  assert.equal(FakeEventSource.instances.length, 1, 'The abandoned load opens no second stream');
  assert.equal(h.stream().url, '/api/sessions/b/events?after=0');
});

test('a repeated send reloads the chat and reopens its stream', async () => {
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': {status: 200, body: {turn_id: 'turn-2', turn: turnRow({id: 'turn-2', turn: 2}), repeated: true}},
    'GET /sessions/a/turns?limit=10': {turns: [turnRow(), turnRow({id: 'turn-2', turn: 2, user_text: 'Only once'})], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  h.element('input').value = 'Only once';
  await h.run('send()');
  assert.equal(FakeEventSource.instances.length, 2, 'The rebuilt view reopens the stream from its cursor');
  assert.equal(h.stream().url, '/api/sessions/a/events?after=0');
});

test('a level change lands on the chat it was made for', async () => {
  let release;
  const answer = new Promise((resolve) => {
    release = resolve;
  });
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a'), session('b', {level: 3})], machine: null, has_more: false},
    'GET /sessions/b': {session: session('b', {level: 3}), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0},
    'GET /sessions/b/turns?limit=10': {turns: [], has_earlier: false, replay_from: 0},
    'POST /sessions/a/level': () => answer,
  }));
  await h.run('init()');
  const pending = h.run('setLevel(1)');
  await h.run('openSession("b", {saveCurrent: false})');
  release({session_id: 'a', level: 1});
  await pending;
  const a = h.state().sessions.find((item) => item.id === 'a');
  assert.equal(a.level, 1, 'The answer belongs to the chat that asked');
  assert.equal(h.state().view.session.level, 3, 'The chat now open keeps its own level');
});

test('a level change that fails goes back', async () => {
  const h = harness(baseRoutes({
    'POST /sessions/a/level': {status: 500, body: {error: {code: 'internal', message: 'the gate is down'}}},
  }));
  await h.run('init()');
  await h.run('setLevel(4)');
  assert.equal(h.state().sessions[0].level, 2);
  assert.match(h.element('notice').textContent, /Could not change the chat level: the gate is down/);
});

test('closing the last chat opens a new one and keeps its waiting draft', async () => {
  const h = harness(baseRoutes({'DELETE /sessions/a': {status: 204, body: {}}}));
  await h.run('init()');
  h.run('S.drafts.new = "precious new"');
  h.element('input').value = 'chat draft';
  await h.run('deleteSession("a")');
  assert.equal(h.state().activeId, null);
  assert.equal(h.element('welcome').hidden, false);
  // The deleted chat's text is not a new-chat draft; the waiting one is.
  assert.equal(h.state().drafts.new, 'precious new');
  assert.equal(h.element('input').value, 'precious new');
  assert.equal(
    JSON.parse(h.session.getItem('localmelo-console-drafts')).new,
    'precious new',
  );
});

test('a busy chat is not closed from the sidebar', async () => {
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a', {busy: true})], machine: null, has_more: false},
    'DELETE /sessions/a': {status: 204, body: {}},
  }));
  await h.run('init()');
  await h.run('deleteSession("a")');
  assert.equal(h.net.callsTo('/sessions/a').filter((call) => call.method === 'DELETE').length, 0);
  assert.equal(h.state().activeId, 'a');
});

test('a running tool card keeps its clock and its limit on the page', async () => {
  const running = turnRow({status: 'running', answer: null, finished_at: 0, running_here: true});
  const h = harness(baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [running], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  h.stream().emit({
    id: 20, turn: 2, ts: 50, type: 'turn.started',
    data: {turn: 2, turn_id: 'turn-2', user_text: 'Run it', client_request_id: 'k2'},
  });
  h.stream().emit({
    id: 21, turn: 2, ts: 51, type: 'tool.started',
    data: {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {}, preview: 'make test'},
  });
  assert.match(h.text('messages'), /Running · [\d:]+ of 10:00/);
  h.run('tick()');
  assert.match(h.text('messages'), /Running · [\d:]+ of 10:00/);
});

test('Show all names the raw character count', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().emit({
    id: 20, turn: 1, ts: 50, type: 'tool.started',
    data: {step: 1, call_id: 'call-1', name: 'shell_exec', arguments: {}, preview: 'cat big'},
  });
  h.stream().emit({
    id: 21, turn: 1, ts: 51, type: 'tool.finished',
    data: {call_id: 'call-1', name: 'shell_exec', status: 'success', error_category: 'none', output: 'x'.repeat(3000), output_chars: 3000, error: '', duration_s: 1, images: [], approval: null, exit_code: 0},
  });
  assert.match(h.text('messages'), /Show all 3000 characters/);
  assert.doesNotMatch(h.text('messages'), /Show all 3\.0k/);
});

test('a message the store does not know says its outcome is unknown', async () => {
  const key = '22222222-2222-2222-2222-222222222222';
  const h = harness(baseRoutes({
    'POST /sessions/a/turns': () => new Error('The connection dropped'),
    [`GET /sessions/a/submissions/${key}`]: {submission: {client_request_id: key, state: 'unknown', session_id: null, turn_id: null, status: null, cause: null, error: '', created_at: null, turn: null}},
  }));
  await h.run('init()');
  h.element('input').value = 'Lost one';
  h.context.crypto = {randomUUID: () => key};
  await h.run('send()');
  assert.equal(h.state().outbox.a.state, 'failed');
  assert.match(h.text('messages'), /Outcome unknown/);
  assert.doesNotMatch(h.text('messages'), /Not sent/);
  assert.equal(h.element('input').value, 'Lost one');
  assert.equal(JSON.parse(h.session.getItem('localmelo-console-drafts')).a, 'Lost one');
  // The message did land after all: its turn clears the draft the page kept.
  h.stream().emit({
    id: 30, turn: 2, ts: 100, type: 'turn.started',
    data: {turn: 2, turn_id: 'turn-2', user_text: 'Lost one', client_request_id: key},
  });
  assert.equal(h.state().outbox.a, undefined);
  assert.equal(h.element('input').value, '');
  assert.equal(h.state().drafts.a, undefined);
});

test('stop posts the running turn\'s id', async () => {
  const h = harness(baseRoutes({
    'POST /turns/turn-2/cancel': {status: 202, body: {turn_id: 'turn-2', cancelling: true}},
  }));
  await h.run('init()');
  h.stream().emit({
    id: 20, turn: 2, ts: 50, type: 'turn.started',
    data: {turn: 2, turn_id: 'turn-2', user_text: 'A long task', client_request_id: 'k2'},
  });
  assert.equal(h.element('stop-btn').hidden, false);
  assert.equal(h.element('send-btn').hidden, true);
  await h.run('stop()');
  assert.deepEqual(h.net.callsTo('/turns/turn-2/cancel').map((call) => call.method), ['POST']);
  assert.equal(h.state().cancelling.size, 0);
});

test('a running turn draws the live row with the terminal\'s facts', async () => {
  const running = turnRow({status: 'running', answer: null, finished_at: 0, running_here: true});
  const h = harness(baseRoutes({
    'GET /sessions/a/turns?limit=10': {turns: [running], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  assert.equal(h.stream().url, '/api/sessions/a/events?after=0');
  h.stream().emit({id: 21, turn: 1, ts: 51, type: 'step.started', data: {step: 1, attempt: 1}});
  h.stream().emit({id: 22, turn: 1, ts: 52, type: 'run.progress', data: {level: 1, state: 'running', step: 2, attempt: 1, phase: 'Verify', cycle: 0, tests: {passed: 3, failed: 0, total: 3}, items: null, past: {runs: 1, last_steps: 4, last_wall_s: 60}, basis: {match: 'exact'}}});
  const live = flatten(h.element('messages'));
  assert.match(live, /Thinking/);
  assert.match(live, /step 2/);
  assert.match(live, /Verify/);
  assert.match(live, /tests 3\/3/);
  assert.match(live, /Esc to stop/);
  assert.match(live, /last time 4 steps, 1m00s/, 'Turn 1 shows the past facts');
});

test('closing a chat deletes it and keeps the other chat\'s draft', async () => {
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a', {updated_at: 2}), session('b', {updated_at: 1})], machine: null, has_more: false},
    'DELETE /sessions/a': {status: 204, body: {}},
    'GET /sessions/b': {session: session('b'), pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0},
    'GET /sessions/b/turns?limit=10': {turns: [], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  h.element('input').value = 'Draft A';
  h.run('saveDraft()');
  await h.run('deleteSession("a")');
  assert.deepEqual(h.net.callsTo('/sessions/a').filter((call) => call.method === 'DELETE').map((call) => call.method), ['DELETE']);
  assert.equal(h.state().activeId, 'b');
  assert.equal(h.state().drafts.a, undefined);
  assert.equal(h.element('input').value, '');
});

test('a chat the store no longer has becomes a new chat with its draft', async () => {
  const missing = new Error('Session not found.');
  missing.status = 404;
  const h = harness(baseRoutes({
    'GET /sessions/a': () => missing,
  }));
  // A reload: the draft is in sessionStorage and the saved chat is gone.
  h.session.setItem('localmelo-console-drafts', JSON.stringify({new: 'Do not lose this'}));
  h.local.setItem('localmelo-console-view', JSON.stringify({id: 'a'}));
  await h.run('init()');
  assert.equal(h.state().activeId, null);
  assert.equal(h.state().sessions.length, 0);
  assert.equal(h.state().drafts.new, 'Do not lose this');
  assert.equal(h.element('input').value, 'Do not lose this');
  assert.match(h.element('notice').textContent, /expired or the gateway restarted/);
});

test('the page boots when the browser denies storage access', async () => {
  const h = harness(baseRoutes());
  for (const name of ['localStorage', 'sessionStorage']) {
    Object.defineProperty(h.context, name, {
      configurable: true,
      get() {
        throw new Error('Browser policy denied storage access');
      },
    });
  }
  await h.run('init()');
  assert.equal(h.state().activeId, 'a');
  assert.equal(h.state().online, true);
  h.element('input').value = 'A draft without browser storage';
  h.run('saveDraft(); syncUI()');
  assert.equal(h.state().drafts.a, 'A draft without browser storage');
  assert.equal(h.element('send-btn').disabled, false);
});

test('the level chip writes the level through /api', async () => {
  const h = harness(baseRoutes({
    'POST /sessions/a/level': {body: {session_id: 'a', level: 1}},
  }));
  await h.run('init()');
  h.element('level-btn').listeners.click[0]();
  assert.equal(h.element('level-menu').hidden, false);
  assert.equal(h.element('level-menu').children.length, 4);
  assert.equal(h.element('level-menu').children[1].attributes['aria-selected'], 'true');
  await h.element('level-menu').children[0].listeners.click[0]();
  const [call] = h.net.callsTo('/sessions/a/level');
  assert.equal(call.method, 'POST');
  assert.deepEqual(call.body, {level: 1});
  assert.equal(h.element('level-btn').textContent, 'P1');
  assert.equal(h.element('level-menu').hidden, true);
});

test('presence is throttled to one request per 2.5 seconds', async () => {
  const posts = [];
  const h = harness(baseRoutes());
  await h.run('init()');
  h.context.fetch = (path, options) => {
    posts.push([path, options]);
    return Promise.resolve({ok: true, status: 200, json: async () => ({ok: true})});
  };
  h.run('presence(true)');
  h.run('presence()');
  assert.equal(posts.length, 1, 'Two heartbeats in a row are one request');
  assert.equal(posts[0][0], '/api/presence');
  assert.equal(JSON.parse(posts[0][1].body).session_id, 'a');
  h.state().lastPresenceAt = Date.now() - 2400;
  h.run('presence()');
  assert.equal(posts.length, 1, 'Inside 2.5 s a typing heartbeat is throttled');
  h.state().lastPresenceAt = Date.now() - 2600;
  h.run('presence()');
  assert.equal(posts.length, 2, 'After 2.5 s a new heartbeat is sent');
});

test('the machine line shows a foreign lock, the gate mode and the battery', async () => {
  const machine = {
    lock: 'foreign', holder: 'agent-benchmark session (personal-agent-bench)',
    purpose: 'real-web UI eval', stale: false, readable: true,
    on_ac: true, battery_pct: 100, battery_state: 'charged', band: 4, gate: 'record',
  };
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a')], machine, has_more: false},
  }));
  await h.run('init()');
  const chip = h.element('machine-state');
  assert.equal(chip.hidden, false);
  assert.match(chip.textContent, /GPU lock: agent-benchmark session/);
  assert.match(chip.textContent, /\(recorded\)/);
  assert.match(chip.attributes['aria-label'], /real-web UI eval/);

  h.state().machine = {...machine, gate: 'wait'};
  h.run('syncUI()');
  assert.match(chip.textContent, /\(chats wait\)/);
  h.state().machine = {...machine, stale: true};
  h.run('syncUI()');
  assert.match(chip.textContent, /· ETA passed/);
  h.state().machine = {...machine, readable: false};
  h.run('syncUI()');
  assert.equal(chip.textContent, 'GPU lock unreadable');
  h.state().machine = {...machine, lock: 'none', on_ac: false, battery_pct: 55, battery_state: 'discharging'};
  h.run('syncUI()');
  assert.equal(chip.textContent, 'Battery 55% · discharging');
  h.state().machine = {...machine, lock: 'none', on_ac: true};
  h.run('syncUI()');
  assert.equal(chip.hidden, true, 'AC with no lock says nothing');
});

test('the header pill and the composer follow a session hold', async () => {
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [session('a', {busy: true, held: 'Resources'})], machine: null, has_more: false},
    'GET /sessions/a': {
      session: session('a', {busy: true, held: 'Resources'}),
      pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'open', tokens: 0,
    },
    'GET /sessions/a/turns?limit=10': {turns: [turnRow({status: 'running', answer: null, finished_at: 0, running_here: true})], has_earlier: false, replay_from: 0},
  }));
  await h.run('init()');
  assert.equal(h.element('connection-label').textContent, 'Waiting: Resources');
  assert.equal(h.element('run-status').textContent, 'Waiting: Resources');
  h.state().view.session.held = '';
  h.state().sessions[0].held = '';
  h.run('syncUI(); renderMessages(true)');
  assert.equal(h.element('connection-label').textContent, 'Working');
  assert.equal(h.element('run-status').textContent, 'Running on your backend');
});

test('the 4 s poll redraws the work group when the hold changes', async () => {
  const running = turnRow({status: 'running', answer: null, finished_at: 0, running_here: true});
  const routes = baseRoutes({
    'GET /sessions': {sessions: [session('a', {busy: true, held: ''})], machine: null, has_more: false},
    'GET /sessions/a': {
      session: session('a', {busy: true, held: ''}),
      pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'open', tokens: 0,
    },
    'GET /sessions/a/turns?limit=10': {turns: [running], has_earlier: false, replay_from: 0},
  });
  const h = harness(routes);
  await h.run('init()');
  assert.match(h.text('messages'), /Working/);
  routes['GET /sessions'] = {sessions: [session('a', {busy: true, held: 'Resources'})], machine: null, has_more: false};
  await h.run('refreshList()');
  assert.match(h.text('messages'), /Waiting: Resources/);
});

test('opening settings closes the trapped drawer first', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.drawer.open('tools');
  await h.element('settings-btn').click();
  assert.equal(h.drawer.opened, false);
});

test('Escape closes the drawer before stopping a running turn', async () => {
  const h = harness(baseRoutes({'POST /turns/turn-2/cancel': {status: 202, body: {turn_id: 'turn-2', cancelling: true}}}));
  await h.run('init()');
  h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'turn-2', user_text: 'A long task'}});
  h.drawer.open('tools');
  const [escape] = h.documentListeners.keydown;
  escape({key: 'Escape', isComposing: false, keyCode: 27});
  assert.equal(h.drawer.opened, false);
  assert.equal(h.net.callsTo('/turns/turn-2/cancel').length, 0);
  escape({key: 'Escape', isComposing: false, keyCode: 27});
  assert.equal(h.net.callsTo('/turns/turn-2/cancel').length, 1);
});

test('Escape closes the level menu before it stops the turn', async () => {
  const h = harness(baseRoutes({
    'POST /turns/turn-2/cancel': {status: 202, body: {turn_id: 'turn-2', cancelling: true}},
  }));
  await h.run('init()');
  h.stream().emit({
    id: 20, turn: 2, ts: 50, type: 'turn.started',
    data: {turn: 2, turn_id: 'turn-2', user_text: 'A long task', client_request_id: 'k2'},
  });
  h.run('showLevelMenu(true)');
  const [escape] = h.documentListeners.keydown;
  escape({key: 'Escape', isComposing: false, keyCode: 27});
  assert.equal(h.state().levelsOpen, false, 'The open menu closes first');
  assert.equal(h.net.callsTo('/turns/turn-2/cancel').length, 0);
  escape({key: 'Escape', isComposing: false, keyCode: 27});
  assert.equal(h.net.callsTo('/turns/turn-2/cancel').length, 1, 'A second Escape stops the turn');
});

test('a closed stream reopens from the last event id', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  assert.equal(h.stream().url, '/api/sessions/a/events?after=7');
  h.stream().emit({id: 12, turn: 1, ts: 30, type: 'turn.undone', data: {turn: 1, turn_id: 'turn-1'}});
  h.stream().fail();
  assert.equal(h.state().stream.status, 'paused');
  const timer = h.timers.find((entry) => !entry.interval && entry.ms === 1000 && !entry.cleared);
  assert.ok(timer, 'The first retry is one second away');
  await timer.handler();
  assert.equal(h.net.callsTo('/sessions/a').length, 2, 'The chat is checked before the stream reopens');
  assert.equal(FakeEventSource.instances.length, 2);
  assert.equal(h.stream().url, '/api/sessions/a/events?after=12');
});

test('the open chat\'s row follows the frames the page sees', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  const row = () => h.state().sessions.find((item) => item.id === 'a');
  assert.equal(row().busy, false);
  h.stream().emit({id: 20, turn: 2, ts: 50, type: 'turn.started', data: {turn: 2, turn_id: 'turn-2', user_text: 'Go', client_request_id: 'k2'}});
  assert.equal(row().busy, true, 'A turn that starts shows Working at once');
  h.stream().emit({id: 21, turn: 2, ts: 51, type: 'approval.requested', data: {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', arguments: {}, preview: 'ls', workspace_root: '/w'}});
  assert.equal(row().pending_approvals, 1, 'A waiting approval shows Approval needed');
  h.stream().emit({id: 22, turn: 2, ts: 52, type: 'approval.resolved', data: {request_id: 'req-1', call_id: 'call-1', tool_name: 'shell_exec', outcome: 'allowed', origin: 'user', reason: '', remember: false}});
  assert.equal(row().pending_approvals, 0);
  h.stream().emit({id: 23, turn: 2, ts: 53, type: 'turn.finished', data: {turn: 2, turn_id: 'turn-2', status: 'completed', error: '', cause: 'run'}});
  assert.equal(row().busy, false, 'The row stops saying Working when the turn ends');
});

test('a renamed session reaches its sidebar row without the poll', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().emit({id: 30, turn: 1, ts: 50, type: 'session.title', data: {title: 'A better title'}});
  const info = h.element('session-list').children[0].children[0];
  assert.match(info.innerHTML, /A better title/);
  assert.equal(h.element('header-title').textContent, 'A better title');
});

test('the paused-stream banner does not overwrite an error', async () => {
  const h = harness(baseRoutes());
  await h.run('init()');
  h.stream().fail();
  await h.run('notify("Could not stop the response: boom", {sticky: true})');
  h.run('tick()');
  assert.match(h.element('notice').textContent, /Could not stop the response/);
});

test('a stream paused on a deleted chat starts a new one instead of reopening', async () => {
  const routes = baseRoutes();
  const h = harness(routes);
  await h.run('init()');
  h.stream().fail();
  routes['GET /sessions/a'] = {status: 404, body: {error: {code: 'not_found', message: 'Session not found.'}}};
  const timer = h.timers.find((entry) => !entry.interval && entry.ms === 1000 && !entry.cleared);
  await timer.handler();
  assert.equal(FakeEventSource.instances.length, 1, 'No stream reopens on a chat that is gone');
  assert.equal(h.state().activeId, null);
  assert.match(h.element('notice').textContent, /This chat expired or the gateway restarted/);
  assert.equal(h.element('welcome').hidden, false);
});

test('worker detail is read-only and isolates the chat draft and stream', async () => {
  const h = harness(baseRoutes({
    'GET /sessions/w': {session: session('w', {kind: 'worker', title: 'Worker', source: 'dsh', status: 'running', steps: 0, tokens: null})},
    'GET /sessions/w/turns?limit=50&events=true': {turns: [turnRow({user_text: 'Worker brief', answer: 'Worker answer', events: [
      {id: 1, turn: 1, type: 'approval.requested', data: {request_id: 'old', call_id: 'call', tool_name: 'file_write'}},
    ]})], has_earlier: false},
  }));
  await h.run('init()');
  const stream = h.stream();
  h.element('input').value = 'Chat draft';
  h.run('showWorkers("w")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(stream.closed, true);
  assert.equal(h.state().activeId, 'a');
  assert.equal(h.state().drafts.a, 'Chat draft');
  assert.equal(h.element('composer-area').hidden, true);
  assert.equal(h.element('details-trigger').hidden, true);
  assert.match(h.text('worker-panel'), /Worker brief/);
  assert.match(h.text('worker-panel'), /Not recorded/);
  assert.match(h.text('worker-panel'), /New steps appear as they are recorded/);
  assert.equal(findNode(h.element('worker-panel'), (node) => node.classList.contains('approval-card')), null);
  assert.equal(findNode(h.element('worker-panel'), (node) => node.classList.contains('live-row')), null);
  await h.run('send()');
  await h.run('stop()');
  h.run('presence(true)');
  h.element('details-trigger').click();
  await h.run('setLevel(3)');
  await h.run('deleteSession("a")');
  assert.equal(h.element('connection-btn').hidden, true);
  const closeChat = findNode(h.element('session-list'), (node) => node.classList.contains('session-close'));
  assert.ok(closeChat.disabled || closeChat.hidden);
  assert.equal(h.drawer.calls.length, 0);
  assert.equal(h.net.calls.filter((call) => call.method !== 'GET' && call.path !== '/presence').length, 0);
  assert.equal(FakeEventSource.instances.length, 1);
});

test('worker images use the worker session id through detail rendering', async () => {
  const h = harness(baseRoutes({
    'GET /sessions/worker-1': {session: session('worker-1', {kind:'worker'})},
    'GET /sessions/worker-1/turns?limit=50&events=true': {turns:[turnRow({events:[
      {id:1, turn:1, type:'tool.started', data:{call_id:'c', name:'image', arguments:{}}},
      {id:2, turn:1, type:'tool.finished', data:{call_id:'c', name:'image', status:'success', images:[{sha256:'abc', mime_type:'image/png', size_bytes:1}]}},
    ]})]},
  }));
  await h.run('init()'); h.run('showWorkers("worker-1")');
  await new Promise((resolve) => setImmediate(resolve));
  const image = findNode(h.element('worker-panel'), (node) => node.tag === 'img');
  assert.equal(image.src, '/api/sessions/worker-1/blobs/abc');
});

test('worker list uses server filters without opening worker streams', async () => {
  const h = harness(baseRoutes({'GET /sessions?kind=worker&limit=50': {sessions: [session('w', {kind: 'worker', source: 'dsh', status: 'completed', duration_s: 0.013850290997652337})], next_cursor: null}}));
  await h.run('init()');
  h.run('showWorkers("")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.text('worker-panel'), /Workers/);
  assert.match(h.text('worker-panel'), /dsh/);
  assert.match(h.text('worker-panel'), /Duration: 0\.0s/);
  const apply = findNode(h.element('worker-panel'), (node) => node.textContent === 'Apply');
  assert.equal(apply.classList.contains('primary-button'), true);
  const css = readFileSync(new URL('../../support/gateway/static/workspace.css', import.meta.url), 'utf8');
  assert.match(css, /\.worker-filters button,[\s\S]*?height: 40px/);
  assert.match(css, /\.worker-panel a:not\(\.worker-row\):visited\s*\{\s*color: var\(--accent\)/);
  assert.match(css, /\.worker-panel\s*\{[^}]*overflow: auto;[^}]*overflow-wrap: anywhere;/);
  assert.match(css, /@media \(max-width: 700px\)\s*\{\s*\.worker-panel\s*\{/);
  assert.equal(h.net.callsTo('/sessions?kind=worker&limit=50').length, 1);
  assert.equal(FakeEventSource.instances.length, 1);
});

test('worker back returns to the list header and Chats resumes the open chat', async () => {
  const h = harness(baseRoutes({
    'GET /sessions?kind=worker&limit=50': {sessions: [session('w', {kind: 'worker'})], next_cursor: null},
    'GET /sessions/w': {session: session('w', {kind: 'worker', title: 'Worker title'})},
    'GET /sessions/w/turns?limit=50&events=true': {turns: [], has_earlier: false},
  }));
  await h.run('init()');
  h.run('showWorkers("")');
  await new Promise((resolve) => setImmediate(resolve));
  h.run('showWorkers("w")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(h.element('header-title').textContent, 'Worker title');
  h.run('showWorkers("")');
  assert.equal(h.element('header-title').textContent, 'Workers');
  assert.equal(h.element('chats-link').href, '#/s/a');
  h.context.location.hash = h.element('chats-link').href;
  await h.windowListeners.hashchange[0]();
  assert.equal(h.state().activeId, 'a');
  assert.equal(h.element('connection-btn').hidden, false);
  assert.equal(h.stream().closed, false);
});

test('an open unfinished worker run follows its stream, appends without a re-read, and stops when finished', async () => {
  let status = 'running';
  const row = () => ({turns: [turnRow({user_text: 'Live brief', status, events: [
    {id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Live brief'}},
  ]})], has_earlier: false});
  const h = harness(baseRoutes({
    'GET /sessions/w': () => ({session: session('w', {kind: 'worker', title: 'Live', source: 'localmelo', status, steps: 0, tokens: null, busy: status === 'running'})}),
    'GET /sessions/w/turns?limit=50&events=true': row,
    'GET /sessions/w/status': {session_id: 'w', segments: ['cache 40% idle 5s', 'step 1']},
  }));
  await h.run('init()');
  h.run('showWorkers("w")'); await settle();
  const panel = h.element('worker-panel');
  const stream = FakeEventSource.instances.at(-1);
  assert.equal(stream.url, '/api/sessions/w/events?after=1', 'the stream starts after the last event read');
  assert.match(flatten(panel), /Run not finished/);
  assert.match(flatten(panel), /cache 40% idle 5s/, 'the status row shows on a worker run');
  const reads = () => h.net.calls.filter((call) => call.path.startsWith('/sessions/w') && !call.path.endsWith('/status')).length;
  const before = reads();
  stream.emit({id: 2, turn: 1, type: 'tool.started', ts: 1, data: {step: 1, call_id: 'c1', name: 'file_read', arguments: {}, preview: 'a.txt'}});
  stream.emit({id: 3, turn: 1, type: 'note', ts: 2, data: {step: 1, kind: 'plan', text: 'Read the file first'}});
  stream.emit({id: 3, turn: 1, type: 'note', ts: 2, data: {step: 1, kind: 'plan', text: 'Read the file first'}});
  const draw = h.timers.findLast((timer) => !timer.interval && timer.ms === 150 && !timer.cleared);
  assert.ok(draw, 'frames are drawn together on one timer');
  draw.handler();
  assert.match(flatten(panel), /file_read/);
  assert.match(flatten(panel), /Note \(plan\): Read the file first/);
  assert.equal(reads(), before, 'new events are folded in without reading the run again');
  status = 'completed';
  stream.fail();
  await settle();
  assert.equal(stream.closed, true);
  assert.doesNotMatch(flatten(panel), /Run not finished/);
  assert.equal(FakeEventSource.instances.filter((source) => source.url.includes('/w/events')).length, 1, 'a finished run is not followed again');
  h.run('leaveWorkers()');
});

test('the open worker list shows a run that starts later', async () => {
  let rows = [{id: 'old', kind: 'worker', title: 'Old run', source: 'localmelo', status: 'completed'}];
  const h = harness(baseRoutes({'GET /sessions?kind=worker&limit=50': () => ({sessions: rows, next_cursor: null})}));
  await h.run('init()');
  h.run('showWorkers("")'); await settle();
  const panel = h.element('worker-panel');
  const poll = h.timers.findLast((timer) => timer.interval && timer.ms === 3000 && !timer.cleared);
  assert.ok(poll, 'the Workers list polls every 3 seconds');
  const before = panel.children[0];
  await poll.handler(); await settle();
  assert.equal(panel.children[0], before, 'an unchanged list is not redrawn');
  rows = [{id: 'new', kind: 'worker', title: 'New run', source: 'localmelo', status: 'running'}, ...rows];
  await poll.handler(); await settle();
  assert.match(flatten(panel), /New run/);
  assert.match(flatten(panel), /localmelo · [^·]* · running/);
});

test('failed worker filters remain an error when navigating away and back', async () => {
  const routes = baseRoutes({'GET /sessions?kind=worker&limit=50': {sessions: [], next_cursor: null},
    'GET /sessions?kind=worker&limit=50&source=dsh': {status: 503, body: {error: {message: 'unavailable'}}}});
  const h = harness(routes);
  await h.run('init()'); h.run('showWorkers("")');
  await new Promise((resolve) => setImmediate(resolve));
  await h.run('workers.state.filters.source = "dsh"; workers.list()');
  assert.match(h.text('worker-panel'), /Worker history is unavailable/);
  h.run('leaveWorkers()'); h.run('showWorkers("")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.text('worker-panel'), /Worker history is unavailable/);
  assert.doesNotMatch(h.text('worker-panel'), /No runs match these filters/);
});

test('worker metadata and errors preserve missing, zero and recorded values', async () => {
  const h = harness(baseRoutes({
    'GET /sessions?kind=worker&limit=50': {sessions: [session('w', {kind:'worker', source:'dsh', model:'', status:'crashed'})], next_cursor:null},
    'GET /sessions/w': {session: session('w', {kind:'worker', status:'crashed', steps:0, tokens:0, duration_s:0})},
    'GET /sessions/w/turns?limit=50&events=true': {turns:[turnRow({status:'crashed', result:{error:'Recorded error\nextra', duration_s:99}, events:[]})]},
  }));
  await h.run('init()'); h.run('showWorkers("")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.text('worker-panel'), /dsh · Not recorded · crashed/);
  h.run('showWorkers("w")'); await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.text('worker-panel'), /Steps 0 Tokens 0/);
  assert.match(h.text('worker-panel'), /Duration 0\.0s/);
  assert.match(h.text('worker-panel'), /Recorded error/);
});

test('worker filter and paging controls cover more than 200 records', async () => {
  const routes = baseRoutes();
  for (let page = 0; page < 5; page++) {
    const path = `/sessions?kind=worker&limit=50${page ? `&cursor=c${page}` : ''}`;
    routes[`GET ${path}`] = {sessions: Array.from({length:50}, (_, i) => session(`w${page*50+i}`, {kind:'worker'})), next_cursor: page < 4 ? `c${page+1}` : null};
  }
  routes['GET /sessions?kind=worker&limit=50&source=dsh&parent_run=parent'] = {sessions: [session('filtered', {kind:'worker', source:'dsh'})], next_cursor:null};
  const h = harness(routes); await h.run('init()'); h.run('showWorkers("")');
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  await settle();
  for (let page = 0; page < 4; page++) {
    const more = findNode(h.element('worker-panel'), (node) => node.textContent === 'Load more');
    assert.ok(more); more.click(); await settle();
  }
  assert.equal(h.run('workers.state.rows.length'), 250);
  let form = findNode(h.element('worker-panel'), (node) => node.tag === 'form');
  findNode(form, (node) => node.tag === 'select' && node.name === 'source').value = 'dsh';
  findNode(form, (node) => node.tag === 'input').value = 'parent';
  form.listeners.submit[0]({preventDefault() {}}); await settle();
  assert.equal(h.net.callsTo('/sessions?kind=worker&limit=50&source=dsh&parent_run=parent').length, 1);
  assert.equal(h.run('workers.state.rows.length'), 1);
  findNode(h.element('worker-panel'), (node) => node.textContent === 'Clear filters').click(); await settle();
  assert.equal(h.run('workers.state.rows.length'), 50);
});

test('worker routes and shortcuts cannot dispatch chat writes', async () => {
  const h = harness(baseRoutes({'GET /sessions?kind=worker&limit=50': {sessions:[], next_cursor:null}}));
  await h.run('init()');
  const stream = h.stream(); h.context.location.hash = '#/workers';
  await h.windowListeners.hashchange[0](); await new Promise((resolve) => setImmediate(resolve));
  assert.equal(stream.closed, true);
  const count = h.net.calls.length;
  h.element('input').listeners.keydown[0]({key:'Enter', shiftKey:false, isComposing:false, keyCode:13, preventDefault() {}});
  for (const key of ['Enter', 'Escape', 'k']) {
    h.documentListeners.keydown[0]({key, metaKey:key === 'k', ctrlKey:false, preventDefault() {}});
  }
  assert.equal(h.net.calls.slice(count).filter((call) => call.method !== 'GET').length, 0);
  h.context.location.hash = '#/s/a'; await h.windowListeners.hashchange[0]();
  assert.equal(h.state().activeId, 'a'); assert.equal(h.stream().closed, false);
});

test('worker title and answer use safe rendering', async () => {
  const hostile = '<img src=x onerror=alert(1)>';
  const h = harness(baseRoutes({
    'GET /sessions/w': {session: session('w', {kind:'worker', title:hostile})},
    'GET /sessions/w/turns?limit=50&events=true': {turns:[turnRow({answer:hostile, events:[]})]},
  }));
  await h.run('init()'); h.run('showWorkers("w")');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(findNode(h.element('worker-panel'), (node) => node.tag === 'h1').textContent, hostile);
  assert.equal(findNode(h.element('worker-panel'), (node) => node.tag === 'img'), null);
  assert.doesNotMatch(findNode(h.element('worker-panel'), (node) => node.classList.contains('answer')).innerHTML, /<img/i);
});

// ── The task view: card, Resume, Stop ──

function taskRow(id, extra = {}) {
  return {
    task_id: id, title: `Task ${id}`, state: 'parked', level: 2, level_eff: 2, wait_reason: '', card: 'Plan: step 1 done.',
    spec: {title: `Task ${id}`, objective: `Objective of ${id}`, workspace: ''}, started_at: 5, updated_at: 100,
    workspace_label: '~/w', ...extra,
  };
}
const settle = () => new Promise((resolve) => setImmediate(resolve));
const labelled = (h, text) => findNode(h.element('task-panel'), (node) => node.tag === 'button' && node.textContent === text);
const taskError = (status, code, message, details) => ({status, body: {error: {code, message, details}}});

async function openTask(routes, id = 't1') {
  const h = harness(baseRoutes(routes));
  await h.run('init()');
  h.context.location.hash = `#/tasks/${id}`;
  await h.windowListeners.hashchange[0](); await settle();
  return h;
}

test('the task list shows state, priority, title and update time, newest first, as text', async () => {
  const hostile = '<img src=x onerror=alert(1)>';
  const h = harness(baseRoutes({'GET /tasks': {tasks: [
    taskRow('old', {updated_at: 10}), taskRow('new', {updated_at: 90, title: hostile, state: 'queued', level_eff: 3}),
  ]}}));
  await h.run('init()');
  h.context.location.hash = '#/tasks';
  await h.windowListeners.hashchange[0](); await settle();
  const panel = h.element('task-panel');
  assert.equal(panel.hidden, false);
  assert.equal(h.element('scroll').hidden, true);
  assert.equal(h.element('tasks-link').attributes['aria-current'], 'page');
  const rows = panel.children.filter((node) => node.className.includes('task-row'));
  assert.deepEqual(rows.map((row) => row.href), ['#/tasks/new', '#/tasks/old']);
  assert.match(flatten(rows[0]), /queued · P3 · updated/);
  assert.equal(findNode(panel, (node) => node.tag === 'img'), null);
  assert.equal(rows[0].children[0].textContent, hostile);
  h.context.location.hash = '#/s/a'; await h.windowListeners.hashchange[0]();
  assert.equal(panel.hidden, true);
  assert.equal(h.element('scroll').hidden, false);
  assert.equal(h.element('tasks-link').attributes['aria-current'], undefined);
});

test('a card that needs confirmation asks first, and Cancel sends nothing', async () => {
  const posts = [];
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1', {card: '<b>not html</b>'})},
    'POST /tasks/t1/resume': (body) => {
      posts.push(body);
      return taskError(409, 'confirmation_required', 'warn', {confirmation_token: 'tok-1', reason: 'unverified_processes', expires_in_s: 300, warning: 'Server warning text.'});
    },
  });
  const panel = h.element('task-panel');
  const pre = findNode(panel, (node) => node.tag === 'pre');
  assert.equal(pre.textContent, '<b>not html</b>');
  assert.equal(findNode(panel, (node) => node.tag === 'b'), null);
  assert.match(flatten(panel), /Evidence Checked when you press Resume/);
  labelled(h, 'Resume').click(); await settle();
  const dialog = findNode(panel, (node) => node.tag === 'dialog');
  assert.equal(dialog.open, true);
  assert.equal(h.state().authenticated, true);
  assert.match(flatten(dialog), /Server warning text\./);
  assert.ok(labelled(h, 'Resume anyway') && labelled(h, 'Cancel'));
  assert.equal(documentDouble.activeElement.textContent, 'Cancel', 'focus starts on the safe choice');
  assert.match(flatten(panel), /confirmation is required to resume/);
  assert.deepEqual(posts.map((body) => body.confirmation_token), [undefined]);
  labelled(h, 'Cancel').click();
  assert.equal(findNode(panel, (node) => node.tag === 'dialog'), null);
  assert.equal(posts.length, 1, 'cancel sends nothing');
});

test('Resume anyway retries the same operation with the token, once, and shows the new state', async () => {
  const posts = [];
  let release;
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1', {state: 'interrupted'})},
    'POST /tasks/t1/resume': (body) => {
      posts.push(body);
      if (!body.confirmation_token) return taskError(409, 'confirmation_required', 'warn', {confirmation_token: 'tok-1', warning: ''});
      return new Promise((resolve) => { release = () => resolve({status: 202, body: {task: taskRow('t1', {state: 'queued'}), operation_id: body.operation_id}}); });
    },
  });
  const panel = h.element('task-panel');
  labelled(h, 'Resume').click(); await settle();
  assert.match(flatten(panel), /A process from before the restart may still be running\. Resuming may repeat effects\. Confirmation does not prove the process stopped or effects happened only once\./);
  labelled(h, 'Resume anyway').click(); await settle();
  const inDialog = () => findNode(findNode(panel, (node) => node.tag === 'dialog'), (node) => node.tag === 'button' && node.textContent === 'Resuming…');
  assert.equal(inDialog().disabled, true, 'disabled while in flight');
  inDialog().click(); await settle();
  assert.equal(posts.length, 2, 'a second click while in flight sends nothing');
  assert.equal(posts[1].operation_id, posts[0].operation_id);
  assert.equal(posts[1].confirmation_token, 'tok-1');
  release(); await settle();
  assert.equal(findNode(panel, (node) => node.tag === 'dialog'), null);
  assert.match(flatten(panel), /Resume accepted\. The task is queued\./);
  assert.equal(labelled(h, 'Resume'), null, 'a queued task is not resumable');
  assert.ok(labelled(h, 'Stop'));
});

test('a stale confirmation shows the API text and the next Resume starts a new operation', async () => {
  const posts = [];
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1')},
    'POST /tasks/t1/resume': (body) => {
      posts.push(body);
      return body.confirmation_token
        ? taskError(409, 'stale_confirmation', 'The recovery evidence changed.')
        : taskError(409, 'confirmation_required', 'warn', {confirmation_token: `tok-${posts.length}`});
    },
  });
  labelled(h, 'Resume').click(); await settle();
  labelled(h, 'Resume anyway').click(); await settle();
  const panel = h.element('task-panel');
  assert.match(flatten(panel), /The recovery evidence changed\./);
  assert.equal(findNode(panel, (node) => node.tag === 'dialog'), null);
  labelled(h, 'Resume').click(); await settle();
  assert.notEqual(posts[2].operation_id, posts[0].operation_id);
  assert.equal(posts[2].confirmation_token, undefined, 'no authorization is carried over');
});

test('Escape closes the dialog without sending anything and the page does not stop a chat', async () => {
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1')},
    'POST /tasks/t1/resume': taskError(409, 'confirmation_required', 'warn', {confirmation_token: 'tok-1'}),
  });
  labelled(h, 'Resume').click(); await settle();
  const panel = h.element('task-panel');
  const dialog = findNode(panel, (node) => node.tag === 'dialog');
  const count = h.net.calls.length;
  dialog.listeners.keydown[0]({key: 'Escape', preventDefault() {}});
  assert.equal(findNode(panel, (node) => node.tag === 'dialog'), null);
  h.documentListeners.keydown[0]({key: 'Escape', preventDefault() {}});
  assert.equal(h.net.calls.length, count);
});

test('a missing recovery card offers a new task, and a retry reuses its request key', async () => {
  const posts = [];
  let fail = true;
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1', {card: '', state: 'interrupted'})},
    'POST /tasks': (body) => {
      posts.push(body);
      if (fail) return new Error('network down');
      return {status: 202, body: {task: taskRow('t2', {state: 'queued'})}};
    },
  });
  const panel = h.element('task-panel');
  assert.match(flatten(panel), /Resume unavailable: recovery information is missing\./);
  assert.match(flatten(panel), /Earlier effects of this task are unknown/);
  assert.equal(labelled(h, 'Resume'), null);
  labelled(h, 'Start again as a new task').click(); await settle();
  assert.equal(posts[0].objective, 'Objective of t1');
  fail = false;
  labelled(h, 'Start again as a new task').click(); await settle();
  assert.equal(posts[1].request_key, posts[0].request_key, 'a retry after a network failure is the same request');
  assert.equal(h.context.location.hash, '#/tasks/t2');
});

test('a task that never started and has no card can still be resumed', async () => {
  const h = await openTask({'GET /tasks/t1': {task: taskRow('t1', {card: '', started_at: null})}});
  assert.ok(labelled(h, 'Resume'));
  assert.equal(labelled(h, 'Start again as a new task'), null);
});

test('a missing_card refusal from the server switches the card to the new-task path', async () => {
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1', {card: '', started_at: null})},
    'POST /tasks/t1/resume': taskError(409, 'invalid_transition', 'Resume unavailable: recovery information is missing. Start again as a new task.', {reason: 'missing_card'}),
  });
  labelled(h, 'Resume').click(); await settle();
  assert.equal(labelled(h, 'Resume'), null);
  assert.ok(labelled(h, 'Start again as a new task'));
});

test('Stop uses the task API, disables while in flight, and shows a refusal', async () => {
  const calls = [];
  let release;
  const h = await openTask({
    'GET /tasks/t1': {task: taskRow('t1', {state: 'running'})},
    'PATCH /tasks/t1': (body) => {
      calls.push(body);
      return calls.length === 1
        ? new Promise((resolve) => { release = () => resolve(taskError(409, 'invalid_transition', 'The task already finished.')); })
        : {task: taskRow('t1', {state: 'cancelled'})};
    },
  });
  assert.equal(labelled(h, 'Resume'), null);
  labelled(h, 'Stop').click(); await settle();
  assert.equal(labelled(h, 'Stopping…').disabled, true);
  labelled(h, 'Stopping…').click(); await settle();
  assert.equal(calls.length, 1);
  release(); await settle();
  assert.match(flatten(h.element('task-panel')), /The task already finished\./);
  labelled(h, 'Stop').click(); await settle();
  assert.deepEqual(calls, [{state: 'stop'}, {state: 'stop'}]);
  assert.match(flatten(h.element('task-panel')), /Stop accepted\. The task is cancelled\./);
  assert.equal(labelled(h, 'Stop'), null, 'a stopped task has no actions');
  assert.equal(h.net.calls.filter((call) => call.path.startsWith('/sessions') && call.method !== 'GET').length, 0, 'never the chat cancel');
});

test('a late answer for a task the user left is dropped', async () => {
  let release;
  const h = harness(baseRoutes({'GET /tasks/t1': () => new Promise((resolve) => { release = () => resolve({task: taskRow('t1')}); })}));
  await h.run('init()');
  h.context.location.hash = '#/tasks/t1';
  const pending = h.windowListeners.hashchange[0](); await settle();
  h.context.location.hash = '#/s/a'; await h.windowListeners.hashchange[0]();
  release(); await pending; await settle();
  assert.equal(h.element('task-panel').children.length, 0);
  assert.equal(h.element('task-panel').hidden, true);
});

test('the Tasks header names Tasks, not the open chat, and the chat comes back', async () => {
  const chat = session('a', {title: 'Chat about novels', model: 'Qwen-7B'});
  const h = harness(baseRoutes({
    'GET /sessions': {sessions: [chat], machine: null, has_more: false},
    'GET /sessions/a': {session: chat, pending_approvals: [], grants: [], granted_at: {}, mcp_error: '', agent: 'not_loaded', tokens: 0},
    'GET /tasks': {tasks: [taskRow('t1')]},
    'GET /tasks/t1': {task: taskRow('t1')},
  }));
  await h.run('init()');
  assert.equal(h.element('header-title').textContent, 'Chat about novels');
  h.context.location.hash = '#/tasks';
  await h.windowListeners.hashchange[0](); await settle();
  assert.equal(h.element('header-title').textContent, 'Tasks');
  assert.equal(h.element('header-model').textContent, 'Scheduler tasks');
  assert.equal(h.element('task-panel').hidden, false);
  h.context.location.hash = '#/tasks/t1';
  await h.windowListeners.hashchange[0](); await settle();
  assert.equal(h.element('header-title').textContent, 'Tasks');
  assert.equal(h.element('header-model').textContent, 'Scheduler tasks');
  h.context.location.hash = '#/s/a';
  await h.windowListeners.hashchange[0](); await settle();
  assert.equal(h.element('header-title').textContent, 'Chat about novels');
  assert.equal(h.element('header-model').textContent, 'Qwen-7B');
});
test('Tasks hands the header to Workers and back without the chat', async () => {
  const h = harness(baseRoutes({'GET /tasks': {tasks: [taskRow('t1')]}, 'GET /sessions?kind=worker&limit=50': {sessions: [], next_cursor: null}}));
  await h.run('init()');
  h.context.location.hash = '#/tasks';
  await h.windowListeners.hashchange[0](); await settle();
  h.context.location.hash = '#/workers';
  await h.windowListeners.hashchange[0](); await settle();
  assert.equal(h.element('header-title').textContent, 'Workers');
  h.context.location.hash = '#/tasks';
  await h.windowListeners.hashchange[0](); await settle();
  assert.equal(h.element('header-title').textContent, 'Tasks');
});

test('an open task card picks up an approval that arrives later, and redraws only on a change', async () => {
  const request = {
    request_id: 'req-late', tool_name: 'shell_exec', arguments: {command: 'make test'}, arguments_complete: true,
    workspace_label: '~/w', target: null,
  };
  let pending = [];
  const h = await openTask({
    'GET /tasks/t1': () => ({task: taskRow('t1', {state: 'held', wait_reason: 'approval', pending_approvals: pending})}),
  });
  const panel = h.element('task-panel');
  assert.equal(findNode(panel, (node) => node.classList.contains('approval-card')), null);
  const poll = h.timers.find((timer) => timer.interval && timer.ms === 3000);
  assert.ok(poll, 'the Tasks view polls every 3 seconds');
  const before = panel.children[0];
  await poll.handler(); await settle();
  assert.equal(panel.children[0], before, 'an unchanged task is not redrawn');
  pending = [request];
  await poll.handler(); await settle();
  assert.equal(findNode(panel, (node) => node.classList.contains('approval-card'))?.id, 'approval-card-req-late');
  assert.match(flatten(panel), /Waiting for your approval/);
  assert.equal(findNode(panel, (node) => node.textContent === 'Refresh'), null, 'the card needs no Refresh button');
});

test('a task card shows its status row under the facts and bumps it on each poll', async () => {
  let segments = ['cache 50%', 'tasks 1 running 0 queued'];
  const h = await openTask({
    'GET /tasks/t1': () => ({task: taskRow('t1', {state: 'running'})}),
    'GET /sessions/t1/status': () => ({session_id: 't1', segments}),
  });
  await settle();
  const panel = h.element('task-panel');
  const row = findNode(panel, (node) => node.classList.contains('status-row'));
  assert.deepEqual(row.children.map((child) => child.textContent), segments);
  assert.equal(panel.children.indexOf(row), panel.children.findIndex((node) => node.tagName === 'DL') + 1);
  const poll = h.timers.find((timer) => timer.interval && timer.ms === 3000);
  segments = ['cache --'];
  await new Promise((resolve) => setTimeout(resolve, 1100));
  await poll.handler(); await settle();
  assert.deepEqual(row.children.map((child) => child.textContent), ['cache --']);
});

test('a task waiting for approval shows the chat approval card and answers it', async () => {
  const request = {
    request_id: 'req-t', tool_name: 'shell_exec', arguments: {command: 'make test'}, arguments_complete: true,
    workspace_label: '~/w', target: null,
  };
  let reads = 0;
  const h = await openTask({
    'GET /tasks/t1': () => ({task: taskRow('t1', {state: 'held', wait_reason: 'approval', pending_approvals: reads++ ? [] : [request]})}),
    'POST /approvals/req-t': {outcome: 'allowed'},
  });
  const panel = h.element('task-panel');
  const card = findNode(panel, (node) => node.classList.contains('approval-card'));
  assert.equal(card.id, 'approval-card-req-t');
  assert.match(flatten(panel), /Waiting for your approval/);
  assert.match(flatten(panel), /shell_exec wants to run a command in ~\/w/);
  const once = findNode(card, (node) => node.textContent === 'Allow once');
  h.run('approvalStates.get("req-t").mountedAt -= 1000');
  h.timers.findLast((timer) => timer.ms <= 500).handler();
  await once.click(); await settle();
  assert.deepEqual(h.net.callsTo('/approvals/req-t')[0].body, {choice: 'once'});
  assert.equal(findNode(h.element('task-panel'), (node) => node.classList.contains('approval-card')), null, 'the answered card goes away');
  assert.doesNotMatch(flatten(h.element('task-panel')), /Waiting for your approval/);
});
