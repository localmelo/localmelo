/**
 * What each slash command does in the console: one function per page action
 * (`GET /api/commands` names the action of every command). An action answers with the lines of a
 * local row, nothing when the page itself shows the result, or throws an `Error` whose message
 * the row shows as a failure. Nothing here reaches the model.
 *
 * `host` is the page (`workspace.js`): `api`, `state()`, and the moves it makes: `send`,
 * `newChat`, `pickChat`, `openChat`, `openMain`, `openSettings`, `openMemory`, `settingsSaved`,
 * `setMode`, `setTitle`, `setNewTitle`, `view`, `reveal`, `writeText`, `isApp`, `canQuit`, `quit`
 * and, once the import screens land, `memoryImport`.
 */

import {saidText, t} from './i18n.mjs';
import {formatDuration, formatTokens, shownItems, turnDuration} from './turns.mjs';

const enc = encodeURIComponent;
const say = (text, tone) => ({text, tone});
const first = (text) => text.split(/\s+/).join(' ').slice(0, 60);
// A web address as the row shows it: without the user name and password it may carry.
const plainUrl = (url) => url.replace(/\/\/[^/@\s]*@/, '//');
// The option a command takes: on, off, or nothing for the other one.
const toggled = (word, now) => ({on: true, off: false, '': !now})[word];

/** The tool calls of the turns, numbered as `/expand turn.call` names them. */
export function toolCalls(turns) {
  return turns.filter((turn) => !turn.undone).flatMap((turn) => shownItems(turn)
    .filter((item) => item.kind === 'tool')
    .map((item, index) => ({ref: `${turn.turn}.${index + 1}`, turn: turn.turn, number: index + 1, item})));
}

export function createActions(host) {
  const {api, state} = host;

  const chat = () => {
    const {id} = state();
    if (!id) throw new Error(t('slash.noChatYet'));
    return `/sessions/${enc(id)}`;
  };
  const newest = () => state().turns.filter((turn) => turn.finishedAt && !turn.undone).at(-1) ?? null;

  /** Save one change to the chat backend's section through the settings route, with its checks. */
  async function changeSettings(edit) {
    const settings = await api('/settings');
    const backend = settings.chat_backend;
    const sections = {[backend]: edit(settings.sections?.[backend] ?? {})};
    const body = {
      chat_backend: backend,
      embedding_backend: settings.embedding_backend,
      workspace: settings.workspace,
      revision: settings.revision,
      sections,
    };
    await host.settingsSaved(await api('/settings', {method: 'PUT', body}));
  }

  async function sessionStatus() {
    return api(`${chat()}/status`);
  }

  const actions = {
    // ── The model ──

    async model({args}) {
      if (!args) return host.openSettings('model');
      await changeSettings(() => ({chat_model: args}));
      return [say(t('slash.modelSet', {model: args}), 'ok')];
    },

    async backend({args}) {
      if (!args) return host.openSettings('model');
      await changeSettings((section) => ({['chat_url' in section ? 'chat_url' : 'base_url']: args}));
      return [say(t('slash.backendSet', {url: plainUrl(args)}), 'ok')];
    },

    // ── Chats ──

    'new-chat'({name, args}) {
      // /clear also clears what is shown; /new and /side keep the old chat's rows with that chat.
      if (name === 'clear') host.clearRows();
      host.newChat(name === 'side' ? args : '');
    },

    chats({args}) {
      host.pickChat(args);
    },

    resume({args}) {
      if (!args) return host.pickChat('');
      const query = args.toLowerCase();
      const all = host.sessions();
      const found = all.filter((session) => session.id === args || session.id.startsWith(args));
      const named = found.length ? found : all.filter((session) => (session.title ?? '').toLowerCase().includes(query));
      if (named.length === 0) throw new Error(t('slash.noChat', {query: args}));
      if (named.length > 1) {
        host.pickChat(args);
        return [say(t('slash.manyChats', {n: named.length, query: args}), 'muted')];
      }
      return host.openChat(named[0].id);
    },

    main() {
      if (!host.openMain()) throw new Error(t('slash.noMain'));
    },

    send({text}) {
      if (!state().main) throw new Error(t('slash.taskMain'));
      return host.send(text);
    },

    // ── What is going on ──

    async status({args}) {
      if (args && args !== 'all') throw new Error(t('slash.useStatus'));
      const {session, health, developer, turns, id, main, title} = state();
      const row = (label, ...parts) => ({label: t(`slash.label.${label}`), parts});
      const model = health?.model;
      const reachable = health?.model_server?.reachable;
      const lines = [row('model',
        {text: [model?.name, model?.backend_label].filter(Boolean).join(' · ') || t('slash.noModel')},
        ...(reachable === true ? [{text: ` ✓ ${t('slash.answers')}`, tone: 'ok'}] : []),
        ...(reachable === false ? [{text: ` ✗ ${t('slash.noAnswer')}`, tone: 'error'}] : []))];
      if (session?.workspace_label) lines.push(row('folder', {text: session.workspace_label}));
      const count = turns.filter((turn) => !turn.undone).length;
      lines.push(row('chat', {text: `${main ? t('chat.main') : `“${title}”`} · ${count ? t('slash.turns', {n: count}) : t('slash.new')}`}));
      lines.push(row('approvals', {text: await approvalsText()}));
      if (id) {
        const body = await sessionStatus().catch(() => null);
        const words = (body?.parts ?? []).map(saidText).filter(Boolean).join(' · ');
        if (words) lines.push(row('context', {text: words}));
      }
      if (args === 'all' || developer) {
        const up = Math.round(health?.uptime_s ?? 0);
        lines.push(row('version', {text: `melo ${health?.version ?? ''}`.trim()}));
        lines.push(row('gateway', {text: t('slash.gateway', {open: health?.sessions?.open ?? 0, running: health?.sessions?.running ?? 0, up: formatDuration(up)})}));
      }
      return lines;
    },

    usage() {
      const turns = state().turns.filter((turn) => turn.finishedAt && !turn.undone);
      if (turns.length === 0) return [say(t('slash.noUsage'), 'muted')];
      const calls = turns.reduce((sum, turn) => sum + shownItems(turn).filter((item) => item.kind === 'tool').length, 0);
      const tokens = turns.reduce((sum, turn) => sum + (turn.outputTokens || 0), 0);
      const time = turns.reduce((sum, turn) => sum + turnDuration(turn), 0);
      return [say(t('slash.usage', {turns: turns.length, calls, tokens: formatTokens(tokens), time: formatDuration(time)}))];
    },

    async progress() {
      const body = await sessionStatus();
      const steps = body.steps;
      if (!steps) return [say(t('slash.noProgress'), 'muted')];
      const lines = [say(t('slash.progress', {step: steps.step, attempt: steps.attempt, phase: steps.phase || '-'}))];
      if (body.context) lines.push(say(t('slash.context', {used: formatTokens(body.context.measured), limit: formatTokens(body.context.limit ?? 0)}), 'muted'));
      return lines;
    },

    mcp() {
      return [say(t('slash.mcp'), 'muted')];
    },

    help({args}) {
      const list = host.commands();
      const word = args.replace(/^\//, '');
      if (word && word !== 'all') {
        const command = host.lookup(word);
        if (!command) throw new Error(t('slash.unknown', {name: word}));
        const also = [...command.aliases.map((alias) => `/${alias}`), ...(command.keys ? [command.keys] : [])];
        return [
          {help: {spelling: command.name, command}},
          ...(also.length ? [say(t('slash.also', {others: also.join(t('common.listSeparator'))}), 'muted')] : []),
          say(t('slash.during', {when: t(`slash.during.${command.during_turn}`)}), 'muted'),
        ];
      }
      const names = word === 'all' ? list.map((command) => command.name) : host.everyday();
      const lines = names.map((name) => ({help: {spelling: name, command: host.lookup(name)}}));
      lines.push(say(t(word === 'all' ? 'slash.helpKeys' : 'slash.helpMore'), 'muted'));
      return lines;
    },

    // ── Leaving ──

    exit() {
      if (!host.isApp()) return [say(t('slash.exitTab'), 'muted')];
      if (!host.canQuit()) return [say(t('slash.exitApp'), 'muted')];
      return [{text: t('slash.exitAsk'), confirm: {yes: t('slash.exitYes'), no: t('common.cancel'), onYes: () => host.quit()}}];
    },

    // ── Turns ──

    async retry() {
      const turn = newest();
      if (!turn) throw new Error(t('slash.nothingToRetry'));
      await api(`${chat()}/retry`, {body: {turn_id: turn.id}});
      return [say(t('slash.retrying', {text: first(turn.userText)}))];
    },

    async undo() {
      const turn = newest();
      if (!turn) throw new Error(t('slash.nothingToUndo'));
      await api(`${chat()}/undo`, {body: {turn_id: turn.id}});
      return [say(t('slash.undone'), 'ok')];
    },

    async copy({args}) {
      if (args && !/^\d+$/.test(args)) throw new Error(t('slash.useCopy'));
      const answered = state().turns.filter((turn) => !turn.undone && turn.answer);
      const turn = args ? answered.find((item) => item.turn === Number(args)) : answered.at(-1);
      if (!turn) throw new Error(t(args ? 'slash.noAnswerFor' : 'slash.nothingToCopy', {turn: args}));
      try {
        await host.writeText(turn.answer);
      } catch {
        throw new Error(t('chat.copyResponseFailed'));
      }
      return [say(t('slash.copied', {turn: turn.turn}), 'ok')];
    },

    expand({args}) {
      const calls = toolCalls(state().turns);
      if (calls.length === 0) throw new Error(t('slash.noCalls'));
      let call = calls.at(-1);
      if (args) {
        const match = /^(?:(\d+)\.)?(\d+)$/.exec(args);
        if (!match) throw new Error(t('slash.useExpand'));
        const turn = match[1] === undefined ? call.turn : Number(match[1]);
        call = calls.find((item) => item.turn === turn && item.number === Number(match[2]));
        if (!call) {
          const count = calls.filter((item) => item.turn === turn).length;
          throw new Error(count ? t('slash.fewCalls', {turn, n: count}) : t('slash.noCalls'));
        }
      }
      host.reveal(call.item);
      return [say(t('slash.expanded', {ref: call.ref, tool: call.item.name}), 'muted')];
    },

    // ── This chat ──

    async title({args}) {
      const {id, main, title} = state();
      if (main) throw new Error(t('slash.titleMain'));
      if (!args) return [say(id ? t('slash.titleIs', {title}) : t('slash.titleNone'), 'muted')];
      if (!id) {
        host.setNewTitle(args);
        return [say(t('slash.titleNext', {title: args}), 'muted')];
      }
      await host.setTitle(args);
      return [say(t('slash.titleSet', {title: args}), 'ok')];
    },

    async memory({args}) {
      if (args === 'window') return memoryWindow();
      if (args === 'import') {
        host.openMemory();
        return host.memoryImport ? host.memoryImport() : [say(t('slash.memoryImport'), 'muted')];
      }
      if (args) throw new Error(t('slash.useMemory'));
      host.openMemory();
    },

    async approvals({args}) {
      const [word = '', tool = ''] = args.split(/\s+/);
      if (word === 'ask' || word === 'auto') {
        await host.setMode(word);
        return [say(t('auto.switched', {mode: t(`auto.mode.${word}`)}), 'ok')];
      }
      if (word === 'deny') throw new Error(t('slash.noDeny'));
      if (word === 'forget') {
        const path = chat();
        try {
          await api(tool ? `${path}/grants/${enc(tool)}` : `${path}/grants`, {method: 'DELETE'});
        } catch (error) {
          throw error.status === 404 ? new Error(t('slash.noGrant', {tool})) : error;
        }
        return [say(tool ? t('slash.forgot', {tool}) : t('slash.forgotAll'), 'ok')];
      }
      if (word) throw new Error(t('slash.useApprovals'));
      return [say(await approvalsText())];
    },

    async browser({args}) {
      if (args && args !== 'on' && args !== 'off') throw new Error(t('slash.useBrowser'));
      let settings = await api('/settings');
      if (args) {
        const {chat_backend: backend, embedding_backend: embedding, workspace, revision} = settings;
        settings = await api('/settings', {method: 'PUT', body: {chat_backend: backend, embedding_backend: embedding, workspace, revision, sections: {}, browser: args === 'on'}});
        await host.settingsSaved(settings);
      }
      return [say(t(settings.browser ? 'slash.browserOn' : 'slash.browserOff'), settings.browser ? 'ok' : 'muted')];
    },

    // ── Files: the console does not take them yet ──

    attach() {
      return [say(t('slash.noFiles'), 'muted')];
    },

    // ── How the page shows things ──

    view({name, args}) {
      if (args && args !== 'on' && args !== 'off') throw new Error(t('slash.useView', {name}));
      const on = toggled(args, host.view.get(name));
      host.view.set(name, on);
      return [say(t(`slash.view.${name}.${on ? 'on' : 'off'}`), 'muted')];
    },
  };
  actions.attachments = actions.attach;
  actions.detach = actions.attach;

  async function approvalsText() {
    const {id, session} = state();
    const mode = session?.approvals === 'auto' ? 'auto' : 'ask';
    const body = id ? await api(`/sessions/${enc(id)}`).catch(() => null) : null;
    const grants = (body?.grants ?? []).join(t('common.listSeparator')) || t('slash.none');
    return t('slash.approvals', {mode: t(`auto.mode.${mode}`), grants});
  }

  async function memoryWindow() {
    const body = await api(`${chat()}/memory`);
    if (state().developer) host.openDrawer?.('memory');
    return [say(t('slash.window', {n: body.working?.length ?? 0, max: body.working_max ?? 0, memory: body.long_term?.enabled ? body.long_term.backend : t('slash.none')}))];
  }

  return actions;
}
