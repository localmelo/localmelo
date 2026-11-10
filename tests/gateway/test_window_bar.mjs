import assert from 'node:assert/strict';
import test from 'node:test';

import {setupWindowBar} from '../../support/gateway/static/window-bar.mjs';

function page({app = true} = {}) {
  const sent = [];
  let listener = null;
  const win = {
    document: {
      documentElement: {style: {getPropertyValue: (name) => (name === '--window-bar' ? '32px' : '')}},
      addEventListener: (type, handler) => { if (type === 'mousedown') listener = handler; },
    },
    webkit: app ? {messageHandlers: {localmeloWindow: {postMessage: (body) => sent.push(body)}}} : undefined,
  };
  const press = ({y = 10, detail = 1, button = 0, control = false} = {}) => {
    let prevented = false;
    const target = {closest: () => (control ? {} : null)};
    listener({clientY: y, detail, button, target, preventDefault: () => { prevented = true; }});
    return prevented;
  };
  return {installed: setupWindowBar({win}), sent, press, listening: () => listener !== null};
}

test('a press on the strip moves the window, a second press is a double click, a control keeps its press', () => {
  const app = page();
  assert.equal(app.installed, true);
  assert.equal(app.press(), true);
  app.press({detail: 2});
  assert.deepEqual(app.sent, [{action: 'drag'}, {action: 'double-click'}]);
  // A control inside the strip, a press below it and any other button stay the page's.
  assert.equal(app.press({control: true}), false);
  app.press({y: 32});
  app.press({button: 2});
  assert.equal(app.sent.length, 2);
});

test('outside the app nothing is installed', () => {
  const browser = page({app: false});
  assert.equal(browser.installed, false);
  assert.equal(browser.listening(), false);
});
