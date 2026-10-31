/**
 * The level table (`levels.mjs`): every level reads as its P-code, with its meaning where there is room.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {LEVELS, levelCode, levelLabel, levelMeaning, levelOf} from '../../support/gateway/static/levels.mjs';

const MEANINGS = [
  'runs first; others wait',
  'today',
  'normal',
  'when there is time',
  'only when the computer is idle',
  'on hold until you start it',
];

test('every level from 0 to 5 has its code and its meaning', () => {
  assert.equal(LEVELS.length, 6);
  MEANINGS.forEach((meaning, level) => {
    assert.equal(levelCode(level), `P${level}`);
    assert.equal(levelMeaning(level), meaning);
    assert.equal(levelLabel(level), `P${level} · ${meaning}`, 'a menu option is the code, then the meaning');
  });
});

test('anything that is not a level reads as P2', () => {
  for (const odd of [undefined, null, -1, 6, 1.5, '1', NaN]) {
    assert.equal(levelOf(odd), LEVELS[2], String(odd));
    assert.equal(levelLabel(odd), 'P2 · normal');
  }
  assert.equal(levelCode(0), 'P0', 'P0 is a level, not a missing one');
});
