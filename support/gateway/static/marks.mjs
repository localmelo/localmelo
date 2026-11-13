/**
 * The mark of each state a step, a row or a card can end in or be in: the terminal's own glyphs (`support/cli/theme.py`),
 * so one state reads the same on every surface. A surface keeps a mark of its own only for a state the others do not have.
 */
export const STATE_MARKS = Object.freeze({ok: '✓', failed: '✗', stopped: '■', waiting: '○', running: '●'});
