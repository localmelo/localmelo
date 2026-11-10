/**
 * The top strip of the macOS app's window, which has no title band: the page reaches the window's top edge and the
 * window's buttons sit over the sidebar. The app gives the page the strip's height and the buttons' reach
 * (`--window-bar`, `--window-buttons` and `data-window-bar` on the root; both are 0 in full screen) and offers the
 * `localmeloWindow` handler. A press on the strip that is not on a control moves the window, and a second press does
 * what a double click on a title bar does. A browser has no handler, and nothing is installed.
 */

/** What a press on the strip leaves to the page. */
const CONTROLS = "a, button, input, select, textarea, label, summary, dialog, [contenteditable], [role], [tabindex]";

/** Listens for presses on the strip; returns whether it did (only in the app). */
export function setupWindowBar({ win = globalThis } = {}) {
  const bridge = win.webkit?.messageHandlers?.localmeloWindow;
  if (!bridge) return false;
  const root = win.document.documentElement;
  win.document.addEventListener(
    "mousedown",
    (event) => {
      const bar = parseFloat(root.style.getPropertyValue("--window-bar")) || 0;
      if (event.button !== 0 || event.clientY >= bar || event.target?.closest?.(CONTROLS)) return;
      // The window moves instead: no text selection starts under the pointer.
      event.preventDefault();
      bridge.postMessage({ action: event.detail === 2 ? "double-click" : "drag" });
    },
    true,
  );
  return true;
}
