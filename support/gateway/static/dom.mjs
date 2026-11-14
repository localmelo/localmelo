/**
 * The element makers every console module builds its DOM with, bound to a document (the page's, or a
 * test's double): `const {node, button} = maker(doc)`. Beside them: `reachable` and `titleWhenCut`, which
 * react to a box's size, and `startOfDay`.
 */

/**
 * `node(tag, className, text)` makes an element; a class or text left out is not set.
 * `button(className, text, onClick, {label, disabled})` makes a `type="button"` with its click handler,
 * an `aria-label` when it has a `label`, and `disabled` when told to.
 */
export function maker(doc) {
  const node = (tag, className, text) => {
    const made = doc.createElement(tag);
    if (className) made.className = className;
    if (text != null) made.textContent = String(text);
    return made;
  };
  const button = (className, text, onClick, {label = '', disabled = false} = {}) => {
    const made = node('button', className, text);
    made.type = 'button';
    if (label) made.setAttribute('aria-label', label);
    if (disabled) made.disabled = true;
    made.addEventListener('click', onClick);
    return made;
  };
  return {node, button};
}

/** The local midnight at or before a time in seconds, in seconds. */
export function startOfDay(seconds) {
  const day = new Date(seconds * 1000);
  day.setHours(0, 0, 0, 0);
  return day.getTime() / 1000;
}

// What a box does when its size changes (and when it is first laid out), kept by the box. One observer serves every box;
// a page without ResizeObserver (a test's double) leaves its boxes as they were drawn.
const reactions = new WeakMap();
let observer = null;

function whenResized(box, react) {
  if (typeof globalThis.ResizeObserver !== 'function') return;
  observer ??= new globalThis.ResizeObserver((entries) => {
    for (const {target} of entries) {
      // A box that left the page is let go.
      if (target.isConnected === false) {
        observer.unobserve(target);
        reactions.delete(target);
      } else reactions.get(target)?.();
    }
  });
  reactions.set(box, react);
  observer.observe(box);
}

/**
 * A box that scrolls, reachable from the keyboard: while its content overflows it is a tab stop with a name, so the arrow
 * keys can scroll it; while the content fits it is neither. A table keeps its own role; any other box is a group.
 */
export function reachable(box, label) {
  whenResized(box, () => {
    const overflows = box.scrollHeight > box.clientHeight + 1 || box.scrollWidth > box.clientWidth + 1;
    if (overflows === box.hasAttribute('tabindex')) return;
    if (overflows) {
      box.setAttribute('tabindex', '0');
      box.setAttribute('aria-label', label);
      if (box.localName !== 'table') box.setAttribute('role', 'group');
    } else {
      for (const name of ['tabindex', 'aria-label', 'role']) box.removeAttribute(name);
    }
  });
  return box;
}

/**
 * Words cut with an ellipsis show in full as the box's tooltip; words that fit have none. The box is read when it is next
 * laid out, not here: this runs from the page's redraw, and a read here would make the browser lay the page out in the
 * middle of it.
 */
export function titleWhenCut(box, words) {
  whenResized(box, () => {
    const title = box.scrollWidth > box.clientWidth + 1 ? words : '';
    if (box.title !== title) box.title = title;
  });
}
