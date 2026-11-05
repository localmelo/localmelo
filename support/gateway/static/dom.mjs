/**
 * The element makers every console module builds its DOM with, bound to a document (the page's, or a
 * test's double): `const {node, button} = maker(doc)`.
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
