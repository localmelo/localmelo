// Reads pages with melo's page script in jsdom, for the Python tests that hand real
// readings to the tools. stdin: {"script": <the page script>, "pages": [<html>, ...]};
// stdout: the readings, in the same order.
import {readFileSync} from 'node:fs';

import {JSDOM} from 'jsdom';

const PRELUDE = `
Object.defineProperty(HTMLElement.prototype, 'innerText', {get() { return this.textContent; }, configurable: true});
Element.prototype.getBoundingClientRect = function () { return {width: 10, height: 10, top: 0, left: 0, right: 10, bottom: 10}; };
Element.prototype.getClientRects = function () { return [this.getBoundingClientRect()]; };
`;

const input = JSON.parse(readFileSync(0, 'utf8'));
const readings = input.pages.map((html) => {
  const dom = new JSDOM(html, {url: 'https://example.com/', runScripts: 'outside-only', pretendToBeVisual: true});
  dom.window.eval(PRELUDE);
  return JSON.parse(dom.window.eval(input.script));
});
process.stdout.write(JSON.stringify(readings));
