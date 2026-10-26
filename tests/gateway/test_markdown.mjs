import assert from 'node:assert/strict';
import test from 'node:test';
import { renderMarkdown } from '../../support/gateway/static/markdown.mjs';

test('fenced code preserves Markdown, blank lines, and source indentation', () => {
  const source = '```python\n# **literal**\n- not a list\n\n  x = "<tag>"\n```';
  assert.equal(renderMarkdown(source), '<pre><code class="language-python"># **literal**\n- not a list\n\n  x = &quot;&lt;tag&gt;&quot;\n</code></pre>');
});

test('inline code protects Markdown and accepts a longer backtick delimiter', () => {
  assert.equal(renderMarkdown('Use `**literal**` and `` a `tick` ``.'), '<p>Use <code>**literal**</code> and <code>a `tick`</code>.</p>');
});

test('code delimiters stay protected inside emphasis and link labels', () => {
  assert.equal(renderMarkdown('**before `**` after**'), '<p><strong>before <code>**</code> after</strong></p>');
  assert.equal(renderMarkdown('[a `]` label](https://example.com)'), '<p><a href="https://example.com/" target="_blank" rel="noopener noreferrer">a <code>]</code> label</a></p>');
});

test('unclosed fences show the remaining source as escaped code', () => {
  assert.equal(renderMarkdown('~~~html\n<img onerror="alert(1)">\n**literal**'), '<pre><code class="language-html">&lt;img onerror=&quot;alert(1)&quot;&gt;\n**literal**</code></pre>');
});

test('shorter fences inside a longer fence remain literal', () => {
  assert.equal(renderMarkdown('````text\n```\n**code**\n```\n````'), '<pre><code class="language-text">```\n**code**\n```\n</code></pre>');
});

test('raw HTML and entity-looking text cannot become executable HTML', () => {
  assert.equal(renderMarkdown('<img src=x onerror=alert(1)> &lt;script&gt;'), '<p>&lt;img src=x onerror=alert(1)&gt; &amp;lt;script&amp;gt;</p>');
  assert.equal(renderMarkdown('```\"><img src=x>\nx\n```'), '<pre><code>x\n</code></pre>');
});

test('unsafe, relative, and malformed link destinations remain text', () => {
  for (const destination of ['javascript:alert(1)', 'data:text/html,<script>alert(1)</script>', '//evil.example', '/relative', 'https://', 'https://host/\nonclick=alert(1)', 'jAvAsCrIpT:alert(1)', 'jav&#x61;script:alert(1)', 'vbscript:msgbox(1)', 'VBScript:msgbox(1)', 'java%73cript:alert(1)', '%6Aavascript:alert(1)', '&#104;ttps://example.com', 'https://example.com/&quot;onmouseover=&quot;alert(1)', 'https://example.com/&#x22;onmouseover=&#x22;alert(1)', 'https://example.com/<b>', "https://example.com/'q'", '<https://example.com/<b>>']) {
    const bare = destination.includes('\n') ? [] : [destination];
    for (const source of ['[click](' + destination + ')', ...bare]) {
      const result = renderMarkdown(source);
      assert.ok(!result.includes('<a '), source);
      assert.ok(!/<(?!\/?p>)/.test(result), source);
    }
  }
});

test('HTTP links have safe attributes and protect new windows', () => {
  assert.equal(renderMarkdown('[**Docs**](https://example.com/a_(b)?x=1&y=2)'), '<p><a href="https://example.com/a_(b)?x=1&amp;y=2" target="_blank" rel="noopener noreferrer"><strong>Docs</strong></a></p>');
  assert.equal(renderMarkdown('[Docs](<https://example.com/docs>)'), '<p><a href="https://example.com/docs" target="_blank" rel="noopener noreferrer">Docs</a></p>');
  const result = renderMarkdown('[safe](https://example.com/\"onmouseover=\"alert(1))');
  assert.equal(result, '<p>[safe](https://example.com/&quot;onmouseover=&quot;alert(1))</p>');
  assert.ok(!result.includes(' onmouseover="'));
});

test('bare http and https URLs link without their trailing punctuation', () => {
  assert.equal(renderMarkdown('Docs: https://example.com/a_b?x=1&y=2.'), '<p>Docs: <a href="https://example.com/a_b?x=1&amp;y=2" target="_blank" rel="noopener noreferrer">https://example.com/a_b?x=1&amp;y=2</a>.</p>');
  assert.equal(renderMarkdown('(see http://localhost:8000/v1), then HTTPS://Example.com/Path!'), '<p>(see <a href="http://localhost:8000/v1" target="_blank" rel="noopener noreferrer">http://localhost:8000/v1</a>), then <a href="https://example.com/Path" target="_blank" rel="noopener noreferrer">HTTPS://Example.com/Path</a>!</p>');
  assert.equal(renderMarkdown('https://en.wikipedia.org/wiki/Markdown_(software) and **https://example.com**'), '<p><a href="https://en.wikipedia.org/wiki/Markdown_(software)" target="_blank" rel="noopener noreferrer">https://en.wikipedia.org/wiki/Markdown_(software)</a> and <strong><a href="https://example.com/" target="_blank" rel="noopener noreferrer">https://example.com</a></strong></p>');
  assert.equal(renderMarkdown('| Site |\n| --- |\n| https://example.com |'), '<div class="table-scroll"><table><thead><tr><th class="align-left">Site</th></tr></thead><tbody><tr><td class="align-left"><a href="https://example.com/" target="_blank" rel="noopener noreferrer">https://example.com</a></td></tr></tbody></table></div>');
});

test('other schemes, quoted or bracketed URLs, code, and link labels do not autolink', () => {
  for (const source of ['ftp://example.com', 'www.example.com', 'mailto:a@example.com', 'xhttps://example.com', 'https://', '"https://example.com"', "'https://example.com'", '<https://example.com>', '`https://example.com`', 'https://example.com/"onmouseover="alert(1)', 'https://example.com/<script>alert(1)</script>']) {
    const result = renderMarkdown(source);
    assert.ok(!result.includes('<a '), source);
    assert.ok(!/<(?!\/?(?:p|code)>)/.test(result), source);
  }
  assert.equal(renderMarkdown('<a href="https://example.com">x</a>'), '<p>&lt;a href=&quot;https://example.com&quot;&gt;x&lt;/a&gt;</p>');
  assert.equal(renderMarkdown('[https://a.example](https://b.example)'), '<p><a href="https://b.example/" target="_blank" rel="noopener noreferrer">https://a.example</a></p>');
});

test('nested emphasis closes without stray delimiters', () => {
  assert.equal(renderMarkdown('**bold *and italic***'), '<p><strong>bold <em>and italic</em></strong></p>');
  assert.equal(renderMarkdown('***both***'), '<p><strong><em>both</em></strong></p>');
  assert.equal(renderMarkdown('*italic **bold***'), '<p><em>italic <strong>bold</strong></em></p>');
  assert.equal(renderMarkdown('*a **b** c* and **a *b* c**'), '<p><em>a <strong>b</strong> c</em> and <strong>a <em>b</em> c</strong></p>');
  assert.equal(renderMarkdown('a * b * c, 2*3*4, *a * b*'), '<p>a * b * c, 2<em>3</em>4, <em>a * b</em></p>');
});

test('bare URLs keep balanced brackets and drop long runs of closers in linear time', () => {
  assert.equal(renderMarkdown('[see https://x/(a)]]'), '<p>[see <a href="https://x/(a)" target="_blank" rel="noopener noreferrer">https://x/(a)</a>]]</p>');
  for (const closer of [')', ']']) {
    const started = performance.now();
    const result = renderMarkdown('https://x/' + closer.repeat(40000));
    assert.ok(performance.now() - started < 1000, closer);
    assert.equal(result, '<p><a href="https://x/" target="_blank" rel="noopener noreferrer">https://x/</a>' + closer.repeat(40000) + '</p>');
  }
});

test('emphasis openers that never close render in linear time', () => {
  for (const [source, html] of [
    ['*a '.repeat(27000), '*a '.repeat(27000)],
    ['void f(char *a, char *b); '.repeat(6000), 'void f(char *a, char *b); '.repeat(6000)],
    ['*a *b* '.repeat(12000), '*a <em>b</em> '.repeat(12000)],
    ['**a b* '.repeat(12000), '*<em>a b</em> '.repeat(12000)],
  ]) {
    const started = performance.now();
    const result = renderMarkdown(source);
    assert.ok(performance.now() - started < 1000, source.slice(0, 20));
    assert.equal(result, '<p>' + html + '</p>');
  }
});

test('long delimiter runs and spaced delimiters before a closer render in linear time', () => {
  for (const source of [
    'a' + '*'.repeat(40000),
    'a' + '_'.repeat(40000),
    'Answer: ' + '_'.repeat(40000),
    'a * b '.repeat(14000) + 'c*d',
    'a _ b '.repeat(14000) + 'c_d',
  ]) {
    const started = performance.now();
    const result = renderMarkdown(source);
    assert.ok(performance.now() - started < 1000, source.slice(0, 20));
    assert.equal(result, '<p>' + source + '</p>');
  }
});

test('normal paragraphs preserve soft breaks and CRLF input', () => {
  assert.equal(renderMarkdown('first\r\nsecond\r\n\r\nthird'), '<p>first<br>\nsecond</p><p>third</p>');
  assert.equal(renderMarkdown(''), '');
  assert.equal(renderMarkdown(null), '');
});

test('headings, emphasis, escaped punctuation, and identifier underscores render', () => {
  assert.equal(renderMarkdown('# Title\n\n**bold** *em* snake_case_value \\*literal\\*'), '<h1>Title</h1><p><strong>bold</strong> <em>em</em> snake_case_value *literal*</p>');
});

test('heading closing markers require whitespace and preserve language names', () => {
  assert.equal(renderMarkdown('# C#\n## Title ##\n###'), '<h1>C#</h1><h2>Title</h2><h3></h3>');
});

test('lists preserve starting numbers and support nested list content', () => {
  assert.equal(renderMarkdown('3. Third\n4. Fourth'), '<ol start="3"><li><p>Third</p></li><li><p>Fourth</p></li></ol>');
  assert.equal(renderMarkdown('- One\n  - Nested\n- Two'), '<ul><li><p>One</p><ul><li><p>Nested</p></li></ul></li><li><p>Two</p></li></ul>');
});

test('blockquote and horizontal rule rendering keeps source escaped', () => {
  assert.equal(renderMarkdown('> **Quote**\n> <tag>\n\n---'), '<blockquote><p><strong>Quote</strong><br>\n&lt;tag&gt;</p></blockquote><hr>');
});

test('tables protect code pipes, escaped pipes, and HTML', () => {
  assert.equal(renderMarkdown('| Name | Value |\n| :--- | ---: |\n| `a|b` | <tag> |\n| a\\|b | **two** |'), '<div class="table-scroll"><table><thead><tr><th class="align-left">Name</th><th class="align-right">Value</th></tr></thead><tbody><tr><td class="align-left"><code>a|b</code></td><td class="align-right">&lt;tag&gt;</td></tr><tr><td class="align-left">a|b</td><td class="align-right"><strong>two</strong></td></tr></tbody></table></div>');
});
