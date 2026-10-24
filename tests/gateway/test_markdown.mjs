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
  for (const destination of ['javascript:alert(1)', 'data:text/html,<script>alert(1)</script>', '//evil.example', '/relative', 'https://', 'https://host/\nonclick=alert(1)', 'jAvAsCrIpT:alert(1)', 'jav&#x61;script:alert(1)']) {
    const result = renderMarkdown('[click](' + destination + ')');
    assert.ok(!result.includes('<a '), destination);
    assert.ok(!result.includes('<script>'), destination);
  }
});

test('HTTP links have safe attributes and protect new windows', () => {
  assert.equal(renderMarkdown('[**Docs**](https://example.com/a_(b)?x=1&y=2)'), '<p><a href="https://example.com/a_(b)?x=1&amp;y=2" target="_blank" rel="noopener noreferrer"><strong>Docs</strong></a></p>');
  const result = renderMarkdown('[safe](https://example.com/\"onmouseover=\"alert(1))');
  assert.match(result, /href="https:\/\/example.com\/%22onmouseover=%22alert\(1\)"/);
  assert.ok(!result.includes(' onmouseover="'));
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
