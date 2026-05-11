# Documentation Site Layout

The GitHub Pages site keeps language-specific pages in separate directories:

```text
docs/
  en/       # English HTML pages
  zh-CN/    # Chinese HTML pages
  assets/   # shared CSS, JavaScript, and images
```

Root-level HTML files are compatibility redirects for older links such as
`/quickstart.html` and `/quickstart.zh-CN.html`. Edit the files under `en/` or
`zh-CN/` for real page content.

Markdown docs that are primarily repository references, such as
`quickstart.md` and `memory.md`, stay at the docs root.
