# Documentation Site Layout

The GitHub Pages site keeps language-specific pages in separate directories:

```text
docs/
  index.html # GitHub Pages root redirect to en/
  en/       # English HTML pages
  zh-CN/    # Chinese HTML pages
  assets/   # shared CSS, JavaScript, and images
```

`docs/index.html` is the only root-level HTML entry point. It redirects the
GitHub Pages root to `en/index.html`. Edit the files under `en/` or `zh-CN/`
for real page content.

Markdown docs that are primarily repository references, such as
`quickstart.md` and `memory.md`, stay at the docs root.
