# Documentation style guide

Purpose: Write localmelo documentation that people can find, follow and verify.
For: Contributors writing or reviewing user-facing documentation.

## Page template

Start with one H1, a purpose sentence and an audience sentence. Name
prerequisites and safety constraints before the first action. Organize the
body around tasks: choose, configure, run, inspect and troubleshoot. Explain
what each action does before listing its options; show success and failure,
then link the next task and the exact reference. Keep API contracts, option
defaults, error texts and limits in their reference page instead of copying
tables into every guide. Do not turn examples or benchmark observations
into promises.

## Pages people read

The website pages (`docs/en/*.html`, `docs/zh-CN/*.html`), the two READMEs,
the guides (`docs/guides/`) and the concepts (`docs/concepts/`) are for people
who skim. The reference pages keep the page template above: exact tables and
lists, with no PR or issue numbers in their prose.

### Layout

- A section is a heading and at most five short points, one line each; a guide keeps the exact detail in a table or a code block, not in a paragraph.
- No paragraph is longer than two sentences.
- Plain words from the reader's side ("You can …"); internals only on a page about internals.
- The roadmap is a board of open work in three columns, Now, Next and Later, with a Done strip of finished work under it: cards with a title, one line and a status chip (building, testing or done); a card whose work has not started carries no chip.
- Updates are newest first: per day a heading and three to six points of what a person sees.

### Pictures and numbers

- A picture before words: a screenshot, a short loop or a diagram for each feature section that has one, placed under the heading; a table for a comparison.
- A command goes in a terminal block, not a bare code block: a title, the prompt, one colour each for the command, its options, its values and its strings, and a Copy button that copies the command alone. `scripts/usage_blocks.py` draws the blocks; edit `scripts/usage_blocks.toml`, never the generated markup.
- A guide that renames or removes a heading keeps the old one as an anchor (`<a id="old-heading"></a>`) so links and the manifest still resolve.
- Pictures live in `docs/assets/img/` as WebP, at most 1280 pixels wide, with alt text; a page's pictures stay under 3 MB. Both languages show the same pictures, except that a screenshot of the console has a Chinese version (the name ends in `-zh`) for the Chinese pages.
- No PR or issue numbers on these pages; where a reader may want detail, add a small "What changed" link to the changelog.
- `docs/for-agents.md`, `docs/llms.txt` and the changelog stay exact and terse, and keep their PR numbers.

## Examples and safety

Run every command and code example when adding or changing it. Record the
command, environment, expected output and exit status in the PR; use a
temporary HOME and workspace, a port-0 stub, no credentials and no real
model, network service or clipboard. Label examples requiring an actual
server or a human action and give a safe stub-based check when possible.
Never hide that tools run with the user's permissions without a sandbox.

## Links and language

Use meaningful link labels and stable headings; preserve existing URLs and
fragment identifiers. Write short paragraphs, consistent terms and accurate
platform caveats. The English page is the source; update its Chinese twin in
the same PR, or add a visible dated stale notice naming the English revision,
affected sections, owner and follow-up issue. Keep code, options, IDs and URLs
identical. A second
reviewer checks meaning, terminology and missing or added facts;
a maintainer spot-checks. The glossary in [Maintaining the docs](project/documentation.md)
records the established terms; the console's and the app's Chinese words follow the
[interface glossary](zh-CN/STYLE.md#interface-glossary) of the Chinese style guide. Every twin must be reviewed before Docs-1
closes.
