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
identical. A DeepSeek
headless reviewer checks meaning, terminology and missing or added facts;
the lead spot-checks. The glossary in [Maintaining the docs](maintaining.md)
records the established terms. Every twin must be reviewed before Docs-1
closes.
