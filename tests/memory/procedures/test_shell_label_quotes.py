"""A shell label never holds a word of quoted text.

The label is read with shlex from the command's first 2,000 characters.
When that cut, or the command itself, leaves a quote open, shlex fails;
the label then comes from the command's first line alone, and when that
fails too from that line's text before its first quote. Raw text is
never split on whitespace, which could make a quoted secret's word the
label.
"""

from __future__ import annotations

from localmelo.melo.memory.procedures.classify import label_for, shell_label


def test_a_cut_inside_quotes_leaves_no_quoted_word() -> None:
    # The 2,000-character cut lands inside the quoted string.
    body = "print('" + "x" * 3000 + "')\n"
    command = f"PGPASSWORD='open sesame' python - <<'EOF'\n{body}EOF"
    label = label_for("shell_exec", {"command": command})
    assert "sesame" not in label and "open" not in label
    assert label == "python"


def test_an_unclosed_quote_on_the_first_line_keeps_only_what_precedes_it() -> None:
    assert shell_label("echo 'open sesame") == "echo"
    assert shell_label('TOKEN="open sesame\nmore') == "shell"
    assert shell_label("'open sesame") == "shell"


def test_a_good_first_line_still_names_the_program() -> None:
    command = "cd /p && git status\necho '" + "x" * 3000 + "'"
    assert shell_label(command) == "git status"
    assert shell_label("git commit -m 'fix\n" + "x" * 3000) == "git commit"
