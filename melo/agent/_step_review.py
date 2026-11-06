"""The model review of steps, and auto mode's blocks (#279 §3.3, #339).

:class:`StepReviewer` belongs to one Agent, so one chat: it keeps the
user's latest messages, the reviews it settled (by step key, the
:data:`REVIEWS_KEPT` most recently used) and the steps auto mode blocked.
:class:`RunReviews` is one run's share: the reviews it sent, its calls, its
blocks and the sites refused in it, and the summary ``run.finished``
carries.

When the model is asked:

- an undecided call that would run without a person (a session grant, or
  auto mode);
- in auto mode, a high-impact call whose categories the user's words can
  allow (:data:`~localmelo.melo.contracts.review.WORDS_CAN_ALLOW`), with
  nothing the rules could not tell, when a user message came after a block
  of the same kind or names something the call acts on.

It is never asked about a routine call or a ``self`` one, nor when a person
will be asked anyway. A request that times out, fails or cannot be read is
sent once more; a review that still has no answer, one past the turn's
:data:`~localmelo.melo.checker.review.model.REVIEWS_PER_TURN` (counted for the
steps the rules could not judge only; a high-impact step the model only
checks against the user's words has a count of its own), and a call too long
to review all leave the call high with the category ``unreviewed``. Auto mode
shows a person that step on a card instead of blocking it without a word
(``ASKS_IN_AUTO``); with nobody to ask it stays blocked (fail closed).

The goal decides (#465): an undecided call the model finds on the goal of the
chat, with no high-impact category, runs; the user's words only ever add to
that. Such a review reads the chat's older messages too, and the notes
recalled for the newest one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import ipaddress
import os
import re
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal
from urllib.parse import urlsplit

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.review.model import (
    CHAT_REQUESTS,
    EARLIER_REQUESTS,
    MAX_ACTION_CHARS,
    MAX_REQUEST_CHARS,
    REVIEW_TIMEOUT_S,
    REVIEW_TRIES,
    REVIEWS_PER_TURN,
    SYSTEM_PROMPT,
    ReviewInputs,
    action_text,
    own_words,
    parse_reply,
    recalled_part,
    shares_a_name,
    step_names,
    user_message,
)
from localmelo.melo.checker.review.named import (
    folder_named,
    key_named,
    local_sites,
    loopback_site,
    script_named,
)
from localmelo.melo.checker.review.posting import (
    account_names,
    asks_to_search,
    asks_to_share,
    is_message,
    is_post,
    post_stop,
    sent_texts,
)
from localmelo.melo.contracts.approvals import Chosen, ReviewMode, block_note
from localmelo.melo.contracts.review import (
    CATEGORY_WORDS,
    DIRENV_SUBJECT,
    MODEL_CAN_ALLOW,
    TITLES,
    WORDS_CAN_ALLOW,
    ActionReview,
    ModelAnswer,
    because,
    named_by,
    step_key,
)
from localmelo.melo.contracts.sign_in import HANDLE, VAULT, handles_in, hide_passwords
from localmelo.melo.memory.procedures.mask import mask_credentials
from localmelo.melo.schema import Message

if TYPE_CHECKING:
    from localmelo.melo.agent._chat import Chat

# Blocks in one turn that stop it: a second of the same kind, or this many.
BLOCKS_PER_TURN: Final = 3
# Reviews of a runner (pytest…) that runs changed files melo does not read,
# a turn: apart from the turn's other reviews, so test cycles leave them.
RUNNER_REVIEWS_PER_TURN: Final = 3
# Blocks of a chat kept for the prompt and the "go ahead" check.
BLOCKS_KEPT: Final = 20
# Settled reviews a chat keeps, the least recently used dropped first.
REVIEWS_KEPT: Final = 512
# The longest path of an address one verdict on its site covers: a longer
# one, which can carry more data out, is reviewed by itself.
PLAIN_PATH_CHARS: Final = 120
# Addresses one verdict on a site lets through until the user's next
# message; past that, each is reviewed by itself.
PLAIN_PAGES_PER_TURN: Final = 10

Purpose = Literal["undecided", "consent"]

# The last label of a bare word that names a file, not a site: ``README.md``,
# ``run.sh`` or ``package.json`` in a message names no host. A full
# ``http://`` or ``https://`` address names its host whatever it ends in.
_FILE_EXTENSION_GROUPS: Final = (
    "md markdown mdx rst txt adoc tex bib log out tmp bak orig swp diff patch",
    "sh bash zsh fish bat cmd ps",
    "py pyi pyc pyx pxd ipynb js mjs cjs jsx ts mts cts tsx vue svelte",
    "go mod sum rs rb pl pm php java kt kts scala swift cs fs lua jl dart ex exs",
    "erl hs elm clj cljs nim zig sol ml mli cc cpp cxx hh hpp hxx mm proto",
    "gql graphql mk cmake gradle tf tfvars hcl nix wasm",
    "html htm xhtml xml xsl xslt xsd svg css scss sass less",
    "json jsonl ndjson geojson yaml yml toml ini cfg conf env properties lock plist",
    "csv tsv parquet arrow feather avro sql db sqlite pkl pickle npy npz pth ckpt",
    "safetensors gguf onnx mlmodel mlpackage",
    "png jpg jpeg gif webp bmp ico icns tif tiff heic heif psd",
    "wav flac ogg aac mov avi mkv webm",
    "pdf doc docx xls xlsx ppt pptx odt ods odp rtf epub key pages numbers",
    "zip tar gz tgz bz xz zst rar dmg pkg deb rpm whl egg jar war exe dll so dylib",
    "bin dat pem crt cer csr pub asc gpg sig ics vcf eml srt vtt ttf otf woff eot map",
)
FILE_EXTENSIONS: Final = frozenset(" ".join(_FILE_EXTENSION_GROUPS).split())
# Scripts written without spaces between words (Thai, Lao, Myanmar, Khmer,
# Japanese kana, Chinese, Korean), as ``re`` ranges: a host right after one
# of their characters, as in "打开example.com看看", is still a word of its own.
_NO_SPACES: Final = (
    r"\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3100-\u312f"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f"
)
# A letter, digit or "_" that carries a word on: one before a host makes it
# part of a longer word ("cher.de" in "bücher.de" is not a host).
_JOINS: Final = rf"[^\W{_NO_SPACES}]"
_HOST: Final = re.compile(
    rf"(?<![.\-])(?<!{_JOINS})"
    r"((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})"
    rf"(?!-|{_JOINS})"
)
_URL: Final = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)


def _plain_host(host: str) -> str:
    """*host* compared as a name: lower case, in ASCII, without one ``www.``."""
    host = host.strip().strip("[]").lower().rstrip(".")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return ""
    return host.removeprefix("www.")


def _site_host(text: str) -> str:
    """The host of *text* (a host, ``host:port`` or a URL), as :func:`_plain_host` has it."""
    text = text.strip()
    try:
        return str(ipaddress.ip_address(text.strip("[]")))
    except ValueError:
        pass
    try:
        host = urlsplit(text if "://" in text else "//" + text).hostname
    except ValueError:
        return ""
    return _plain_host(host or "")


def named_sites(text: str) -> set[str]:
    """The hosts *text* names (as :func:`_plain_host` has them).

    The host of each full ``http://`` or ``https://`` address, and each
    bare word shaped as a host (``example.com``) whose last label is not a
    common file extension (:data:`FILE_EXTENSIONS`). A word inside a longer
    one is not a host: ``bücher.de`` does not name ``cher.de``.
    """
    names: set[str] = set()
    for address in _URL.findall(text):
        try:
            host = urlsplit(address).hostname or ""
        except ValueError:
            continue
        if ":" not in host:
            # The host as far as it is one word: "https://example.com)." is
            # example.com.
            word = re.match(r"[\w.-]*", host)
            host = word.group(0) if word else ""
        if host := _plain_host(host):
            names.add(host)
    for word in _HOST.findall(text.lower()):
        if word.rsplit(".", 1)[-1] not in FILE_EXTENSIONS:
            names.add(_plain_host(word))
    names.discard("")
    return names


@dataclass
class _Block:
    category: str
    key: str
    marker: int  # the user message count when it was blocked
    title: str
    preview: str


@dataclass
class RunReviews:
    """One run's reviews: what ``run.finished`` reports, and the stop rule's state."""

    # Reviews of steps the rules could not judge, of high-impact steps the
    # rules judged (the model reads only the user's words), and of runners.
    used: int = 0
    consent_used: int = 0
    runner_used: int = 0
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    blocks: list[tuple[str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {"routine": 0, "high": 0, "undecided": 0, "model": 0}
    )
    blocked: list[dict[str, str]] = field(default_factory=list)
    # A second block of the same kind: the run stops and answers.
    stop: bool = False
    # The hosts of the sites refused in this run, oldest first (#433): the
    # review of a later step is told of them.
    refused_sites: list[str] = field(default_factory=list)
    # The steps declined on auto mode's card in this run (#465), by each of
    # their keys: the person's Don't allow, or a card nobody answered (no
    # answer, not a no: the model is told to go on and melo asks again in the
    # next run). The same step, or one alike, is refused again without asking
    # until the run ends, so a card that waits is not shown again and again.
    declined: dict[str, Chosen] = field(default_factory=dict)

    def decline(self, keys: Iterable[str], decision: Chosen) -> None:
        """Keep a step declined on the card (*decision*), under each of *keys*."""
        for key in keys:
            if key:
                self.declined.setdefault(key, decision)

    def declined_again(self, keys: Iterable[str]) -> Chosen | None:
        """The refusal of a step with any of *keys* declined in this run, else None.

        It keeps who declined it and why (the person's note, or that nobody
        answered), and says that nobody was asked this time (``again``).
        """
        for key in keys:
            found = self.declined.get(key)
            if found is not None:
                return dataclasses.replace(found, remember=False, again=True)
        return None

    def count(self, review: ActionReview | None) -> None:
        if review is not None:
            self.counts[review.verdict] = self.counts.get(review.verdict, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "blocked": [dict(b) for b in self.blocked]}


def local_address(host: str) -> bool:
    """Whether *host* (or ``host:port``, or a URL) is an address written as a number that is not public.

    ``127.0.0.1``, ``[::1]:8123`` and ``10.0.0.5`` are; a name and a public address are not.

    Such an address can be reached only on this computer or its network, so
    what the user's words name and what one review covers do not carry over
    to it: it is another server on another port, and its pages are not a
    site's pages.
    """
    text = host.strip()
    try:
        return not ipaddress.ip_address(text.strip("[]")).is_global
    except ValueError:
        pass
    try:  # host:port or a URL
        name = urlsplit(text if "://" in text else "//" + text).hostname or ""
        return not ipaddress.ip_address(name).is_global
    except ValueError:
        return False


def plain_page_key(
    tool_name: str, arguments: Mapping[str, Any], workspace: str | None
) -> str:
    """The key a routine verdict on a site's plain pages is kept under, or ``""``.

    For a site question that carries the address it reads (web_fetch's
    ``url``): when the address has no query, no fragment and no user name,
    is shown whole and its path has at most :data:`PLAIN_PATH_CHARS`
    characters, the question without it, so one review lets the site's
    other plain pages be read in this chat (at most
    :data:`PLAIN_PAGES_PER_TURN` until the user's next message). Any other
    address, which can carry more data out, is reviewed by itself, and so is
    every address on this computer (:func:`local_address`).
    """
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip() or url.endswith("…"):
        return ""
    if any(mark in url for mark in "?#"):
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if "@" in parts.netloc or len(parts.path) > PLAIN_PATH_CHARS:
        return ""
    if local_address(parts.hostname or ""):
        return ""
    rest = {name: value for name, value in arguments.items() if name != "url"}
    return "site|" + step_key(tool_name, rest, workspace)


def _texts(value: object, depth: int = 0) -> Iterable[str]:
    """The strings in a step's *value*: itself, or a mapping's values and a
    list's items, at most 4 levels deep."""
    if isinstance(value, str):
        yield value
    elif depth < 4 and isinstance(value, Mapping):
        for item in value.values():
            yield from _texts(item, depth + 1)
    elif depth < 4 and isinstance(value, (list, tuple)):
        for item in value:
            yield from _texts(item, depth + 1)


def holds_secret(arguments: Mapping[str, Any], preview: str = "") -> bool:
    """Whether a step's *arguments* or *preview* hold a secret by melo's detectors.

    A key or token shape, a bearer token, a code next to its word or a private
    key (:func:`~localmelo.melo.checker.redaction.redact_secrets`), and an
    address's password or a value named as a secret (``token=…``,
    :func:`~localmelo.melo.memory.procedures.mask.mask_credentials`).
    """
    for text in (preview, *_texts(arguments)):
        if text and (redact_secrets(text)[1] or mask_credentials(text) != text):
            return True
    return False


def _title(review: ActionReview) -> str:
    first = review.categories[0] if review.categories else "beyond_request"
    return TITLES.get(first, "Run this?").rstrip("?").format(folder="the folder")


class StepReviewer:
    """Asks the user's model about steps the rules left open; keeps the chat's blocks."""

    def __init__(
        self,
        chat: Callable[[], Chat],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        # The Agent's Chat, read at each review (a test may replace it).
        self._chat = chat
        self._clock = clock
        self._wall = wall
        # What code can match of the person in a post: the account's real name
        # (a test may set it). See :func:`~localmelo.melo.checker.review.posting.personal_data`.
        self.person_names: tuple[str, ...] = account_names()
        # When the chat began: the rules count a file changed after it as
        # written in this chat. A session runner sets the stored start.
        self.since = wall()
        self.reset()

    def reset(self, *, new_chat: bool = False) -> None:
        """Forget the messages, reviews and blocks kept (a reload, an undo).

        With *new_chat*, the chat also begins now (``since``).
        """
        # The person's own words of the chat, newest last: the newest
        # EARLIER_REQUESTS + 1 are its recent requests, the rest (and the first
        # message, kept) what a goal review also reads (#465).
        self._words: deque[str] = deque(maxlen=CHAT_REQUESTS + EARLIER_REQUESTS + 1)
        self._first = ""
        self._added = 0
        # melo's recalled notes that came with the person's newest message.
        self._recalled = ""
        self._marker = 0
        # Whether the person's earlier words were put in since the reset.
        self._seeded = False
        self._cache: OrderedDict[str, tuple[ActionReview, int, Purpose]] = OrderedDict()
        # Addresses each site verdict let through since the user's last message.
        self._shared_used: dict[str, int] = {}
        self._blocks: deque[_Block] = deque(maxlen=BLOCKS_KEPT)
        # The handles of passwords the person's messages of this run hold
        # (#547): only they can be typed, and none in a run they did not begin.
        self._live: frozenset[str] = frozenset()
        if new_chat:
            self.since = self._wall()

    # ── The user's words ──

    def user_message(self, text: str, recalled: str | None = None) -> None:
        """A message of the user's own: a run's request, or one it read mid-run.

        Recalled notes and earlier chats appended to it are left out: they
        are not the user's words, and an earlier answer could argue a call.
        The newest :data:`EARLIER_REQUESTS` messages before it stay as
        evidence of what the user asked (#465): "I signed in" after "check
        me in" still asks for the check-in. A review that judges a step on
        the goal also reads the chat's older messages (:attr:`earlier`) and
        *recalled*, the notes recalled for this message, which count only
        where the person's words point to them. Without *recalled*, the notes
        *text* carries replace the kept ones; a message that carries none
        (one read mid-run) keeps them.
        """
        if recalled is None:
            recalled = recalled_part(text) or self._recalled
        self._recalled = recalled
        text = own_words(text)
        # A password this message holds can be typed in this run (#547).
        self._live |= handles_in(text)
        # A message counts even with no words (an attachment): a review
        # settled for the last message is asked again, with the new words.
        self._marker += 1
        self._shared_used.clear()
        if text.strip():
            self._add(text[:MAX_REQUEST_CHARS])

    def _add(self, text: str) -> None:
        if not self._added:
            self._first = text
        self._added += 1
        self._words.append(text)

    def earlier_words(self, words: Sequence[str], first: str = "") -> None:
        """The user's own words of the chat's messages before this run's, oldest first.

        Only for a window that starts over (a reload, a chat opened again,
        a new Agent): the newest :data:`EARLIER_REQUESTS` go in before the
        run's own message, so the person's earlier words still count
        (#465). The caller gives only what the person typed: never melo's
        answers, a task's questions or page text; recalled notes are left
        out here too. *first* is what they typed in the chat's first turn:
        it goes in before them, as the task the chat began with (#465), unless
        it is one of them. Once a message was kept, or the words were put in
        since the reset, nothing changes.
        """
        if self._marker or self._seeded:
            return
        self._seeded = True
        texts = [own_words(text)[:MAX_REQUEST_CHARS] for text in words]
        texts = [text for text in texts[-EARLIER_REQUESTS:] if text.strip()]
        first = own_words(first)[:MAX_REQUEST_CHARS]
        if first.strip() and first not in texts:
            self._add(first)
        for text in texts:
            self._add(text)

    # ── Sign-in details the person gives (#547) ──

    def hide_passwords(self, text: str) -> str:
        """*text*, the person's message, with each password it gives replaced by its handle.

        Called where a message of the person's comes in (a session's runner,
        :meth:`~localmelo.melo.agent.Agent.run_task`, a note read mid-run),
        before any model, record, log or memory sees it: the value stays in
        this process's vault and the handle (``[password-1]``) takes its
        place (see :mod:`localmelo.melo.contracts.sign_in`). The recalled
        notes a message may carry (:func:`own_words`) are not the person's
        words: nothing is taken from them. The same text hidden again
        changes nothing.
        """
        own = own_words(text)
        return hide_passwords(own) + text[len(own) :]

    def begin_run(self) -> None:
        """A run starts: no password can be typed in it until the person's message of the run holds one (#547).

        A run the person did not begin (a task's follow-up, a chore) never
        types one, whatever an earlier message held.
        """
        self._live = frozenset()

    def sign_in_value(self, handle: str, site: str) -> str | None:
        """The password *handle* stands for, for a password box on *site*, or ``None``.

        Only a handle a message of the person's run holds, and only on a site their
        own words name (:meth:`names_host`): the model, a page or an earlier
        message never makes melo type one. The browser tool asks this and
        types the value; nothing else ever gets it.
        """
        if (
            not HANDLE.fullmatch(handle)
            or handle not in self._live
            or not self.names_host(site)
        ):
            return None
        return VAULT.value(handle)

    def signs_in(self, site: str) -> bool:
        """Whether a message of the person's run holds a password and their words name *site*."""
        return any(VAULT.value(h) is not None for h in self._live) and self.names_host(
            site
        )

    @property
    def requests(self) -> tuple[str, ...]:
        """The person's newest messages: :data:`EARLIER_REQUESTS` and the latest."""
        return tuple(self._words)[-(EARLIER_REQUESTS + 1) :]

    @property
    def earlier(self) -> tuple[str, ...]:
        """The person's older messages of the chat before :attr:`requests`, oldest first.

        The chat's first message (since the window began) is kept when it
        fell out: the task the chat is about (#465).
        """
        older = tuple(self._words)[: -(EARLIER_REQUESTS + 1)]
        if self._added > len(self._words):
            return (self._first, *older)
        return older

    def names_host(self, host: str) -> bool:
        """Whether the user's recent messages name *host* itself.

        The same host, or it with ``www.`` added or removed, and nothing
        under it: naming ``example.com`` does not name ``docs.example.com``,
        so a site under a shared domain (``vercel.app``) is not named by
        it. See :func:`named_sites` for what counts as naming. An address
        on this computer (:func:`local_address`) is named only by its host
        and its port together, as the person wrote them: a host alone is not
        its port, and a page may have steered the model to another.
        """
        local = loopback_site(host)
        if local is not None:
            # A server on this computer is named by its host and its port
            # together, as the person wrote them (#547); the port a page or the
            # model picked is another site.
            return local in self.written_sites()
        if local_address(host):
            return False
        wanted = _site_host(host)
        return bool(wanted) and any(
            wanted in named_sites(text) for text in self.requests
        )

    def name(self, review: ActionReview) -> ActionReview:
        """*review* with the categories the person's own words name (#547).

        A key file the person wrote the path or the specific name of, a
        script they wrote the whole address of: for those auto mode shows a
        card instead of stopping the step (:func:`~localmelo.melo.contracts.approvals.asks_in_auto`).
        Decided here by code from the rules' evidence and the person's words
        (:attr:`requests`): the model and a page never name anything.
        """
        named = named_by(review, self._names)
        return dataclasses.replace(review, named=named) if named else review

    def _names(self, category: str, subject: str) -> bool:
        if category == "credentials":
            return key_named(subject, self.requests, os.path.expanduser("~"))
        if category == "persistence":
            return subject == DIRENV_SUBJECT  # a card whatever the words say
        return category == "remote_code" and script_named(subject, self.requests)

    def names_folder(self, path: str) -> bool:
        """Whether the person's own words name *path* (canonical, outside the workspace) or a folder above it.

        The home folder and the folders above it never count (see
        :func:`~localmelo.melo.checker.review.named.folder_named`); a place
        melo never opens does not reach here at all.
        """
        return folder_named(path, self.requests, os.path.expanduser("~"))

    def written_sites(self) -> frozenset[tuple[str, int]]:
        """The servers on this computer the person's own words write, as ``(address, port)``.

        ``localhost`` is read as ``127.0.0.1``. The browser and web_fetch let
        exactly these through (see :func:`~localmelo.support.local_ports.written_sites`);
        a page's text, a tool's result and the model's own address never count.
        """
        return local_sites(self.requests)

    @property
    def all_words(self) -> tuple[str, ...]:
        """Every message of the person's the reviewer keeps: the chat's first, the older ones and the newest."""
        return (*self.earlier, *self.requests)

    def asks_to_share(self) -> bool:
        """Whether the person's own words speak of sharing or making something public (#547)."""
        return asks_to_share(self.requests)

    def searching(self, arguments: Mapping[str, Any]) -> bool:
        """Whether a step types or sends a search of the person's (#547).

        The browser says ``field: "search"`` when the box, or every text box of
        the form, is described as a search box
        (:func:`~localmelo.support.browser.fields.searches`); a page can
        describe a box so, so it counts only when the person's own words ask to
        search too (:func:`~localmelo.melo.checker.review.posting.asks_to_search`).
        A search is no post whatever its length: a query is not a message.
        """
        return (
            arguments.get("access") in ("type", "submit")
            and arguments.get("field") == "search"
            and asks_to_search(self.requests)
        )

    def _posting(
        self,
        settled: ActionReview,
        rules: ActionReview,
        answer: ModelAnswer,
        tool_name: str,
        arguments: Mapping[str, Any],
        workspace: str,
    ) -> ActionReview:
        """A post that the model or the person's words let through, asked about when code finds a reason (#547).

        A post is a step that types or sends text on a page or posts or sends a
        message (:func:`is_post`) and that either holds a message
        (:func:`is_message`, whatever category the model gave) or is called
        ``send_out`` by the rules or the model. It runs without a card only if
        the person's own words ask for a send or the person wrote the text, the
        text holds none of their phone, address or real name, a review or
        rating is their own text, and the task does not repeat
        (:func:`~localmelo.melo.checker.review.posting.post_stop`): whatever
        else the model said, a page cannot make these true. The step is
        then high ``send_out`` with that reason, which a card shows when a
        person watches.
        """
        sent = sent_texts(tool_name, arguments, workspace)
        if (
            settled.verdict != "routine"
            or settled.by != "model"
            or self.searching(arguments)
            or not is_post(arguments, rules.evidence)
            or not (
                is_message(sent)
                or "send_out" in rules.categories
                or answer.category == "send_out"
            )
        ):
            return settled
        stop = post_stop(
            self.all_words,
            sent,
            self.person_names,
            typing=arguments.get("access") == "type",
        )
        if not stop:
            return settled
        return dataclasses.replace(
            settled, verdict="high", categories=("send_out",), reasons=(stop,)
        )

    # ── Reviews ──

    def _purpose(
        self, review: ActionReview, mode: ReviewMode, wanted: set[str]
    ) -> Purpose | None:
        if mode == "ask" or review.verdict == "routine" or "self" in review.categories:
            return None
        if review.verdict == "undecided":
            return "undecided"
        if (
            mode != "auto"
            or review.doubts  # words never allow what melo could not read
            or not set(review.categories) <= WORDS_CAN_ALLOW
        ):
            return None
        after_block = any(
            block.category in review.categories and block.marker < self._marker
            for block in self._blocks
        )
        if after_block or shares_a_name(wanted, self.requests):
            return "consent"
        return None

    async def review(
        self,
        run: RunReviews,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        preview: str,
        review: ActionReview,
        mode: ReviewMode,
        key: str,
        workspace: str | None,
        shared: str = "",
        refused_sites: Sequence[str] = (),
        secret_free: bool = False,
    ) -> ActionReview:
        """*review* settled by the model when the rules leave it open, else as it is.

        A settled review is kept by step key: one that lets an undecided
        call run is used again in this chat; any other only until the
        user's next message, since new words can change the answer. A
        verdict that lets it run is kept under *shared* too (a site's plain
        pages, :func:`plain_page_key`) and used for any call with that key.
        *refused_sites* are the hosts of the sites refused in this task
        (#433): the model is told of them, and a verdict given before one
        was refused is not used again. *secret_free*: the step carries no
        value (a click, a choice, a point), so a model's ``credentials`` does
        not hold for it (#465, see :func:`_settle`), unless melo's own
        detectors find a secret in its arguments or preview
        (:func:`holds_secret`): then the answer stands.
        """
        if secret_free and holds_secret(arguments, preview):
            secret_free = False
        action = action_text(tool_name, arguments)
        purpose = self._purpose(review, mode, step_names(tool_name, arguments))
        if purpose is None:
            return review
        # A runner's review holds while the files it could not read stay as
        # they are.
        key = key + "|" + "|".join(review.changed) if review.changed else key
        if refused_sites:
            refused = "|refused:" + ",".join(refused_sites)
            key, shared = key + refused, shared and shared + refused
        cached = self._kept(key)
        if cached is not None and (
            cached[1] == self._marker
            or cached[2] == "undecided"
            and cached[0].verdict == "routine"
        ):
            return cached[0]
        # A site's verdict never lets through an address judged otherwise
        # before, and only so many a turn.
        kept = (
            self._kept(shared)
            if shared
            and (cached is None or cached[0].verdict == "routine")
            and self._shared_used.get(shared, 0) < PLAIN_PAGES_PER_TURN
            else None
        )
        if kept is not None and kept[2] == "undecided" and kept[0].verdict == "routine":
            self._shared_used[shared] = self._shared_used.get(shared, 0) + 1
            return kept[0]
        runner = bool(review.changed)
        goal = purpose == "undecided"
        used = run.runner_used if runner else run.used if goal else run.consent_used
        if len(action) > MAX_ACTION_CHARS:
            answer = ModelAnswer(error="the call is too long to review")
        elif used >= (RUNNER_REVIEWS_PER_TURN if runner else REVIEWS_PER_TURN):
            answer = ModelAnswer(error="this turn's reviews are used up")
        else:
            if runner:
                run.runner_used += 1
            elif goal:
                run.used += 1
            else:
                run.consent_used += 1
            run.counts["model"] += 1
            inputs = ReviewInputs(
                workspace=workspace or "",
                requests=self.requests,
                tool=tool_name,
                action=action,
                findings=because(review) if review.reasons else "",
                calls=run.calls,
                blocked=[(b.title, b.preview) for b in self._blocks],
                refused_sites=tuple(refused_sites),
                # A step judged on the goal reads the whole chat and the notes
                # the person may point to; a consent only their newest words.
                earlier=self.earlier if goal else (),
                recalled=self._recalled if goal else "",
            )
            answer = await self._ask(inputs)
        settled = _settle(review, answer, purpose, secret_free=secret_free)
        settled = self._posting(
            settled, review, answer, tool_name, arguments, workspace or ""
        )
        if not answer.error:
            self._keep(key, (settled, self._marker, purpose))
            if shared and purpose == "undecided" and settled.verdict == "routine":
                self._keep(shared, (settled, self._marker, purpose))
        return settled

    def _kept(self, key: str) -> tuple[ActionReview, int, Purpose] | None:
        found = self._cache.get(key)
        if found is not None:
            self._cache.move_to_end(key)
        return found

    def _keep(self, key: str, settled: tuple[ActionReview, int, Purpose]) -> None:
        """Keep a settled review; past :data:`REVIEWS_KEPT`, the least recently used goes."""
        self._cache[key] = settled
        self._cache.move_to_end(key)
        while len(self._cache) > REVIEWS_KEPT:
            self._cache.popitem(last=False)

    async def _ask(self, inputs: ReviewInputs) -> ModelAnswer:
        """One review: up to :data:`REVIEW_TRIES` requests; any failure is an answer with an ``error``.

        A request that timed out, failed or came back unreadable is sent
        once more (a busy local model often answers the second time); the
        answer of the last one stands. ``ms`` spans all of them.
        """
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_message(inputs)),
        ]
        started = self._clock()
        answer = ModelAnswer(error="the review did not run")
        for _ in range(REVIEW_TRIES):
            answer = await self._request_once(messages, started)
            if not answer.error:
                break
        return answer

    async def _request_once(
        self, messages: list[Message], started: float
    ) -> ModelAnswer:
        def ms() -> int:
            return int((self._clock() - started) * 1000)

        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                reply = await self._chat()._request(
                    messages, None, kind="review", retry=False
                )
        except TimeoutError:
            return ModelAnswer(error="the review timed out", ms=ms())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the review fails closed, the run goes on
            return ModelAnswer(
                error=f"the review failed ({type(exc).__name__})", ms=ms()
            )
        answer = parse_reply(reply.content)
        if answer is None:
            return ModelAnswer(error="its answer could not be read", ms=ms())
        return dataclasses.replace(answer, ms=ms())

    # ── Blocks ──

    def blocked(
        self,
        run: RunReviews,
        *,
        review: ActionReview | None,
        key: str,
        call_id: str,
        preview: str,
        words: str = "",
        task: bool = False,
        workspace: str | None = None,
        changed: bool = False,
    ) -> str:
        """Keep a block of auto mode and return the note the model reads.

        A second block of the same step or category in this run, or the
        :data:`BLOCKS_PER_TURN`-th, sets ``run.stop``: the note then says to
        stop and answer.
        """
        reviewed = review if review is not None else ActionReview("high")
        first = reviewed.categories[0] if reviewed.categories else "beyond_request"
        second = any(k == key or c == first for k, c in run.blocks)
        run.blocks.append((key, first))
        if second or len(run.blocks) >= BLOCKS_PER_TURN:
            run.stop = True
        title = _title(reviewed)
        self._blocks.append(_Block(first, key, self._marker, title, preview))
        run.blocked.append({"call_id": call_id, "title": title, "step_key": key})
        folder = os.path.basename(os.path.normpath(workspace)) if workspace else ""
        return block_note(
            review,
            words=words,
            task=task,
            folder=folder or "the folder",
            second=run.stop,
            changed=changed,
        )


def _model_reason(category: str) -> str:
    """melo's own sentence for what the model found: its free text is only kept.

    It starts with a verb, as every reason does ("Uses passwords or keys, as
    your model's review says."), so a note reads "It uses ...".
    """
    if category == "beyond_request":
        return "Does something the request did not ask for."
    if category == "opaque":
        return "Does something melo could not check."
    words = CATEGORY_WORDS[category]
    return f"{words[:1].upper()}{words[1:]}, as your model's review says."


# The model's categories that name no high-impact kind: for a step it finds on
# the goal (``fits_request: yes``) the goal decides, and it runs whatever the
# verdict and the user's words say (#465). Any other category keeps today's
# rule.
GOAL_DECIDES: Final = frozenset({"none", "", "beyond_request"})


def _settle(
    review: ActionReview,
    answer: ModelAnswer,
    purpose: Purpose,
    *,
    secret_free: bool = False,
) -> ActionReview:
    """The review once the model answered (or failed to).

    A call runs only on a category of :data:`MODEL_CAN_ALLOW`: an answer
    that names a category only a click can allow keeps it blocked, and an
    undecided call the model files under a category words can allow runs
    only when the user's words ask for it. An undecided call the model finds
    on the goal with no high-impact category (:data:`GOAL_DECIDES`) runs:
    whether the user's words name it only ever allows (#465). For a
    *secret_free* step (a click, a choice or a point with no secret in it: no
    value goes anywhere) a ``credentials`` answer is no credential use: the
    step is judged on the goal like one with no category. A read, a search,
    typed text, a form sent and every gate step are never *secret_free*.
    """
    kind = answer.category
    if secret_free and kind == "credentials":
        kind = "none"  # the answer itself is kept as it was given
    allows = kind in MODEL_CAN_ALLOW
    if purpose == "consent":
        if answer.user_asked == "yes" and answer.fits_request == "yes" and allows:
            # The user's words allow it: it runs, its categories kept.
            return dataclasses.replace(
                review, verdict="routine", by="model", model=answer
            )
        return dataclasses.replace(review, by="model", model=answer)
    if answer.error:
        return dataclasses.replace(
            review,
            verdict="high",
            categories=("unreviewed",),
            reasons=(f"melo could not review this call: {answer.error}.",),
            by="model",
            model=answer,
        )
    if answer.fits_request == "yes" and kind in GOAL_DECIDES:
        # On the goal, nothing high-impact: the goal decides (#465).
        return dataclasses.replace(review, verdict="routine", by="model", model=answer)
    plain = kind in ("none", "") or answer.user_asked == "yes"
    if (
        answer.verdict == "routine"
        and answer.fits_request == "yes"
        and allows
        and plain
    ):
        return dataclasses.replace(review, verdict="routine", by="model", model=answer)
    category = "opaque" if kind == "unclear" else kind
    if category not in TITLES or category in ("self", "none", "unreviewed"):
        category = "beyond_request" if answer.fits_request == "no" else "opaque"
    return dataclasses.replace(
        review,
        verdict="high",
        categories=(category,),
        reasons=(_model_reason(category),),
        by="model",
        model=answer,
    )


def answer_on_block(title: str) -> str:
    """The answer of a run auto mode stopped when the model wrote none."""
    what = (
        f"a step this needs ({title[:1].lower() + title[1:]})"
        if title
        else "a step this needs"
    )
    return (
        f"I stopped here: auto mode blocked {what}. You can allow it once from "
        "the chat, or switch this chat to Ask so that I ask you first."
    )
