"""The review of one tool call: routine, high-impact or undecided (#279).

The checker's rules (:mod:`localmelo.melo.checker.review`) classify each
``shell_exec``, ``python_exec`` and ``file_write`` call before an approval
policy decides. A session grant covers routine calls only: a high-impact
call asks every time and is never granted. Under a grant, and in auto
mode, the user's model reviews an undecided call (``by="model"``). In auto
mode (#339) nobody is asked: a call that is not routine is blocked, unless
the user allowed that exact step once (:func:`step_key`) or asked for it in
their own words. This module imports only the standard library.
"""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import json
import os
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal
from urllib.parse import unquote, urlsplit

Verdict = Literal["routine", "high", "undecided"]
RULES_VERSION: Final = 1

# The card's title per category, strongest first: a call's title and the
# order of its reasons follow this order.
TITLES: Final[dict[str, str]] = {
    "self": "Change localmelo itself?",
    "credentials": "Use your passwords or keys?",
    "system": "Change system settings?",
    "persistence": "Set up something that runs later?",
    "remote_code": "Run code downloaded from the internet?",
    "install": "Install software?",
    "restore": "Install the project's packages?",
    "env": "Create a programming environment?",
    "download": "Download files to this Mac?",
    "outside_write": "Change files outside {folder}?",
    "delete": "Delete files that cannot be restored?",
    "service": "Open a network service?",
    "send_out": "Send data from this Mac?",
    "opaque": "Run code melo cannot check?",
    "beyond_request": "Do something you did not ask for?",
    "unreviewed": "Run a step melo could not review?",
}
# Each category in a few words, for a blocked step's tag ("blocked: installs
# software") and the reason an auto-mode block records.
CATEGORY_WORDS: Final[dict[str, str]] = {
    "self": "changes localmelo itself",
    "credentials": "uses passwords or keys",
    "system": "changes system settings",
    "persistence": "sets up something that runs later",
    "remote_code": "runs code from the internet",
    "install": "installs software",
    "restore": "installs the project's packages",
    "env": "creates a programming environment",
    "download": "downloads files",
    "outside_write": "changes files outside the folder",
    "delete": "deletes files",
    "service": "opens a network service",
    "send_out": "sends data from this Mac",
    "opaque": "runs code melo cannot check",
    "beyond_request": "does something you did not ask for",
    "unreviewed": "could not be reviewed",
}
# Auto mode: the categories the user's own words can allow (they can name
# what the step acts on), and those only an "Allow once" of the exact step
# can. ``self`` is in neither: it is never allowed.
WORDS_CAN_ALLOW: Final = frozenset(
    {
        "install",
        "restore",
        "env",
        "download",
        "outside_write",
        "delete",
        "send_out",
        "service",
        "beyond_request",
    }
)
CLICK_ONLY: Final = frozenset(
    {"credentials", "system", "persistence", "remote_code", "opaque", "unreviewed"}
)
# The reason a call of the ``self`` category is denied without asking.
SELF_DENIAL: Final = "melo never lets a call change its own runtime, settings or keys"
# The tools whose arguments the rules read (melo.checker.review); another
# tool is routine to them, and auto mode makes one that is not read-only
# undecided.
RULED_TOOLS: Final = frozenset(["shell_exec", "python_exec", "file_write", "file_edit"])
# A model's category that can let a call run: none, or one words can allow.
MODEL_CAN_ALLOW: Final = frozenset({"none", ""}) | WORDS_CAN_ALLOW


@dataclass(frozen=True)
class ModelAnswer:
    """What the user's model answered about one call (see ``melo.checker.review.model``).

    ``verdict`` is ``routine`` or ``ask``; ``fits_request`` whether the call
    serves what the user asked (``yes``, ``no`` or ``unclear``);
    ``user_asked`` whether the user's own words ask for it; ``reason`` one
    sentence for the user; ``ms`` how long the review took. ``error`` names
    why there is no usable answer (a timeout, a failed request, a reply that
    does not parse, the turn's limit, a call too long to review), and the
    other fields are then empty.
    """

    verdict: str = ""
    category: str = ""
    fits_request: str = ""
    user_asked: str = ""
    reason: str = ""
    ms: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        if self.error:
            return {"error": self.error, "ms": self.ms}
        return {
            "verdict": self.verdict,
            "category": self.category,
            "fits_request": self.fits_request,
            "user_asked": self.user_asked,
            "reason": self.reason,
            "ms": self.ms,
        }


@dataclass(frozen=True)
class ActionReview:
    """What the rules found in one call, and the model when it was asked.

    ``categories`` are keys of :data:`TITLES`, strongest first; ``reasons``
    are at most three sentences for the person, each starting with a verb
    ("Installs playwright with pip."); ``paths`` are write paths outside
    the workspace that the call names. A review the model settled has
    ``by="model"`` and its :class:`ModelAnswer`; a high-impact call the
    user's words allowed keeps its categories with the verdict ``routine``.
    ``doubts`` are what the rules could not tell about a high-impact call
    (the reasons an undecided call has in ``reasons``): words never allow
    such a call. ``inputs`` name the local files whose content the rules
    read, each as ``path:digest``; :func:`step_key` includes them, so a
    changed file makes another step. ``changed`` names the files changed
    in this chat that a runner (``pytest``…) runs and melo could not read,
    as they are now: not part of the key, it tells one review of the step
    from the next.
    """

    verdict: Verdict
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    by: Literal["rules", "model"] = "rules"
    rules_version: int = RULES_VERSION
    model: ModelAnswer | None = None
    doubts: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "categories": list(self.categories),
            "reasons": list(self.reasons),
            "paths": list(self.paths),
            "by": self.by,
            "rules_version": self.rules_version,
        }
        if self.doubts:
            out["doubts"] = list(self.doubts)
        if self.model is not None:
            out["model"] = self.model.to_json()
        return out


def covered_by_grant(review: ActionReview | None) -> bool:
    """Whether a session grant of the call's tool lets it run unasked."""
    return review is None or review.verdict == "routine"


def offers_grant(review: ActionReview | None) -> bool:
    """Whether the card may offer "allow for the session": never for high impact."""
    return review is None or review.verdict != "high"


def review_title(review: ActionReview | None, folder: str) -> str:
    """The card's title for a high-impact call, or ``""``."""
    if review is None or review.verdict != "high" or not review.categories:
        return ""
    return TITLES.get(review.categories[0], "Run this?").format(folder=folder)


def because(review: ActionReview) -> str:
    """The reasons as one clause: "installs x; creates y" (no final stop)."""
    return "; ".join(r[:1].lower() + r[1:].rstrip(".") for r in review.reasons)


def block_words(review: ActionReview | None) -> str:
    """The first category in a few words ("installs software"), or ``""``."""
    if review is None or not review.categories:
        return ""
    return CATEGORY_WORDS.get(review.categories[0], "")


def _normalized(tool_name: str, arguments: Mapping[str, Any]) -> Any:
    """What a call does: Python as its syntax tree, a command as its exact text."""
    if tool_name == "python_exec":
        code = str(arguments.get("code") or "")
        try:
            return ["python", ast.dump(ast.parse(code))]
        except (SyntaxError, ValueError, RecursionError):
            return ["python-text", " ".join(code.split())]
    if tool_name == "shell_exec":
        command = arguments.get("command")
        # The exact text: a "#" inside a word, or a quote, changes what runs.
        if isinstance(command, list):
            return ["shell-argv", [str(part) for part in command]]
        return ["shell", str(command or "")]
    if tool_name in ("file_write", "file_edit"):
        rest = {k: v for k, v in arguments.items() if k != "path"}
        body = json.dumps(rest, sort_keys=True, ensure_ascii=False, default=str)
        path = os.path.realpath(str(arguments.get("path") or ""))
        return ["file", path, hashlib.sha256(body.encode()).hexdigest()]
    return [
        "call",
        json.loads(json.dumps(dict(arguments), sort_keys=True, default=str)),
    ]


def step_key(
    tool_name: str,
    arguments: Mapping[str, Any],
    workspace_root: str | None,
    inputs: Sequence[str] = (),
) -> str:
    """The hash of one step: the same call in the same folder, on the same files.

    Python is compared as its syntax tree, so a comment or a reflowed line
    is the same step; a shell command by its exact text; a file write by
    its resolved path and a hash of what it writes. *inputs* are the
    review's (:attr:`ActionReview.inputs`): the local files it read, with
    their digests. 32 hex characters.
    """
    root = os.path.realpath(workspace_root) if workspace_root else ""
    shape = [tool_name, root, _normalized(tool_name, arguments), sorted(inputs)]
    text = json.dumps(shape, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Steps alike (#411): keys that only ever refuse ──

# The calls whose first argument is a command a Python step runs.
_RUNS: Final = frozenset(
    {
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "system",
        "popen",
        "getoutput",
    }
)
# Flags that only make a command quieter: the same step with or without them.
_QUIET: Final = frozenset({"-q", "-qq", "-qqq", "--quiet", "--silent"})
_PYTHON: Final = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|py)$")
_PIP: Final = re.compile(r"^pip(?:\d+(?:\.\d+)*)?$")
_SHELL_SPLIT: Final = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
# What stands for melo's own Python in a step: the same program as "python".
_EXECUTABLE: Final = "sys.executable"


def _program(word: str) -> str:
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name == _EXECUTABLE or _PYTHON.match(name):
        return "python"
    return "pip" if _PIP.match(name) else name


def _command(words: Sequence[str]) -> list[str]:
    """*words* as a command alike: the program by name, ``python -m pip`` as
    ``pip``, quoting and spacing gone, flags that only quieten it dropped."""
    tokens = [w.strip() for w in words if w.strip()]
    if not tokens:
        return []
    tokens[0] = _program(tokens[0])
    if tokens[0] == "python" and tokens[1:3] == ["-m", "pip"]:
        tokens = ["pip", *tokens[3:]]
    elif tokens[0] == "uv" and tokens[1:2] == ["pip"]:
        tokens = ["pip", *tokens[2:]]
    return [t for t in tokens if t not in _QUIET]


def _shell_commands(text: str) -> list[list[str]]:
    found = []
    for part in _SHELL_SPLIT.split(text):
        try:
            words = shlex.split(part)
        except ValueError:
            words = part.split()
        if words:
            found.append(words)
    return found


def _word(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        dotted = f"{node.value.id}.{node.attr}"
        return dotted if dotted == _EXECUTABLE else None
    return None


def _python_commands(code: str) -> list[list[str]]:
    """The commands a Python step runs with literal arguments: ``subprocess.run([...])``,
    ``os.system("...")`` and the like."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    found: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _RUNS:
            continue
        first = node.args[0]
        if isinstance(first, ast.List | ast.Tuple):
            words = [_word(item) for item in first.elts]
            if words and all(word is not None for word in words):
                found.append([str(word) for word in words])
        elif (text := _word(first)) is not None:
            found += _shell_commands(text)
    return found


def _installs(tokens: Sequence[str]) -> bool:
    return len(tokens) > 2 and tokens[1] == "install"


def loose_keys(
    tool_name: str, arguments: Mapping[str, Any], categories: Sequence[str] = ()
) -> list[str]:
    """Keys that make two calls the same step though their text differs (#411).

    One for the whole list of commands the call runs, alike when only
    spacing, quoting, the program's spelling (``pip3``, ``python -m pip``) or a
    quietening flag differ. A part of it has keys of its own only when it is
    what was blocked: for an install (*categories* has ``install``, or is
    empty: the call is not reviewed yet), each install part, and one key per
    program and what it installs. So a blocked ``pip install numpy && python
    fetch.py`` refuses a later ``pip3 install numpy`` but not ``python
    fetch.py``. They only ever refuse: no allowance is kept by them.
    """
    if tool_name == "shell_exec":
        command = arguments.get("command")
        if isinstance(command, list):
            commands = [[str(word) for word in command]]
        else:
            commands = _shell_commands(str(command or ""))
    elif tool_name == "python_exec":
        commands = _python_commands(str(arguments.get("code") or ""))
    else:
        return []
    parts = [tokens for tokens in (_command(words) for words in commands) if tokens]
    if not parts:
        return []

    def key(text: str) -> str:
        return "cmd:" + hashlib.sha256(text.encode()).hexdigest()[:32]

    keys = {key(" && ".join(" ".join(tokens) for tokens in parts))}
    if not categories or "install" in categories:
        for tokens in filter(_installs, parts):
            keys.add(key(" ".join(tokens)))
            targets = sorted({t.lower() for t in tokens[2:] if not t.startswith("-")})
            if targets:
                keys.add(f"install:{tokens[0]}:{','.join(targets)}")
    return sorted(keys)


# ── The sites a step names (#433): keys that only ever refuse ──

# The loose key of a site: ``ask:site:<host>`` (approvals.site_key).
SITE_KEY: Final = "ask:site:"
# A step naming an http(s) address whose encoded host melo cannot read as
# one site (a percent-encoding or an international name that does not
# decode): refused wherever a site was refused, never anywhere else.
UNREADABLE_SITE: Final = "ask:site-unreadable"
_ADDRESS: Final = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
# What a host built at run time has: a variable, a template, a format
# placeholder (``%s``, a ``%`` without two hex digits), an f-string's value.
_BUILT: Final = re.compile(r"[$`{}\x00]|%(?![0-9a-fA-F]{2})")
# A host melo reads as one site, once encoded: labels of letters, digits,
# ``-`` and ``_``.
_HOST_NAME: Final = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")
# One brace group a shell (or curl) expands into words: ``{a,b}``.
_BRACES: Final = re.compile(r"^([^{}]*)\{([^{}]*,[^{}]*)\}([^{}]*)$")
# Characters after a host that end a sentence or a call, not the host.
_HOST_END: Final = ")],;!'\""
# A value an f-string fills in: no host is read through it.
_FILLED: Final = "\x00"


def site_host(host: str) -> str | None:
    """*host* as one site: in ASCII (IDNA), lower case, without one final dot.

    An IP address is written the one standard way (``::1``). ``None`` when
    it is empty or cannot be encoded. What
    :func:`~localmelo.melo.contracts.approvals.site_key` uses, so one site
    has one key however its name is spelt.
    """
    try:
        if not host.isascii():
            host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    host = host.lower()
    host = host[:-1] if host.endswith(".") else host
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host or None


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def url_site(url: str) -> str | None:
    """The site of an http(s) *url* (:func:`site_host`), or ``None``."""
    try:
        parts = urlsplit(url.strip())
        host = site_host(parts.hostname or "")
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    return host


def _raw_hosts(address: str) -> set[str]:
    """The host texts *address* (``http(s)://…``) can reach.

    Clients read an address differently: a ``\\`` ends the host for some
    and not for others, and an ``@`` that appears twice is split at the
    first or the last. Each reading is kept, so a refused site cannot hide
    behind one of them.
    """
    rest = address.split("://", 1)[1]
    hosts: set[str] = set()
    for authority in {re.split(r"[/?#]", rest)[0], re.split(r"[/?#\\]", rest)[0]}:
        for hostport in {authority.rpartition("@")[2], authority.partition("@")[2]}:
            if not hostport:
                continue
            if hostport.startswith("["):
                hosts.add(hostport[1:].partition("]")[0])
            else:
                hosts.add(hostport.partition(":")[0])
    return hosts


def _host_key(raw: str) -> str | None:
    """The loose key of one host text, :data:`UNREADABLE_SITE`, or ``None`` (unknown)."""
    raw = raw.rstrip(_HOST_END)
    if not raw:
        return None
    if _BUILT.search(raw):
        return None  # built at run time: the review judges it
    encoded = "%" in raw
    if encoded:
        try:
            raw = unquote(raw, errors="strict")
        except UnicodeDecodeError:
            return UNREADABLE_SITE
    host = site_host(raw)
    if host is None:
        return UNREADABLE_SITE if encoded or not raw.isascii() else None
    if not (_HOST_NAME.fullmatch(host) or _is_ip(host)):
        # Not a name a client resolves (a pattern, a stray character):
        # unknown, unless an encoding hid what it is.
        return UNREADABLE_SITE if encoded else None
    return SITE_KEY + host


def _expanded(raw: str) -> list[str]:
    """*raw* with one brace group expanded, as a shell or curl does: ``{a,b}.x``."""
    found = _BRACES.match(raw)
    if found is None:
        return [raw]
    head, choices, tail = found.groups()
    return [head + choice + tail for choice in choices.split(",")]


def _address_keys(text: str) -> Iterator[str]:
    for address in _ADDRESS.findall(text):
        for raw in _raw_hosts(address):
            for host in _expanded(raw):
                key = _host_key(host)
                if key is not None:
                    yield key


# ── Network steps: scheme-less curl and wget addresses, hosts melo cannot read ──

# The options of curl and wget whose value is the next word (or joined to
# them): that word is never an address.
_CURL_SHORT: Final = frozenset("AbcCdDeEFHKmoPQrtTuUwxXyYz")
_CURL_LONG_NAMES: Final = """abstract-unix-socket alt-svc aws-sigv4 cacert capath cert cert-type
    ciphers config connect-timeout connect-to continue-at cookie cookie-jar
    create-file-mode crlfile curves data data-ascii data-binary data-raw
    data-urlencode delegation dns-interface dns-ipv4-addr dns-ipv6-addr
    dns-servers doh-url dump-header ech engine etag-compare etag-save
    expect100-timeout form form-string ftp-account ftp-alternative-to-user
    ftp-method ftp-port ftp-ssl-ccc-mode happy-eyeballs-timeout-ms
    haproxy-clientip header hostpubmd5 hostpubsha256 hsts interface ip-tos
    ipfs-gateway json keepalive-cnt keepalive-time key key-type krb libcurl
    limit-rate local-port login-options mail-auth mail-from mail-rcpt
    max-filesize max-redirs max-time netrc-file noproxy oauth2-bearer output
    output-dir parallel-max pass pinnedpubkey preproxy proto proto-default
    proto-redir proxy proxy-cacert proxy-capath proxy-cert proxy-cert-type
    proxy-ciphers proxy-crlfile proxy-header proxy-key proxy-key-type
    proxy-pass proxy-pinnedpubkey proxy-service-name proxy-tls13-ciphers
    proxy-tlsauthtype proxy-tlspassword proxy-tlsuser proxy-user proxy1.0
    pubkey quote random-file range rate referer request request-target
    resolve retry retry-delay retry-max-time sasl-authzid service-name socks4
    socks4a socks5 socks5-gssapi-service socks5-hostname speed-limit
    speed-time stderr telnet-option tftp-blksize time-cond tls-max
    tls13-ciphers tlsauthtype tlspassword tlsuser trace trace-ascii
    trace-config unix-socket upload-file url url-query user user-agent
    variable write-out"""
_CURL_LONG: Final = frozenset(_CURL_LONG_NAMES.split())
_WGET_SHORT: Final = frozenset("aABDeiIloOPQRtTUwX")
_WGET_LONG_NAMES: Final = """accept accept-regex append-output base bind-address bind-dns-address
    body-data body-file ca-certificate ca-directory certificate
    certificate-type config connect-timeout crl-file cut-dirs default-page
    directory-prefix dns-servers dns-timeout domains egd-file exclude-directories
    exclude-domains execute follow-tags ftp-password ftp-user header
    hsts-file http-password http-user ignore-tags include-directories
    input-file level limit-rate load-cookies local-encoding max-redirect
    method output-document output-file password pinnedpubkey post-data
    post-file prefer-family private-key private-key-type progress
    proxy-password proxy-user quota random-file read-timeout referer reject
    reject-regex rejected-log regex-type remote-encoding report-speed
    restrict-file-names save-cookies secure-protocol timeout tries
    use-askpass user user-agent wait waitretry warc-cdx warc-dedup warc-file
    warc-header warc-max-size warc-tempdir"""
_WGET_LONG: Final = frozenset(_WGET_LONG_NAMES.split())
# Options that take the addresses from a file: melo cannot read them.
_FROM_FILE: Final = {"curl": ("K", "config"), "wget": ("i", "input-file")}
# Options that only print something: no address is needed.
_INFO: Final = frozenset({"version", "help", "manual", "V", "h", "M"})
# A word curl or wget reads as an address without a scheme: a dotted host,
# perhaps a user before it, a port and a path after it.
_BARE: Final = re.compile(
    r"(?:[^\s/?#@]+@)?([^\s/?#@:\[\]]+\.[^\s/?#@:\[\].]+\.?)(?::\d+)?(?:[/?#]\S*)?"
)
# Text a shell or Python fills in when the step runs.
_EXPANDS: Final = re.compile(r"[$`\x00]")
# Programs that reach other computers whose targets melo does not key.
_NETWORK: Final = frozenset(
    {"nc", "ncat", "netcat", "telnet", "socat", "websocat", "ssh", "scp", "sftp"}
    | {"rsync", "ftp", "aria2c", "http", "https", "xh"}
)
_GIT_NETWORK: Final = frozenset({"clone", "fetch", "pull", "push", "ls-remote"})
_SHELLS: Final = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
# Words that run the command after them; the value of each option given.
_WRAPPERS: Final[dict[str, frozenset[str]]] = {
    "command": frozenset(),
    "exec": frozenset(),
    "time": frozenset(),
    "nohup": frozenset(),
    "env": frozenset({"-u", "-C", "-S"}),
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}),
    "xargs": frozenset({"-n", "-I", "-P", "-L", "-d", "-E", "-s", "-a", "-J", "-R"}),
    "nice": frozenset({"-n"}),
    "timeout": frozenset({"-s", "-k"}),
}
_ASSIGNMENT: Final = re.compile(r"[A-Za-z_]\w*=")
# Python: calls that fetch an address, and calls that connect to a host.
_PY_URL_CALLS: Final = frozenset({"urlopen", "urlretrieve", "Request"})
_PY_METHODS: Final = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream"}
)
_PY_CLIENT_MODULES: Final = frozenset({"requests", "httpx", "urllib3", "aiohttp"})
_PY_CLIENTS: Final = frozenset(
    {"Session", "Client", "AsyncClient", "PoolManager", "ClientSession"}
)
_PY_HOST_CALLS: Final = frozenset(
    {"HTTPConnection", "HTTPSConnection", "create_connection", "open_connection"}
)
_DEPTH: Final = 3

Sites = tuple[set[str], bool]


def _shell_words(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(word) for word in command]
    text = str(command or "")
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _unwrapped(words: Sequence[str]) -> list[str]:
    """*words* from the program they run: assignments and wrappers (``sudo``, ``xargs``) dropped."""
    rest = list(words)
    while rest:
        if _ASSIGNMENT.match(rest[0]):
            rest = rest[1:]
            continue
        wrapper = _program(rest[0])
        valued = _WRAPPERS.get(wrapper)
        if valued is None:
            break
        rest = rest[1:]
        while rest and rest[0].startswith("-") and rest[0] != "-":
            rest = rest[2:] if rest[0] in valued else rest[1:]
        if rest and wrapper == "timeout":
            rest = rest[1:]  # its duration
    return rest


def _fetcher_sites(program: str, args: Sequence[str]) -> Sites:
    """What a ``curl`` or ``wget`` command reaches: keys of its scheme-less addresses."""
    curl = program == "curl"
    short, long_ = (_CURL_SHORT, _CURL_LONG) if curl else (_WGET_SHORT, _WGET_LONG)
    file_short, file_long = _FROM_FILE[program]
    targets: list[str] = []
    from_file, info_only, ended, index = False, True, False, 0
    while index < len(args):
        arg = args[index]
        index += 1
        if ended or arg == "-" or not arg.startswith("-"):
            targets.append(arg)
        elif arg == "--":
            ended = True
        elif arg.startswith("--"):
            name, joined, value = arg[2:].partition("=")
            info_only = info_only and name in _INFO
            from_file = from_file or name == file_long
            if name in long_ and not joined:
                value = args[index] if index < len(args) else ""
                index += 1
            if curl and name == "url":
                targets.append(value)
        else:
            info_only = info_only and all(c in _INFO for c in arg[1:])
            for at, letter in enumerate(arg[1:], 1):
                from_file = from_file or letter == file_short
                if letter in short:
                    index += at == len(arg) - 1  # its value is the next word
                    break
    keys: set[str] = set()
    unread = from_file or not targets and not info_only
    for target in targets:
        if _EXPANDS.search(target):
            unread = True
        elif "://" not in target and _BARE.fullmatch(target):
            keys.update(_address_keys("http://" + target))
    return keys, unread


def _command_sites(words: Sequence[str], depth: int) -> Sites:
    words = _unwrapped(words)
    if not words:
        return set(), False
    program, args = _program(words[0]), list(words[1:])
    if program in ("curl", "wget"):
        return _fetcher_sites(program, args)
    if "-c" in args[:-1] and (program in _SHELLS or program == "python"):
        code = args[args.index("-c") + 1]
        if depth >= _DEPTH:
            return set(), bool(_EXPANDS.search(code))
        if program == "python":
            return _python_sites(code, depth + 1)
        return _shell_sites(code, depth + 1)
    plain = [arg for arg in args if not arg.startswith("-")]
    if (
        program in _NETWORK
        or program == "git"
        and plain[:1]
        and plain[0] in _GIT_NETWORK
    ):
        return set(), any(_EXPANDS.search(arg) for arg in args)
    return set(), False


def _shell_sites(command: Any, depth: int = 0) -> Sites:
    words = _shell_words(command)
    keys = {key for word in words for key in _address_keys(word)}
    unread = False
    commands = (
        [words] if isinstance(command, list) else _shell_commands(str(command or ""))
    )
    for found in commands:
        more, hidden = _command_sites(found, depth)
        keys |= more
        unread = unread or hidden
    return keys, unread


def _python_texts(code: str) -> list[str]:
    """The literal strings of *code*: an f-string with each value it fills in marked."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []  # it does not run
    return _texts(tree)


def _texts(tree: ast.AST) -> list[str]:
    texts: list[str] = []
    parts: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts.update(id(value) for value in node.values)
            texts.append(_marked(node))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in parts:
            continue
        if isinstance(node.value, str):
            texts.append(node.value)
        elif isinstance(node.value, bytes):
            texts.append(node.value.decode("latin-1"))
    return texts


def _marked(node: ast.JoinedStr) -> str:
    """An f-string's text, each value it fills in as :data:`_FILLED`."""
    return "".join(
        (
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else _FILLED
        )
        for value in node.values
    )


def _call_name(node: ast.Call) -> tuple[str, ast.expr | None]:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr, func.value
    return (func.id, None) if isinstance(func, ast.Name) else ("", None)


class _PythonNames:
    """What the names of a snippet hold: one literal each, and the clients made."""

    def __init__(self, tree: ast.AST) -> None:
        bound: dict[str, list[ast.expr | None]] = {}
        self.clients: set[str] = set()
        self.sockets: set[str] = set()
        for node in ast.walk(tree):
            pairs: list[tuple[ast.expr, ast.expr | None]] = []
            if isinstance(node, ast.Assign):
                pairs = [(target, node.value) for target in node.targets]
            elif isinstance(node, ast.AnnAssign | ast.AugAssign):
                pairs = [(node.target, None)]
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                pairs = [(node.optional_vars, node.context_expr)]
            elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
                pairs = [(node.target, None)]
            elif isinstance(node, ast.NamedExpr):
                pairs = [(node.target, node.value)]
            for target, value in pairs:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        single = value if target is name else None
                        bound.setdefault(name.id, []).append(single)
                if isinstance(target, ast.Name) and isinstance(value, ast.Call):
                    made = _call_name(value)[0]
                    if made in _PY_CLIENTS:
                        self.clients.add(target.id)
                    elif made == "socket":
                        self.sockets.add(target.id)
        self.values = {
            name: values[0] for name, values in bound.items() if len(values) == 1
        }

    def text(self, node: ast.expr | None, depth: int = 0) -> str | None:
        """The literal text *node* holds (an f-string's values marked), or ``None``."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return _marked(node)
        if isinstance(node, ast.Name) and depth < _DEPTH:
            return self.text(self.values.get(node.id), depth + 1)
        return None

    def client(self, node: ast.expr | None) -> bool:
        if isinstance(node, ast.Name):
            return node.id in _PY_CLIENT_MODULES or node.id in self.clients
        return isinstance(node, ast.Call) and _call_name(node)[0] in _PY_CLIENTS


def _argument(node: ast.Call, position: int, *names: str) -> ast.expr | None:
    if len(node.args) > position:
        return node.args[position]
    return next((k.value for k in node.keywords if k.arg in names), None)


def _python_call_sites(node: ast.Call, names: _PythonNames, depth: int) -> Sites:
    name, receiver = _call_name(node)
    if name in _RUNS and node.args:
        first = node.args[0]
        if isinstance(first, ast.List | ast.Tuple):
            words = [names.text(item) for item in first.elts]
            return _command_sites(
                [w if w is not None else _FILLED for w in words], depth
            )
        command = names.text(first)
        return _shell_sites(command, depth) if command is not None else (set(), False)
    if name in _PY_URL_CALLS or name in _PY_METHODS and names.client(receiver):
        position = 1 if name == "request" else 0
        url = names.text(_argument(node, position, "url"))
        if url is None:
            return set(), True
        keys = set(_address_keys(url))
        return keys, "://" in url and not keys
    is_socket = isinstance(receiver, ast.Name) and receiver.id in names.sockets
    if name in _PY_HOST_CALLS or name == "connect" and is_socket:
        target = _argument(node, 0, "host", "address")
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[0]
        host = names.text(target)
        if host is None or _EXPANDS.search(host):
            return set(), True
        key = _host_key(host)
        return ({key} if key is not None else set()), False
    return set(), False


def _python_sites(code: str, depth: int = 0) -> Sites:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return set(), False  # it does not run
    keys = {key for text in _texts(tree) for key in _address_keys(text)}
    unread = False
    names = _PythonNames(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            more, hidden = _python_call_sites(node, names, depth)
            keys |= more
            unread = unread or hidden
    return keys, unread


def _sites(tool_name: str, arguments: Mapping[str, Any]) -> Sites:
    if tool_name == "shell_exec":
        return _shell_sites(arguments.get("command"))
    if tool_name == "python_exec":
        return _python_sites(str(arguments.get("code") or ""))
    return set(), False


def site_keys(tool_name: str, arguments: Mapping[str, Any]) -> list[str]:
    """The loose keys of the sites a ``shell_exec`` or ``python_exec`` step names (#433).

    One ``ask:site:<host>`` (:func:`site_host`) per address in it: each
    literal ``http://`` or ``https://`` address in a word of a command or a
    string of Python code (an address passed to ``urllib``, ``requests`` or
    ``httpx``, kept in a variable, or in an f-string whose host is written
    out); each address ``curl`` or ``wget`` is given without a scheme (a
    word that is not an option's value and is a dotted host, perhaps with a
    port and a path); and the literal host of ``http.client``'s
    connections and of ``socket.create_connection``. A percent-encoded host
    is decoded first; one that does not decode into a site gives
    :data:`UNREADABLE_SITE`. A host built at run time (``$HOST``,
    ``{host}``, ``%s``) gives nothing (see :func:`reaches_unread_site`).
    They only ever refuse a step that names a site refused earlier
    (:class:`~localmelo.melo.contracts.approvals.EarlierBlocks`): a blocked
    step never keeps them, since what was refused was the step.
    """
    return sorted(_sites(tool_name, arguments)[0])


def reaches_unread_site(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    """Whether a ``shell_exec`` or ``python_exec`` step reaches a host melo cannot read.

    ``curl`` or ``wget`` with an address the shell fills in (``$URL``,
    ``https://$H/``), one read from a file (``-K``, ``-i``) or none at all
    (``xargs curl``); another network program (``nc``, ``ssh``, ``git
    clone``…) with an argument the shell fills in; Python that fetches an
    address or connects to a host not written out in the code (a variable
    from elsewhere, text joined at run time, an f-string that fills in the
    host), or runs ``curl`` with such an argument. While a site is refused
    in the task, such a step is reviewed instead of running as routine
    (#433).
    """
    return _sites(tool_name, arguments)[1]
