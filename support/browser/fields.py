"""Fields melo never types into: passwords, payment cards and one-time codes.

A page can dress a secret field as an ordinary one, so the decision does not
rest on ``type=password`` alone. A text box is refused when any of these says
it holds a secret:

- its ``type`` is ``password``, or the page masks what is typed in it with CSS
  (``-webkit-text-security``);
- its ``autocomplete`` names a password (``current-password``,
  ``new-password``), a one-time code (``one-time-code``) or a card
  (``cc-number``, ``cc-csc``, any ``cc-*``);
- the words that name it (its name, id, ``aria-label``, ``aria-labelledby``
  and ``aria-describedby`` text, title, placeholder, the text of its labels, its
  fieldset's legend and the text just before it) say password, PIN (also
  written into another word: "PINcode"), card, CVC or CVV with a digit after
  it, expiry, security code, verification code, token or a bare "code" (also
  at the end of a longer word: "Bestätigungscode"), in English, Chinese and a
  few other languages (a postal, promo or country code is not one);
- it is a box for one character (a split code), a numeric, telephone or
  number field of four to eight characters, or a short box of four to eight
  characters whose names say nothing ("q", "f1"): the shape of a code. The page script
  takes the length from a digits-only ``pattern`` or an all-9s ``max`` when the
  box has no ``maxlength``, and a digits-only ``pattern`` counts as numeric.

When it is unclear, the field counts as secret: melo asks the person to fill
it in, which costs a sentence, and typing into a code field costs more.

The page script reads these facts into a :class:`FieldAttrs` and the decision
is made here, once when the page is read and again at the moment melo types,
on what the page says then. The words are matched on a normalized copy:
``userPassword`` and ``user_password`` both read ``user password``. The page
script applies the same rules from :data:`JS_RULES`, which is built from the
patterns below, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable

from localmelo.support.browser.driver import (
    Element,
    FieldAttrs,
    Sensitive,
)

_PASSWORD_AUTOCOMPLETE = frozenset({"current-password", "new-password"})
PASSWORD_WORDS = (
    r"password|passwd|passphrase|passcode|pwd|\bpin\b|密码|口令|passwort"
    r"|contraseña|contrasena|mot de passe|パスワード|비밀번호"
    # PIN written together with another word: "PINcode", "atmpin", "newpin".
    r"|\bpin(code|number|num|no|entry|input|pad|feld|field)\b"
    r"|(my|new|old|atm|bank|sim|card|current|confirm|enter|user|login)pin\b"
)
CARD_WORDS = (
    r"card ?(number|num|no)\b|cardnumber|credit ?card|debit ?card|\bcard\b|\bcards\b"
    r"|\bcc ?(num|number|no|exp)|\bcv[cvn]\d?\b|\bcsc\b|\bcid\b"
    r"|expir|\bexp ?(date|month|year|mm|yy)|\bmm ?yy\b|\bmm ?yyyy\b"
    r"|kartennummer|\bkarte\b|numero de carte|numéro de carte|tarjeta|carta di credito"
    r"|卡号|银行卡|信用卡|有效期|カード番号|카드번호"
)
CODE_WORDS = (
    r"\botp\b|one ?time|\b2fa\b|\bmfa\b|\btotp\b|\bhotp\b|verification|verify|\bverif"
    r"|security code|auth(entication|enticator)? ?(code|token|app)?\b|sms code"
    r"|confirm(ation)? code|login code|sign ?in code|access code|two ?step|2 ?step"
    r"|two ?factor|2 ?factor|passcode|one time"
    r"|验证码|校验码|动态码|认证码|安全码|验证|驗證碼|驗證|二次验证|认证|確認コード|인증"
)
# A bare "code" or "token" is secret unless it is plainly another sort of code. A
# word that ends in one ("Bestätigungscode", "csrftoken") is the same thing, and
# so is the word for code in a few other languages.
BARE_CODE_WORDS = (
    r"\bcodes?\b|\btokens?\b|\bkey\b|\w{3,}codes?\b|\w{3,}tokens?\b"
    r"|código|codigo|codice|\bkod\b|コード|코드|代码|代碼"
)
BENIGN_CODE_WORDS = (
    r"zip|postal|post ?code|promo|coupon|voucher|discount|referral|invit|area"
    r"|country|dial|language|colou?r|\bhex|source|captcha|barcode|bar code|qr"
    r"|product|item code|sku|isbn|api key|ssh key|public key|unicode|geo|currenc"
    r"|plz|postleitzahl|邮编|邮政|优惠|折扣|邀请|源代码|源碼|国家|地区|区号"
)
_PASSWORD = re.compile(PASSWORD_WORDS, re.ASCII)
_CARD = re.compile(CARD_WORDS, re.ASCII)
_CODE = re.compile(CODE_WORDS, re.ASCII)
_BARE_CODE = re.compile(BARE_CODE_WORDS, re.ASCII)
_BENIGN_CODE = re.compile(BENIGN_CODE_WORDS, re.ASCII)
_CAMEL = re.compile(r"([a-z])([A-Z])")
_NOT_WORDS = re.compile(r"[\W_]+")
_DIGITS_INPUTMODE = frozenset({"numeric", "decimal", "tel"})
_DIGITS_TYPE = frozenset({"tel", "number"})
_BOX_TYPES = frozenset({"", "text", "tel", "number"})
# Words that name a field: three Latin letters in a row, or any letter of another script
# ("q", "f1" or an empty name say nothing).
NAMED_WORDS = r"[a-z]{3,}|[^\x00-\x7f]"
_NAMED = re.compile(NAMED_WORDS)

# What the plain reason says, by kind. The text box is named by what it is for,
# never by the page's words for it.
REASONS: dict[str, str] = {
    "password": "a password field",
    "code": "a one-time or verification code field",
    "card": "a payment field",
}


def normalized(words: str) -> str:
    """*words* in lower case with ``camelCase`` and punctuation spaced out."""
    return _NOT_WORDS.sub(" ", _CAMEL.sub(r"\1 \2", words)).lower().strip()


def sensitive_kind(attrs: FieldAttrs | None) -> Sensitive:
    """Why melo never types in a text box described by *attrs*: ``""`` for an ordinary one."""
    if attrs is None:
        return ""
    tokens = attrs.autocomplete.lower().split()
    kind = attrs.type.lower()
    if (
        kind == "password"
        or attrs.masked
        or _PASSWORD_AUTOCOMPLETE.intersection(tokens)
    ):
        return "password"
    if "one-time-code" in tokens:
        return "code"
    if any(token.startswith("cc-") for token in tokens):
        return "card"
    words = normalized(attrs.words)
    if _PASSWORD.search(words):
        return "password"
    if _CARD.search(words):
        return "card"
    if _CODE.search(words):
        return "code"
    if _BARE_CODE.search(words) and not _BENIGN_CODE.search(words):
        return "code"
    digits = attrs.inputmode.lower() in _DIGITS_INPUTMODE or kind in _DIGITS_TYPE
    if attrs.maxlength == 1:  # one box per character of a split code
        return "code"
    if digits and 4 <= attrs.maxlength <= 8:
        return "code"
    if not _NAMED.search(words) and 4 <= attrs.maxlength <= 8 and kind in _BOX_TYPES:
        return "code"  # a short box that names nothing: shaped like a code
    return ""


# New lines, tabs and the two joiners (emoji and a few scripts need them) show or join.
_SHOWN_ANYWAY = frozenset("\n\t\u200c\u200d")
_UNSHOWN = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
# Characters outside those categories that draw nothing or nearly nothing: the Hangul
# fillers (U+3164, U+FFA0), the blank Braille cell (U+2800) and the combining grapheme
# joiner (U+034F).
_BLANK = frozenset("\u3164\uffa0\u2800\u034f")
# Variation selectors change how the character before them is drawn and have no look of
# their own: U+FE00-FE0F and U+E0100-E01EF.
_VARIATION = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))


def _invisible(char: str) -> bool:
    if char in _SHOWN_ANYWAY:
        return False
    if char in _BLANK or unicodedata.category(char) in _UNSHOWN:
        return True
    code = ord(char)
    return any(low <= code <= high for low, high in _VARIATION)


def unshowable(text: str) -> str:
    """What in *text* a question could not show a person, in words; ``""`` when all of it shows.

    These would make the question say less than what is typed, so a text with
    one is not typed: control, format (zero-width spaces, direction overrides,
    a byte-order mark, tag), surrogate, private-use and unassigned characters,
    line and paragraph separators, the Hangul fillers U+3164 and U+FFA0, the
    blank Braille cell U+2800, the combining grapheme joiner U+034F and the
    variation selectors (U+FE00-FE0F, U+E0100-E01EF). New lines, tabs and the
    two joiners U+200C and U+200D are allowed.
    """
    bad = [c for c in text if _invisible(c)]
    return f"{len(bad)} control or invisible character(s)" if bad else ""


HIDDEN = "(hidden)"
_TOKEN_LIKE = re.compile(r"[A-Za-z0-9+/=_\-.]{24,}")


def form_pairs(rows: list[object]) -> tuple[tuple[str, str], ...]:
    """The fields of a form as ``(name, value)`` pairs the person may be shown.

    A hidden field, a field that looks secret (by its name, as for typing) and a
    value that looks like a token (a long run of letters, digits and signs with
    no spaces) show as ``(hidden)``: the question about sending a form says
    what goes, never a secret. The page script already leaves out the value of a
    field it took for secret (flag ``s``) and of a hidden one (flag ``h``).
    """
    out: list[tuple[str, str]] = []
    for row in rows[:12]:
        if not isinstance(row, list) or len(row) < 2:
            continue
        name, value = str(row[0])[:40], str(row[1])
        flagged = len(row) > 2 and bool(row[2])
        if (
            flagged
            or _TOKEN_LIKE.fullmatch(value)
            or sensitive_kind(FieldAttrs(words=name))
        ):
            value = HIDDEN
        out.append((name, value[:80]))
    return tuple(out)


# "search" in what names a box, but not "research".
_SEARCH_WORDS = re.compile(r"(?<!re)search|搜索")


def is_search(attrs: FieldAttrs | None) -> bool:
    """Whether the page describes a text box as a search box: ``type=search``, or "search" in the words that name it (#547).

    A page can describe any box so; the person's own words must also ask to
    search before it counts for anything
    (:meth:`~localmelo.melo.agent._step_review.StepReviewer.searching`).
    """
    if attrs is None:
        return False
    return attrs.type.lower() == "search" or bool(
        _SEARCH_WORDS.search(normalized(attrs.words))
    )


def searches(item: Element, elements: Iterable[Element]) -> bool:
    """Whether what *item* types or sends is a search: every text box of its form is a search box, or it is a search box in no form (#547).

    One box that is no search box (a comment next to a search field) makes the
    form a message, so it is reviewed like one.
    """
    if item.form:
        boxes = [
            box for box in elements if box.kind == "textbox" and box.form == item.form
        ]
    else:
        boxes = [item] if item.kind == "textbox" else []
    return bool(boxes) and all(is_search(box.attrs) for box in boxes)


# Words of a control that agrees to something for the person: terms, consent, a
# newsletter, or a sign-in with another account. A click on one by a point is asked
# about, whatever was allowed before.
CONSENT_WORDS = (
    r"\bagree|\baccept|\bconsent|\bterms\b|terms of|\bconditions\b|privacy|\bpolicy\b"
    r"|i have read|i\'ve read|subscribe|newsletter|marketing|opt ?in"
    r"|(sign|log) ?in with|continue with|authori[sz]e"
    r"|同意|接受|条款|协议|隐私|订阅|授权|规约|約款|동의|약관"
)
_CONSENT = re.compile(CONSENT_WORDS, re.IGNORECASE)


def consent(words: str) -> bool:
    """Whether a control's *words* agree to terms, consent, marketing or a sign-in with another account."""
    return bool(_CONSENT.search(words))


# Words of a box that makes what the person does public or shares it: "make my
# activity public", "share to social media". A page may tick one before the person
# sees the form; melo does not send the form with it ticked unless the person asked
# to share (#547).
PUBLIC_WORDS = (
    r"\bpublic(ly)?\b|\bshar(e|es|ed|ing)\b|\bsocial\b|\bfacebook\b|\btwitter\b"
    r"|\bcross-?post|\bpost (to|on) |公开|分享|朋友圈|同步到|社交"
)
_PUBLIC = re.compile(PUBLIC_WORDS, re.IGNORECASE)


def public(words: str) -> bool:
    """Whether a box's *words* make what the person does public or share it."""
    return bool(_PUBLIC.search(words))


def _js(pattern: str) -> str:
    return f"new RegExp({json.dumps(pattern)})"


# The same rules as :func:`sensitive_kind`, for the page script. ``secretKind(a)``
# takes the object the script builds for :class:`FieldAttrs` and answers "", "password",
# "code" or "card".
JS_RULES = (
    "var RX_PASSWORD = "
    + _js(PASSWORD_WORDS)
    + ", RX_CARD = "
    + _js(CARD_WORDS)
    + ", RX_CODE = "
    + _js(CODE_WORDS)
    + ", RX_BARE = "
    + _js(BARE_CODE_WORDS)
    + ", RX_BENIGN = "
    + _js(BENIGN_CODE_WORDS)
    + ", RX_NAMED = "
    + _js(NAMED_WORDS)
    + ";\n"
    + r"""
var wordsOf = function (s) {
  return String(s).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/[^\p{L}\p{N}]+/gu, " ").toLowerCase().trim();
};
var secretKind = function (a) {
  var tokens = String(a.autocomplete || "").toLowerCase().split(/\s+/).filter(Boolean);
  var kind = String(a.type || "").toLowerCase();
  if (kind === "password" || a.masked || tokens.indexOf("current-password") >= 0 || tokens.indexOf("new-password") >= 0) return "password";
  if (tokens.indexOf("one-time-code") >= 0) return "code";
  if (tokens.some(function (t) { return t.indexOf("cc-") === 0; })) return "card";
  var words = wordsOf(a.words || "");
  if (RX_PASSWORD.test(words)) return "password";
  if (RX_CARD.test(words)) return "card";
  if (RX_CODE.test(words)) return "code";
  if (RX_BARE.test(words) && !RX_BENIGN.test(words)) return "code";
  var mode = String(a.inputmode || "").toLowerCase();
  var digits = mode === "numeric" || mode === "decimal" || mode === "tel" || kind === "tel" || kind === "number";
  var max = a.maxlength || 0;
  if (max === 1) return "code";
  if (digits && max >= 4 && max <= 8) return "code";
  if (!RX_NAMED.test(words) && max >= 4 && max <= 8 && /^(|text|tel|number)$/.test(kind)) return "code";
  return "";
};
"""
)


__all__ = [
    "CONSENT_WORDS",
    "PUBLIC_WORDS",
    "HIDDEN",
    "consent",
    "JS_RULES",
    "form_pairs",
    "REASONS",
    "normalized",
    "sensitive_kind",
]
