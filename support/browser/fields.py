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
  fieldset's legend and the text just before it) say password, PIN, card,
  expiry, security code, verification code, token or a bare "code", in
  English or Chinese (a postal, promo or country code is not one);
- it is a box for one character (a split code), or a numeric, telephone or
  number field of four to eight characters, the shape of a code.

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

from localmelo.support.browser.driver import BrowserError, FieldAttrs, Sensitive

_PASSWORD_AUTOCOMPLETE = frozenset({"current-password", "new-password"})
PASSWORD_WORDS = (
    r"password|passwd|passphrase|passcode|pwd|\bpin\b|密码|口令|passwort"
    r"|contraseña|contrasena|mot de passe|パスワード|비밀번호"
)
CARD_WORDS = (
    r"card ?(number|num|no)\b|cardnumber|credit ?card|debit ?card|\bcard\b|\bcards\b"
    r"|\bcc ?(num|number|no|exp)|\bcvv\b|\bcvc\b|\bcsc\b|\bcvn\b|\bcid\b"
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
# A bare "code" or "token" is secret unless it is plainly another sort of code.
BARE_CODE_WORDS = r"\bcode\b|\bcodes\b|\btoken\b|\btokens\b|\bkey\b"
BENIGN_CODE_WORDS = (
    r"zip|postal|post ?code|promo|coupon|voucher|discount|referral|invit|area code"
    r"|country|dial|language|colou?r|\bhex\b|source|captcha|barcode|bar code|qr"
    r"|product|item code|sku|isbn|api key|ssh key|public key|邮编|邮政|优惠|折扣|邀请"
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
    return ""


def refuse_typing(attrs: FieldAttrs | None) -> None:
    """Raise :class:`BrowserError`, in plain words, when melo must not type in the field."""
    kind = sensitive_kind(attrs)
    if kind:
        raise BrowserError(
            f"That is {REASONS[kind]}: melo never types into passwords, payment "
            "cards or one-time codes. Ask the person to fill it in themselves."
        )


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
  return "";
};
"""
)


__all__ = [
    "HIDDEN",
    "JS_RULES",
    "form_pairs",
    "REASONS",
    "normalized",
    "refuse_typing",
    "sensitive_kind",
]
