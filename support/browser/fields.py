"""Fields melo never types into: passwords, payment cards and one-time codes.

A page can dress a secret field as an ordinary one, so the decision does not
rest on ``type=password`` alone. A text box is refused when any of these says
it holds a secret:

- its ``type`` is ``password``, or the page masks what is typed in it with CSS
  (``-webkit-text-security``);
- its ``autocomplete`` names a password (``current-password``,
  ``new-password``), a one-time code (``one-time-code``) or a card
  (``cc-number``, ``cc-csc``, any ``cc-*``);
- the words that name it (its name, id, ``aria-label``, placeholder and the
  text of its labels) say password, PIN, card number, security code or
  verification code, in English or Chinese;
- it is a numeric field of four to eight digits, the shape of a code.

The page script reads these facts into a :class:`FieldAttrs` and the decision
is made here, once when the page is read and again at the moment melo types,
on what the page says then. The words are matched on a normalized copy:
``userPassword`` and ``user_password`` both read ``user password``.
"""

from __future__ import annotations

import re

from localmelo.support.browser.driver import BrowserError, FieldAttrs, Sensitive

_PASSWORD_AUTOCOMPLETE = frozenset({"current-password", "new-password"})
_PASSWORD = re.compile(
    r"password|passwd|passphrase|passcode|pwd|\bpin\b|密码|口令|passwort"
    r"|contraseña|contrasena|mot de passe|パスワード|비밀번호"
)
_CARD = re.compile(
    r"card ?(number|num|no)\b|cardnumber|\bcc ?(num|number|no)\b|\bcvv\b|\bcvc\b"
    r"|\bcsc\b|\bcvn\b|卡号|银行卡|信用卡号"
)
_CODE = re.compile(
    r"\botp\b|one ?time|\b2fa\b|\bmfa\b|\btotp\b|verification|verify code"
    r"|security code|auth(entication)? code|sms code|confirm(ation)? code"
    r"|login code|验证码|校验码|动态码|认证码|安全码|驗證碼"
)
_CAMEL = re.compile(r"([a-z])([A-Z])")
_NOT_WORDS = re.compile(r"[\W_]+")

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
    if (
        attrs.type.lower() == "password"
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
    if attrs.inputmode.lower() == "numeric" and 4 <= attrs.maxlength <= 8:
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


__all__ = ["REASONS", "normalized", "refuse_typing", "sensitive_kind"]
