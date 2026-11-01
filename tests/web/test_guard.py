"""Which addresses web_fetch reaches: every refused form, and the public ones that pass.

No test here touches the network: the address checks are pure, and the
resolver is always a fake.
"""

from __future__ import annotations

import ipaddress

import pytest

from localmelo.support.web.guard import (
    MAX_URL_CHARS,
    AddressError,
    address_problem,
    check_url,
    checked_addresses,
)

PUBLIC = [
    ("https://example.com/", "https://example.com/", "example.com", 443),
    ("HTTP://Example.COM./a?b=1#frag", "http://example.com/a?b=1", "example.com", 80),
    ("https://example.com:8443/x", "https://example.com:8443/x", "example.com", 8443),
    ("http://example.com:80/", "http://example.com/", "example.com", 80),
    ("http://8.8.8.8/", "http://8.8.8.8/", "8.8.8.8", 80),
    (
        "https://[2606:4700:4700::1111]/",
        "https://[2606:4700:4700::1111]/",
        "2606:4700:4700::1111",
        443,
    ),
    (
        "https://bücher.example/",
        "https://xn--bcher-kva.example/",
        "xn--bcher-kva.example",
        443,
    ),
    (
        "https://sub_domain.example.org/",
        "https://sub_domain.example.org/",
        "sub_domain.example.org",
        443,
    ),
]


@pytest.mark.parametrize(("url", "normal", "host", "port"), PUBLIC)
def test_a_public_address_passes_in_a_normal_form(
    url: str, normal: str, host: str, port: int
) -> None:
    target = check_url(url)
    assert (target.url, target.host, target.port) == (normal, host, port)


def test_the_host_header_names_the_host_and_a_port_that_is_not_the_schemes() -> None:
    assert check_url("https://example.com/").host_header == "example.com"
    assert check_url("http://example.com:8080/").host_header == "example.com:8080"
    assert check_url("http://[2606:4700::1]:81/").host_header == "[2606:4700::1]:81"


REFUSED = {
    # what the address is: (address, words the refusal says)
    "empty": ("", "http or https"),
    "spaces only": ("   ", "http or https"),
    "not a string": (None, "http or https"),
    "file": ("file:///etc/passwd", "http and https"),
    "ftp": ("ftp://example.com/", "not ftp:"),
    "javascript": ("javascript:alert(1)", "http and https"),
    "data": ("data:text/html,hi", "http and https"),
    "no scheme": ("example.com/page", "http and https"),
    "no host": ("http:///path", "no host"),
    "scheme only": ("http:example.com", "no host"),
    "user and password": ("http://user:pw@example.com/", "user name or password"),
    "user only": ("https://user@example.com/", "user name or password"),
    "userinfo trick": ("http://example.com@127.0.0.1/", "user name or password"),
    "backslash trick": ("http://example.com\\@127.0.0.1/", "backslash"),
    "tab inside": ("http://exa\tmple.com/", "control characters"),
    "newline inside": ("http://example.com/\nHost: x", "control characters"),
    "space inside": ("http://example.com/a b", "spaces"),
    "too long": ("https://example.com/" + "a" * MAX_URL_CHARS, "longer than"),
    "bad port": ("http://example.com:99999/", "invalid port"),
    "port zero": ("http://example.com:0/", "invalid port"),
    "port letters": ("http://example.com:http/", "invalid port"),
    "port non-ascii digits": ("http://example.com:٨٠/", "invalid port"),
    "zone id": ("http://[fe80::1%25en0]/", "%"),
    "percent host": ("http://127%2e0%2e0%2e1/", "%"),
    # The URL parser itself refuses these.
    "unclosed bracket": ("http://[::1/", "not a valid address"),
    "bad ipv6": ("http://[::zz]/", "not a valid address"),
    "localhost": ("http://localhost/", "this computer or the local network"),
    "localhost with port": (
        "http://localhost:8401/api",
        "this computer or the local network",
    ),
    "localhost upper and dot": (
        "http://LOCALHOST./",
        "this computer or the local network",
    ),
    "sub.localhost": ("http://app.localhost/", "this computer or the local network"),
    "dot local": ("http://printer.local/", "this computer or the local network"),
    "localdomain": (
        "http://localhost.localdomain/",
        "this computer or the local network",
    ),
    "internal": (
        "http://metadata.google.internal/",
        "this computer or the local network",
    ),
    "home.arpa": ("http://nas.home.arpa/", "this computer or the local network"),
    "lan": ("http://router.lan/", "this computer or the local network"),
    "single label": ("http://router/", "without a dot"),
    "single label with port": ("http://nas:5000/", "without a dot"),
    "loopback": ("http://127.0.0.1/", "loopback"),
    "loopback other": ("http://127.8.9.10:8500/", "loopback"),
    "zero address": ("http://0.0.0.0/", "unspecified"),
    "private 10": ("http://10.1.2.3/", "private network"),
    "private 172": ("http://172.16.0.1/", "private network"),
    "private 172 top": ("http://172.31.255.255/", "private network"),
    "private 192": ("http://192.168.1.1/", "private network"),
    "carrier nat": ("http://100.64.0.1/", "shared carrier"),
    "cloud metadata": ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    "link local": ("http://169.254.1.1/", "link-local"),
    "multicast": ("http://224.0.0.251/", "multicast"),
    "reserved": ("http://240.0.0.1/", "reserved"),
    "broadcast": ("http://255.255.255.255/", "reserved"),
    "documentation": ("http://192.0.2.10/", "documentation"),
    "benchmarking": ("http://198.18.0.1/", "benchmarking"),
    "decimal": ("http://2130706433/", "unusual form"),
    "octal": ("http://0177.0.0.1/", "unusual form"),
    "octal parts": ("http://017700000001/", "unusual form"),
    "hex": ("http://0x7f000001/", "unusual form"),
    "hex parts": ("http://0x7f.0x0.0x0.0x1/", "unusual form"),
    "short form": ("http://127.1/", "unusual form"),
    "three parts": ("http://10.0.1/", "unusual form"),
    "leading zero": ("http://010.0.0.1/", "unusual form"),
    "numeric last label": ("http://example.123/", "unusual form"),
    "bare 0x": ("http://foo.0x/", "unusual form"),
    "full-width digits": ("http://１２７．０．０．１/", "loopback"),
    "ideographic dots": ("http://127。0。0。1/", "loopback"),
    "ipv6 loopback": ("http://[::1]/", "loopback"),
    "ipv6 unspecified": ("http://[::]/", "unspecified"),
    "ipv4 mapped": ("http://[::ffff:127.0.0.1]/", "IPv4 address written as IPv6"),
    "ipv4 mapped hex": ("http://[::ffff:7f00:1]/", "IPv4 address written as IPv6"),
    "ipv4 mapped public": ("http://[::ffff:8.8.8.8]/", "IPv4 address written as IPv6"),
    "ipv4 compatible": ("http://[::127.0.0.1]/", "IPv4 address written as IPv6"),
    "nat64": ("http://[64:ff9b::a00:1]/", "IPv4 address written as IPv6"),
    "6to4": ("http://[2002:7f00:1::]/", "IPv4 address written as IPv6"),
    "teredo": ("http://[2001:0:4136:e378::1]/", "special-purpose"),
    "ula": ("http://[fd00::1]/", "unique-local"),
    "ula fc": ("http://[fc00::1]/", "unique-local"),
    "aws metadata v6": ("http://[fd00:ec2::254]/", "unique-local"),
    "ipv6 link local": ("http://[fe80::1]/", "link-local"),
    "ipv6 site local": ("http://[fec0::1]/", "site-local"),
    "ipv6 multicast": ("http://[ff02::1]/", "multicast"),
    "ipv6 documentation": ("http://[2001:db8::1]/", "documentation"),
    "ipv6 not unicast": ("http://[4000::1]/", "not a public address"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_every_local_or_odd_address_is_refused_with_why(name: str) -> None:
    url, words = REFUSED[name]
    with pytest.raises(AddressError) as caught:
        check_url(url)  # type: ignore[arg-type]
    assert words in str(caught.value), str(caught.value)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.20.0.1",
        "192.168.0.10",
        "100.100.100.200",  # a cloud's metadata address in carrier space
        "169.254.169.254",
        "192.0.0.192",  # another cloud's metadata address
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "::ffff:10.0.0.1",
        "fd12:3456::1",
        "fe80::1",
        "fe80::1%en0",
        "ff02::1",
    ],
)
def test_no_local_address_is_public(address: str) -> None:
    assert address_problem(ipaddress.ip_address(address)) is not None


@pytest.mark.parametrize(
    "address",
    ["8.8.8.8", "93.184.216.34", "1.1.1.1", "2606:4700::1111", "2a00:1450::1"],
)
def test_public_addresses_pass(address: str) -> None:
    assert address_problem(ipaddress.ip_address(address)) is None


def _resolver(answers: dict[str, list[tuple[str, int]]]):  # type: ignore[no-untyped-def]
    async def resolve(host: str, port: int) -> list[tuple[str, int]]:
        if host not in answers:
            raise OSError("not found")
        return answers[host]

    return resolve


async def test_a_name_that_resolves_to_a_private_address_is_refused() -> None:
    resolve = _resolver({"sneaky.example": [("10.0.0.5", 80)]})
    with pytest.raises(
        AddressError, match=r"sneaky.example: it resolves to 10.0.0.5, a private"
    ):
        await checked_addresses(check_url("http://sneaky.example/"), resolve)


async def test_one_private_answer_among_public_ones_refuses_them_all() -> None:
    resolve = _resolver({"mixed.example": [("93.184.216.34", 443), ("127.0.0.1", 443)]})
    with pytest.raises(AddressError, match="loopback"):
        await checked_addresses(check_url("https://mixed.example/"), resolve)


@pytest.mark.parametrize(
    "answer",
    ["::1", "::ffff:192.168.1.1", "169.254.169.254", "fd00::1", "0.0.0.0", "not-an-ip"],
)
async def test_every_private_dns_answer_is_refused(answer: str) -> None:
    resolve = _resolver({"rebind.example": [(answer, 80)]})
    with pytest.raises(AddressError):
        await checked_addresses(check_url("http://rebind.example/"), resolve)


async def test_public_answers_are_kept_once_each_in_order() -> None:
    resolve = _resolver(
        {
            "ok.example": [
                ("93.184.216.34", 80),
                ("93.184.216.34", 80),
                ("2606:2800::1", 80),
            ]
        }
    )
    found = await checked_addresses(check_url("http://ok.example/"), resolve)
    assert [str(ip) for ip, _ in found] == ["93.184.216.34", "2606:2800::1"]


async def test_a_name_with_no_answer_is_not_found() -> None:
    with pytest.raises(AddressError, match="was not found"):
        await checked_addresses(check_url("http://gone.example/"), _resolver({}))


async def test_an_allow_hook_lets_only_the_address_it_names_through() -> None:
    resolve = _resolver(
        {"pages.test": [("127.0.0.1", 9)], "other.test": [("127.0.0.1", 9)]}
    )

    def allow(host: str, ip: object) -> bool:
        return host == "pages.test"

    found = await checked_addresses(
        check_url("http://pages.test/"), resolve, allow=allow
    )
    assert [str(ip) for ip, _ in found] == ["127.0.0.1"]
    with pytest.raises(AddressError, match="loopback"):
        await checked_addresses(check_url("http://other.test/"), resolve, allow=allow)
    # A name the URL refuses never reaches the hook or the resolver.
    with pytest.raises(AddressError):
        check_url("http://localhost/")


async def test_a_slow_resolver_is_given_up_on() -> None:
    import asyncio

    async def slow(host: str, port: int) -> list[tuple[str, int]]:
        await asyncio.sleep(5)
        return [("93.184.216.34", port)]

    with pytest.raises(AddressError, match="took too long"):
        await checked_addresses(check_url("http://slow.example/"), slow, timeout_s=0.05)
