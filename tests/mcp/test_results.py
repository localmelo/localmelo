"""Conversion of MCP tools/call results into ToolOutput."""

from __future__ import annotations

import base64

from localmelo.support.mcp import tool_result_to_output

PNG = base64.b64encode(b"\x89PNG fake").decode()


def test_text_parts_are_joined() -> None:
    out = tool_result_to_output(
        {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    )
    assert out.text == "a\nb"
    assert out.images == []
    assert not out.is_error


def test_image_part() -> None:
    out = tool_result_to_output(
        {"content": [{"type": "image", "data": PNG, "mimeType": "image/png"}]}
    )
    assert out.text == ""
    assert [(i.mime_type, i.size_bytes) for i in out.images] == [("image/png", 9)]


def test_malformed_image_becomes_note() -> None:
    out = tool_result_to_output({"content": [{"type": "image", "data": PNG}]})
    assert out.images == []
    assert out.text == "[malformed image content]"


def test_embedded_resources() -> None:
    out = tool_result_to_output(
        {
            "content": [
                {"type": "resource", "resource": {"uri": "file:///a", "text": "doc"}},
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///p",
                        "blob": PNG,
                        "mimeType": "image/png",
                    },
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///z",
                        "blob": "AAAA",
                        "mimeType": "application/zip",
                    },
                },
            ]
        }
    )
    assert out.text.splitlines() == ["doc", "[binary resource file:///z omitted]"]
    assert len(out.images) == 1


def test_links_audio_and_unknown_types() -> None:
    out = tool_result_to_output(
        {
            "content": [
                {"type": "resource_link", "name": "Inbox", "uri": "mail://inbox"},
                {"type": "audio", "data": "AAAA", "mimeType": "audio/wav"},
                {"type": "hologram"},
            ]
        }
    )
    assert out.text.splitlines() == [
        "[resource link: Inbox mail://inbox]",
        "[audio content omitted (audio/wav)]",
        "[unsupported MCP content type: 'hologram']",
    ]


def test_structured_content_used_only_without_text() -> None:
    assert tool_result_to_output(
        {"content": [], "structuredContent": {"n": 1}}
    ).text == ('{"n": 1}')
    out = tool_result_to_output(
        {"content": [{"type": "text", "text": "t"}], "structuredContent": {"n": 1}}
    )
    assert out.text == "t"


def test_is_error_flag() -> None:
    assert tool_result_to_output({"content": [], "isError": True}).is_error
    assert not tool_result_to_output({"content": [], "isError": "yes"}).is_error


def test_invalid_result() -> None:
    out = tool_result_to_output(["not", "a", "dict"])
    assert out.is_error
    assert "invalid MCP tool result" in out.text
