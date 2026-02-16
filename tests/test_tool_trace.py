import json
import unittest

from tool_trace import (
    ensure_tool_trace_base,
    extract_tool_trace_json,
    extract_urls,
    record_urls,
    render_tool_trace_markdown,
)


class TestToolTrace(unittest.TestCase):
    def test_extract_urls_strips_punctuation_and_dedupes(self) -> None:
        text = "See https://example.com/foo), https://example.com/foo. and https://example.com/bar!"
        self.assertEqual(
            extract_urls(text),
            ["https://example.com/foo", "https://example.com/bar"],
        )

    def test_record_urls_appends_unique_with_limit(self) -> None:
        trace = ensure_tool_trace_base({})
        record_urls(
            trace,
            bucket="web_search_urls",
            urls=["https://a.com", "https://b.com", "https://a.com", "https://c.com"],
            max_urls=2,
        )
        self.assertEqual(trace["web_search_urls"], ["https://a.com", "https://b.com"])

    def test_render_and_extract_roundtrip(self) -> None:
        trace = ensure_tool_trace_base({"local_crawl_urls": ["https://x.com"]})
        md = render_tool_trace_markdown(trace, max_chars=10_000)
        parsed = extract_tool_trace_json(md)
        self.assertIsInstance(parsed, dict)
        assert parsed is not None
        self.assertEqual(parsed["local_crawl_urls"], ["https://x.com"])
        # Ensure JSON is valid.
        json.dumps(parsed)


if __name__ == "__main__":
    unittest.main()

