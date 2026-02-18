import unittest

from source_catalog import (
    apply_patch_ops,
    dump_catalog,
    load_catalog,
    render_sources_markdown,
    suggest_sources_for_question,
)


class TestSourceCatalog(unittest.TestCase):
    def test_load_empty_catalog(self) -> None:
        catalog = load_catalog("")
        self.assertEqual(catalog["version"], 1)
        self.assertIsInstance(catalog["sources"], list)

    def test_apply_patch_add_remove_update(self) -> None:
        catalog = load_catalog("version: 1\nsources: []\n")

        catalog, summary = apply_patch_ops(
            catalog,
            {
                "add": [
                    {"url": "https://example.com/a", "title": "A"},
                    {"url": "https://example.com/b", "title": "B"},
                ]
            },
        )
        self.assertEqual(summary.added, 2)
        self.assertEqual(len(catalog["sources"]), 2)

        catalog, summary = apply_patch_ops(
            catalog,
            {"update": [{"url": "https://example.com/a", "fields": {"notes": "ok"}}]},
        )
        self.assertEqual(summary.updated, 1)

        catalog, summary = apply_patch_ops(
            catalog,
            {"remove": [{"url": "https://example.com/b"}]},
        )
        self.assertEqual(summary.removed, 1)
        self.assertEqual(len(catalog["sources"]), 1)

        dumped = dump_catalog(catalog)
        self.assertIn("https://example.com/a", dumped)

    def test_invalid_urls_are_ignored(self) -> None:
        catalog = load_catalog("version: 1\nsources: []\n")
        catalog, summary = apply_patch_ops(
            catalog,
            {"add": [{"url": "file:///etc/passwd"}, {"url": "notaurl"}]},
        )
        self.assertEqual(summary.added, 0)
        self.assertEqual(len(catalog["sources"]), 0)

    def test_suggest_sources_for_question(self) -> None:
        catalog = load_catalog(
            dump_catalog(
                {
                    "version": 1,
                    "updated_at": "1970-01-01T00:00:00+00:00",
                    "sources": [
                        {
                            "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php",
                            "title": "USGS earthquakes",
                            "tags": ["usgs", "earthquakes"],
                            "notes": "Official earthquake feeds",
                        },
                        {
                            "url": "https://fred.stlouisfed.org/",
                            "title": "FRED",
                            "tags": ["macro", "rates", "inflation"],
                            "notes": "Macro time series",
                        },
                    ],
                }
            )
        )
        picks = suggest_sources_for_question(
            catalog, query_text="Earthquake magnitude in Alaska", max_items=1
        )
        self.assertEqual(len(picks), 1)
        self.assertIn("usgs.gov", picks[0]["url"])

    def test_render_sources_markdown_bounds_output(self) -> None:
        sources = [
            {
                "url": "https://example.com/a",
                "title": "A",
                "tags": ["x", "y"],
                "notes": "n" * 200,
            },
            {"url": "https://example.com/b", "title": "B"},
        ]
        text, urls = render_sources_markdown(sources, max_chars=120)
        self.assertTrue(text)
        self.assertTrue(urls)
        self.assertLessEqual(len(text), 120 + len("\n[TRUNCATED]"))


if __name__ == "__main__":
    unittest.main()

