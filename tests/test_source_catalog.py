import unittest

from source_catalog import apply_patch_ops, dump_catalog, load_catalog


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


if __name__ == "__main__":
    unittest.main()

