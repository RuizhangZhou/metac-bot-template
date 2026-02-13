import asyncio
import unittest

from local_web_crawl import LocalCrawlLimits, PlaywrightWebPageParser


class TestLocalCrawlUrlSafety(unittest.TestCase):
    def test_blocks_private_and_local_hosts_by_default(self) -> None:
        limits = LocalCrawlLimits(resolve_dns=False)
        parser = PlaywrightWebPageParser(limits=limits)

        self.assertFalse(asyncio.run(parser.is_url_safe_for_crawl("http://127.0.0.1/")))
        self.assertFalse(asyncio.run(parser.is_url_safe_for_crawl("http://localhost/")))
        self.assertFalse(asyncio.run(parser.is_url_safe_for_crawl("http://10.0.0.1/")))
        self.assertFalse(
            asyncio.run(parser.is_url_safe_for_crawl("http://169.254.169.254/"))
        )

    def test_allows_private_hosts_when_enabled(self) -> None:
        limits = LocalCrawlLimits(resolve_dns=False, allow_private_hosts=True)
        parser = PlaywrightWebPageParser(limits=limits)
        self.assertTrue(asyncio.run(parser.is_url_safe_for_crawl("http://127.0.0.1/")))
        self.assertTrue(asyncio.run(parser.is_url_safe_for_crawl("http://localhost/")))

    def test_blocks_non_http_schemes(self) -> None:
        limits = LocalCrawlLimits(resolve_dns=False)
        parser = PlaywrightWebPageParser(limits=limits)
        self.assertFalse(asyncio.run(parser.is_url_safe_for_crawl("file:///etc/passwd")))
        self.assertFalse(asyncio.run(parser.is_url_safe_for_crawl("javascript:alert(1)")))

