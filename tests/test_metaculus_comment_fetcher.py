import unittest
from datetime import datetime, timezone

from metaculus_comment_fetcher import (
    extract_comment_text,
    group_comments_by_post_id,
    select_comment_for_forecast_start_time,
)


class TestGroupCommentsByPostId(unittest.TestCase):
    def test_groups_only_int_post_ids(self) -> None:
        comments = [
            {"id": 1, "on_post": 10, "text": "a"},
            {"id": 2, "on_post": 20, "text": "b"},
            {"id": 3, "on_post": 10, "text": "c"},
            {"id": 4, "on_post": "not-an-int", "text": "d"},
        ]
        grouped = group_comments_by_post_id(comments)
        self.assertEqual([c["id"] for c in grouped[10]], [1, 3])
        self.assertEqual([c["id"] for c in grouped[20]], [2])
        self.assertNotIn("not-an-int", grouped)  # type: ignore[operator]


class TestSelectCommentForForecastStartTime(unittest.TestCase):
    def test_selects_closest_start_time(self) -> None:
        target = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        comments = [
            {
                "id": 1,
                "created_at": "2026-02-02T00:00:00Z",
                "included_forecast": {"start_time": "2026-02-01T00:00:09Z"},
                "text": "A",
            },
            {
                "id": 2,
                "created_at": "2026-02-01T00:00:00Z",
                "included_forecast": {"start_time": "2026-02-01T00:00:00Z"},
                "text": "B",
            },
        ]
        picked = select_comment_for_forecast_start_time(
            comments=comments, forecast_start_time=target
        )
        assert picked is not None
        self.assertEqual(picked.get("id"), 2)

    def test_falls_back_to_latest_when_no_match(self) -> None:
        target = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        comments = [
            {"id": 1, "created_at": "2026-02-01T00:00:00Z", "text": "old"},
            {"id": 2, "created_at": "2026-02-03T00:00:00Z", "text": "new"},
        ]
        picked = select_comment_for_forecast_start_time(
            comments=comments, forecast_start_time=target
        )
        assert picked is not None
        self.assertEqual(picked.get("id"), 2)
        self.assertEqual(extract_comment_text(picked), "new")

