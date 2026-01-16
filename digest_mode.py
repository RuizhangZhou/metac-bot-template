import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


def extract_tournament_identifier(value: str) -> str | None:
    raw = value.strip()
    if not raw or raw.startswith("#"):
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        tournament_match = re.search(r"/tournament/([^/]+)/?", raw)
        if tournament_match:
            slug = tournament_match.group(1).strip("/")
            return slug if slug.isdigit() else slug.lower()
        index_match = re.search(r"/index/([^/]+)/?", raw)
        if index_match:
            return f"index:{index_match.group(1)}"
        return None

    slug = raw.strip("/")
    return slug if slug.isdigit() else slug.lower()


def load_tournament_identifiers(
    tournaments_file: str | None, extra_identifiers: list[str] | None
) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    unsupported: list[str] = []

    if tournaments_file:
        path = Path(tournaments_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                identifier = extract_tournament_identifier(line)
                if not identifier:
                    continue
                if identifier.startswith("index:"):
                    unsupported.append(identifier)
                else:
                    identifiers.append(identifier)
        else:
            logger.warning(f"Tournaments file not found: {tournaments_file}")

    for raw in extra_identifiers or []:
        identifier = extract_tournament_identifier(raw)
        if not identifier:
            continue
        if identifier.startswith("index:"):
            unsupported.append(identifier)
        else:
            identifiers.append(identifier)

    seen: set[str] = set()
    unique_identifiers: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_identifiers.append(identifier)

    return unique_identifiers, unsupported


def _prediction_to_compact_jsonable(prediction: object) -> object:
    if isinstance(prediction, (float, int, str, bool)) or prediction is None:
        return prediction
    model_dump = getattr(prediction, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(prediction, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    as_dict = getattr(prediction, "dict", None)
    if callable(as_dict):
        return as_dict()
    return str(prediction)


def _get_close_time_iso(question: MetaculusQuestion) -> str | None:
    close_time = getattr(question, "close_time", None)
    if close_time is None:
        return None
    if isinstance(close_time, datetime):
        return close_time.astimezone(timezone.utc).isoformat()
    return str(close_time)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_until(close_time_iso: str | None) -> float | None:
    dt = _parse_iso_datetime(close_time_iso)
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds() / 86400


def _threshold_factor_from_days_left(days_left: float | None) -> float:
    if days_left is None:
        return 1.0
    if days_left <= 2:
        return 0.5
    if days_left <= 7:
        return 0.7
    return 1.0


def _get_numeric_percentile_map(pred: dict) -> dict[float, float]:
    percentiles = pred.get("declared_percentiles")
    if not isinstance(percentiles, list):
        return {}
    out: dict[float, float] = {}
    for p in percentiles:
        if not isinstance(p, dict):
            continue
        perc = p.get("percentile")
        val = p.get("value")
        if isinstance(perc, (int, float)) and isinstance(val, (int, float)):
            out[float(perc)] = float(val)
    return out


def _approx_median_from_percentiles(pmap: dict[float, float]) -> float | None:
    if 0.5 in pmap:
        return pmap[0.5]
    p40 = pmap.get(0.4)
    p60 = pmap.get(0.6)
    if p40 is not None and p60 is not None:
        return 0.5 * (p40 + p60)
    if not pmap:
        return None
    nearest = min(pmap.keys(), key=lambda p: abs(p - 0.5))
    return pmap[nearest]


def _is_significant_change(
    *,
    old_pred: object | None,
    new_pred: object | None,
    question_type: str,
    close_time_iso: str | None,
    old_close_time_iso: str | None,
) -> tuple[bool, str]:
    days_left = _days_until(close_time_iso)
    factor = _threshold_factor_from_days_left(days_left)

    if old_pred is None and new_pred is not None:
        return True, "new_question"
    if old_pred is not None and new_pred is None:
        return False, "question_missing_now"

    if close_time_iso and old_close_time_iso and close_time_iso != old_close_time_iso:
        new_dt = _parse_iso_datetime(close_time_iso)
        old_dt = _parse_iso_datetime(old_close_time_iso)
        if new_dt and old_dt:
            delta_hours = abs((new_dt - old_dt).total_seconds()) / 3600
            if delta_hours >= 24:
                return True, f"close_time_changed_{delta_hours:.1f}h"

    if question_type == "binary":
        if not isinstance(old_pred, (int, float)) or not isinstance(
            new_pred, (int, float)
        ):
            return True, "binary_type_changed"
        old_p = float(old_pred)
        new_p = float(new_pred)
        base = 0.10
        threshold = base * factor
        abs_delta = abs(new_p - old_p)
        crossed = (old_p - 0.5) * (new_p - 0.5) < 0
        if abs_delta >= threshold:
            return True, f"abs_delta={abs_delta:.3f}"
        if crossed and abs_delta >= max(0.04, 0.5 * threshold):
            return True, f"crossed_50_abs_delta={abs_delta:.3f}"
        return False, f"abs_delta={abs_delta:.3f}"

    if question_type == "multiple_choice":
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "mc_type_changed"

        def to_map(d: dict) -> dict[str, float]:
            if "predicted_options" in d and isinstance(d["predicted_options"], list):
                m: dict[str, float] = {}
                for item in d["predicted_options"]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("option_name")
                    prob = item.get("probability")
                    if isinstance(name, str) and isinstance(prob, (int, float)):
                        m[name] = float(prob)
                return m
            return {
                k: float(v)
                for k, v in d.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }

        old_map = to_map(old_pred)
        new_map = to_map(new_pred)
        keys = sorted(set(old_map) | set(new_map))
        tvd = 0.5 * sum(
            abs(new_map.get(k, 0.0) - old_map.get(k, 0.0)) for k in keys
        )
        base = 0.15
        threshold = base * factor
        if tvd >= threshold:
            return True, f"tvd={tvd:.3f}"
        return False, f"tvd={tvd:.3f}"

    if question_type in {"numeric", "date", "discrete"}:
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "numeric_type_changed"
        old_map = _get_numeric_percentile_map(old_pred)
        new_map = _get_numeric_percentile_map(new_pred)
        old_med = _approx_median_from_percentiles(old_map)
        new_med = _approx_median_from_percentiles(new_map)
        if old_med is None or new_med is None:
            return True, "missing_median"
        old_p10 = old_map.get(0.1)
        old_p90 = old_map.get(0.9)
        new_p10 = new_map.get(0.1)
        new_p90 = new_map.get(0.9)
        old_width = (
            (old_p90 - old_p10)
            if (old_p10 is not None and old_p90 is not None)
            else None
        )
        new_width = (
            (new_p90 - new_p10)
            if (new_p10 is not None and new_p90 is not None)
            else None
        )
        width: float
        if old_width is not None and new_width is not None:
            width = max(1e-9, 0.5 * (old_width + new_width))
        elif old_width is not None:
            width = max(1e-9, old_width)
        elif new_width is not None:
            width = max(1e-9, new_width)
        else:
            width = max(1e-9, abs(old_med), abs(new_med))

        normalized_median_shift = abs(new_med - old_med) / width
        base = 0.35
        threshold = base * factor

        if question_type == "date":
            median_shift_days = abs(new_med - old_med) / 86400
            absolute_day_threshold = 30 * factor
            if median_shift_days >= absolute_day_threshold:
                return True, f"median_shift_days={median_shift_days:.1f}"

        if normalized_median_shift >= threshold:
            return True, f"norm_median_shift={normalized_median_shift:.3f}"
        return False, f"norm_median_shift={normalized_median_shift:.3f}"

    if question_type == "conditional":
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "conditional_type_changed"
        old_child = old_pred.get("child")
        new_child = new_pred.get("child")
        child_type = "unknown"
        if isinstance(new_child, (int, float)):
            child_type = "binary"
        elif isinstance(new_child, dict) and "predicted_options" in new_child:
            child_type = "multiple_choice"
        elif isinstance(new_child, dict) and "declared_percentiles" in new_child:
            child_type = "numeric"
        return _is_significant_change(
            old_pred=old_child,
            new_pred=new_child,
            question_type=child_type,
            close_time_iso=close_time_iso,
            old_close_time_iso=old_close_time_iso,
        )

    return True, f"unknown_type_{question_type}"


def matrix_send_message(message: str) -> None:
    homeserver = os.getenv("MATRIX_HOMESERVER")
    access_token = os.getenv("MATRIX_ACCESS_TOKEN")
    room_id = os.getenv("MATRIX_ROOM_ID")
    if not homeserver or not access_token or not room_id:
        return

    txn_id = uuid4().hex
    room_id_escaped = requests.utils.quote(room_id, safe="")
    url = (
        f"{homeserver.rstrip('/')}/_matrix/client/v3/rooms/"
        f"{room_id_escaped}/send/m.room.message/{txn_id}"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"msgtype": "m.text", "body": message}
    response = requests.put(url, headers=headers, json=payload, timeout=30)
    if not response.ok:
        logger.warning(f"Matrix send failed: {response.status_code} {response.text}")


async def run_digest(
    *,
    template_bot: ForecastBot,
    tournaments: list[str],
    state_path: Path,
    out_dir: Path,
) -> None:
    from forecasting_tools.data_models.data_organizer import DataOrganizer

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    previous_state: dict = {}
    if state_path.exists():
        try:
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read previous state: {e}")
            previous_state = {}

    previous_questions: dict = (
        previous_state.get("questions", {}) if isinstance(previous_state, dict) else {}
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    new_state_questions: dict[str, dict] = {}
    significant_changes: list[dict] = []
    failures: list[dict] = []

    def safe_report_attr(report: object, attr: str) -> str | None:
        try:
            return getattr(report, attr)
        except Exception:
            return None

    for tournament_id in tournaments:
        try:
            reports_or_errors = await template_bot.forecast_on_tournament(
                tournament_id, return_exceptions=True
            )
        except Exception as e:
            failures.append({"tournament": tournament_id, "error": str(e)})
            continue

        for item in reports_or_errors:
            if isinstance(item, BaseException):
                failures.append({"tournament": tournament_id, "error": repr(item)})
                continue

            report = item
            question = report.question
            key = "|".join(
                [
                    str(question.id_of_post or ""),
                    str(question.id_of_question or ""),
                    str(question.conditional_type or ""),
                    str(question.group_question_option or ""),
                ]
            )

            if isinstance(question, BinaryQuestion):
                qtype = "binary"
            elif isinstance(question, MultipleChoiceQuestion):
                qtype = "multiple_choice"
            elif isinstance(question, DateQuestion):
                qtype = "date"
            elif isinstance(question, NumericQuestion):
                qtype = "numeric"
            elif isinstance(question, ConditionalQuestion):
                qtype = "conditional"
            else:
                qtype = "unknown"

            close_time_iso = _get_close_time_iso(question)
            prediction_jsonable = _prediction_to_compact_jsonable(report.prediction)
            try:
                readable_prediction = DataOrganizer.get_readable_prediction(
                    report.prediction
                )
            except Exception as e:
                readable_prediction = f"(failed to format prediction: {e})"

            state_snapshot = {
                "generated_at": now_iso,
                "tournament": tournament_id,
                "question_text": question.question_text,
                "page_url": question.page_url,
                "id_of_post": question.id_of_post,
                "id_of_question": question.id_of_question,
                "close_time": close_time_iso,
                "question_type": qtype,
                "prediction": prediction_jsonable,
            }
            new_state_questions[key] = state_snapshot

            old_snapshot = previous_questions.get(key)
            old_pred = (
                old_snapshot.get("prediction") if isinstance(old_snapshot, dict) else None
            )
            old_close_time = (
                old_snapshot.get("close_time") if isinstance(old_snapshot, dict) else None
            )

            significant, reason = _is_significant_change(
                old_pred=old_pred,
                new_pred=prediction_jsonable,
                question_type=qtype,
                close_time_iso=close_time_iso,
                old_close_time_iso=old_close_time,
            )
            if significant:
                significant_changes.append(
                    {
                        "key": key,
                        "tournament": tournament_id,
                        "page_url": question.page_url,
                        "question_text": question.question_text,
                        "question_type": qtype,
                        "close_time": close_time_iso,
                        "reason": reason,
                        "old_prediction": old_pred,
                        "new_prediction": prediction_jsonable,
                        "readable_prediction": readable_prediction,
                        "summary": safe_report_attr(report, "summary"),
                        "research": safe_report_attr(report, "research"),
                        "forecast_rationales": safe_report_attr(
                            report, "forecast_rationales"
                        ),
                        "explanation": getattr(report, "explanation", None),
                    }
                )

    new_state = {"version": 1, "generated_at": now_iso, "questions": new_state_questions}
    state_path.write_text(
        json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    digest_path = out_dir / f"digest_{now_iso[:10]}.md"
    changes_path = out_dir / "changes.md"
    failures_path = out_dir / "failures.json"

    def fmt_pred(pred: object | None) -> str:
        if pred is None:
            return "(none)"
        try:
            return json.dumps(pred, ensure_ascii=False)
        except Exception:
            return str(pred)

    lines: list[str] = []
    lines.append(f"# Metaculus digest ({now_iso})")
    lines.append("")
    lines.append("## Tournaments")
    for tid in tournaments:
        lines.append(f"- {tid}")
    lines.append("")

    lines.append(f"## Significant changes ({len(significant_changes)})")
    if not significant_changes:
        lines.append("- (none)")
    else:
        for ch in significant_changes:
            url = ch.get("page_url") or ch.get("key")
            lines.append(f"### {ch['question_text']}")
            lines.append(f"- Tournament: {ch['tournament']}")
            lines.append(f"- URL: {url}")
            if ch.get("close_time"):
                lines.append(f"- Close time (UTC): {ch['close_time']}")
            lines.append(f"- Type: {ch['question_type']}")
            lines.append(f"- Reason: {ch['reason']}")
            lines.append(f"- Old: {fmt_pred(ch['old_prediction'])}")
            lines.append(f"- New: {fmt_pred(ch['new_prediction'])}")
            lines.append("")
            lines.append("**Readable prediction**")
            lines.append(ch.get("readable_prediction") or "")
            lines.append("")
            if ch.get("summary"):
                lines.append("**Report summary**")
                lines.append(str(ch["summary"]))
                lines.append("")
            if ch.get("research"):
                lines.append("**Report research**")
                lines.append(str(ch["research"]))
                lines.append("")
            if ch.get("forecast_rationales"):
                lines.append("**Report forecast**")
                lines.append(str(ch["forecast_rationales"]))
                lines.append("")

    digest_path.write_text("\n".join(lines), encoding="utf-8")
    changes_path.write_text("\n".join(lines), encoding="utf-8")
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if significant_changes:
        message_lines = [
            f"Metaculus digest: {len(significant_changes)} significant change(s) ({now_iso[:10]})",
        ]
        for ch in significant_changes[:10]:
            url = ch.get("page_url") or ch.get("key")
            message_lines.append(
                f"- {ch['tournament']}: {ch['question_text']} ({url})"
            )
        if len(significant_changes) > 10:
            message_lines.append(f"... and {len(significant_changes) - 10} more")
        matrix_send_message("\n".join(message_lines))
