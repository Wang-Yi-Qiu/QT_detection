from collections import Counter
from datetime import datetime


def summarize_by_class(records: list[dict]) -> dict[str, int]:
    return dict(Counter(record.get("class_name", "unknown") for record in records))


def summarize_by_hour(records: list[dict]) -> dict[str, int]:
    hour_counter = Counter()
    for record in records:
        stamp = record.get("time", "")
        try:
            dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            key = dt.strftime("%Y-%m-%d %H:00")
        except ValueError:
            key = "unknown"
        hour_counter[key] += 1
    return dict(sorted(hour_counter.items(), key=lambda item: item[0]))
