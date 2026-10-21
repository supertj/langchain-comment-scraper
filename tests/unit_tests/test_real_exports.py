"""Optional check against real exports, which aren't committed because they hold other
people's comments. Put JSON exports in a folder and run:

    CS_REAL_EXPORTS_DIR=/path/to/exports uv run pytest tests/unit_tests/test_real_exports.py

If a thread was also exported as Markdown and the .md file sits next to the .json with the
same name (the extension names both after the thread title), the comment numbers and reply
links are compared line by line with that Markdown file.
"""

import os
import re
from pathlib import Path

import pytest

from langchain_comment_scraper import CommentScraperLoader

FOLDER = os.environ.get("CS_REAL_EXPORTS_DIR")
pytestmark = pytest.mark.skipif(
    not FOLDER, reason="set CS_REAL_EXPORTS_DIR to a folder of real exports"
)
MARKDOWN_LINE = re.compile(
    r"^\[c(\d+)\](?: reply to \[c(\d+)\]| (reply to a comment not fetched))? \(score ", re.MULTILINE
)


def real_files() -> list[Path]:
    return sorted(Path(FOLDER).glob("*.json")) if FOLDER else []


@pytest.mark.parametrize("path", real_files(), ids=lambda path: path.name)
def test_real_export(path: Path) -> None:
    docs = CommentScraperLoader(path, include_post=False, include_deleted=True).load()
    by_thread: dict[str, list[int]] = {}
    for doc in docs:
        by_thread.setdefault(doc.metadata["thread_id"], []).append(doc.metadata["comment_number"])
    for numbers in by_thread.values():
        assert numbers == list(range(1, len(numbers) + 1))

    markdown = path.with_suffix(".md")
    if markdown.exists() and len(by_thread) == 1:
        found = [
            (int(number), int(reply) if reply else None, bool(not_fetched))
            for number, reply, not_fetched in MARKDOWN_LINE.findall(
                markdown.read_text(encoding="utf-8")
            )
        ]
        loaded = [
            (
                doc.metadata["comment_number"],
                doc.metadata["reply_to_number"],
                doc.metadata["parent_missing"],
            )
            for doc in docs
        ]
        assert found == loaded
