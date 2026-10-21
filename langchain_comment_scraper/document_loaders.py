"""Loads Comment Scraper JSON exports as LangChain Documents.

Comment Scraper is a browser extension that exports Reddit and Hacker News threads. This
loader reads the two JSON formats it writes ("JSON, one row per comment" and "JSON, replies
nested") from local files. It makes no network requests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

GENERATOR = "Comment Scraper"


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _is_text_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


def _is_int_or_none(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_number_or_none(value: object) -> bool:
    return value is None or _is_number(value)


# The fields this loader reads, checked the way comment-scraper-mcp checks them. A file can
# carry the generator marker and still be truncated or edited by hand; better to say so than
# to hand a vector store half a thread.
_META_CHECKS: dict[str, Callable[[object], bool]] = {
    "source": lambda value: value in ("reddit", "hn"),
    "thread_id": _is_text,
    "thread_title": _is_text,
    "permalink": _is_text,
    "fetched_at": _is_text,
}

_RECORD_CHECKS: dict[str, Callable[[object], bool]] = {
    "record_type": lambda value: value in ("post", "comment"),
    "comment_id": _is_text_or_none,
    "parent_id": _is_text_or_none,
    "depth": _is_int_or_none,
    "score": _is_number_or_none,
    "score_hidden": lambda value: isinstance(value, bool),
    "created_utc": _is_text,
    "edited_utc": _is_text_or_none,
    "is_op": lambda value: isinstance(value, bool),
    "body": _is_text,
    "permalink": _is_text,
    "comment_state": _is_text,
    "subreddit": _is_text_or_none,
}


def _missing_field(value: object, checks: dict[str, Callable[[object], bool]]) -> str | None:
    """The first field that is absent or has the wrong type, or None when all pass."""
    if not isinstance(value, dict):
        return "not an object"
    for name, check in checks.items():
        if name not in value or not check(value[name]):
            return name
    return None


def _reject_constant(name: str) -> float:
    # The extension writes JSON with JSON.stringify, which never emits NaN or Infinity.
    raise ValueError(f"not valid JSON: {name}")


def _flatten_nested(post: object, comments: object) -> list[object] | None:
    """Walks a nested export back into the extension's tree order: post, then depth first.

    A loop rather than recursion, because a long "continue this thread" chain can be
    thousands of replies deep. Returns None if a replies list is missing or not a list.
    """
    if not isinstance(comments, list):
        return None
    records: list[object] = [post]
    finished = object()
    stack = [iter(comments)]
    while stack:
        node = next(stack[-1], finished)
        if node is finished:
            stack.pop()
            continue
        if not isinstance(node, dict) or not isinstance(node.get("replies"), list):
            return None
        records.append({key: value for key, value in node.items() if key != "replies"})
        stack.append(iter(node["replies"]))
    return records


class _DamagedThread(ValueError):
    pass


def _thread_records(entry: object) -> list[dict[str, Any]]:
    """One entry of an export's "threads" list as records in tree order: the post, then the
    comments depth first. Flat exports already are; nested ones are walked back into it."""
    if not isinstance(entry, dict):
        raise _DamagedThread("not an object")
    problem = _missing_field(entry.get("meta"), _META_CHECKS)
    if problem:
        raise _DamagedThread(f"meta field {problem}")
    if isinstance(entry.get("records"), list):
        records = entry["records"]
    elif "post" in entry:
        records = _flatten_nested(entry["post"], entry.get("comments"))
        if records is None:
            raise _DamagedThread("a replies list is missing")
    else:
        raise _DamagedThread("no records")

    seen_ids: set[str] = set()
    posts = 0
    for record in records:
        problem = _missing_field(record, _RECORD_CHECKS)
        if problem:
            raise _DamagedThread(f"record field {problem}")
        if record["record_type"] == "post":
            posts += 1
            continue
        # [cN] numbers are keyed by comment id, so an empty or repeated id would give two
        # comments one number and a citation would point at the wrong one.
        comment_id = record["comment_id"]
        if not comment_id or comment_id in seen_ids:
            raise _DamagedThread(f"empty or repeated comment id {comment_id!r}")
        seen_ids.add(comment_id)
    if posts != 1:
        raise _DamagedThread(f"{posts} post records")
    return cast("list[dict[str, Any]]", records)


class CommentScraperLoader(BaseLoader):
    """Load Reddit and Hacker News threads exported with Comment Scraper.

    Each comment becomes one Document whose ``page_content`` is the comment text. The
    thread's post comes first, as its own Document, unless ``include_post=False``.

    Comments are numbered the way the extension's Markdown export numbers them, so
    ``metadata["comment_number"] == 12`` is the comment shown as ``[c12]`` there. Deleted
    and removed comments keep their number but are skipped, because they have no text,
    unless ``include_deleted=True``.

    Exports contain no usernames, so neither do the Documents. ``is_op`` tells you when a
    comment came from the person who started the thread.

    Setup:
        .. code-block:: bash

            pip install -U langchain-comment-scraper

        In the Comment Scraper extension, export a thread as "JSON, one row per comment"
        or "JSON, replies nested". CSV and Markdown exports can't be loaded.

    Instantiate:
        .. code-block:: python

            from langchain_comment_scraper import CommentScraperLoader

            loader = CommentScraperLoader("which-crm-do-small-teams-actually-stick-with.json")

    Lazy load:
        .. code-block:: python

            for doc in loader.lazy_load():
                print(doc.metadata["comment_number"], doc.page_content[:60])

    Args:
        file_path: One export file, or a list of them. A file can hold several threads
            when it comes from a batch export.
        include_post: Also yield the thread's post as a Document, with
            ``record_type == "post"``, the title on the first line and then the post text.
        include_deleted: Also yield deleted and removed comments. Their ``page_content``
            is ``"(deleted)"`` or ``"(removed)"``, as in the Markdown export.
        encoding: Text encoding of the files. A byte order mark is ignored.

    Raises:
        ValueError: when a file isn't a Comment Scraper JSON export, or a thread in it is
            damaged. The message names the file.
    """

    def __init__(
        self,
        file_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        include_post: bool = True,
        include_deleted: bool = False,
        encoding: str = "utf-8-sig",
    ) -> None:
        if isinstance(file_path, (str, os.PathLike)):
            file_path = [file_path]
        self.file_paths = [Path(path).expanduser() for path in file_path]
        self.include_post = include_post
        self.include_deleted = include_deleted
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        for path in self.file_paths:
            for meta, records in self._threads(path):
                yield from self._documents(meta, records, str(path))

    def _threads(self, path: Path) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """The threads in one file, each as (meta, records in tree order)."""
        text = path.read_text(encoding=self.encoding)
        not_an_export = (
            f"{path} is not a Comment Scraper JSON export. Export the thread as "
            '"JSON, one row per comment" or "JSON, replies nested"; '
            "CSV and Markdown exports can't be loaded."
        )
        try:
            parsed = json.loads(text, parse_constant=_reject_constant)
        except RecursionError:
            raise ValueError(
                f"{path}: the replies are nested too deeply for Python's JSON parser. "
                'Export the thread as "JSON, one row per comment" instead.'
            ) from None
        except ValueError:
            raise ValueError(not_an_export) from None
        if (
            not isinstance(parsed, dict)
            or parsed.get("generator") != GENERATOR
            or not isinstance(parsed.get("threads"), list)
        ):
            raise ValueError(not_an_export)

        threads = []
        for index, entry in enumerate(parsed["threads"], start=1):
            try:
                records = _thread_records(entry)
            except _DamagedThread as error:
                raise ValueError(f"{path}: thread {index} is damaged ({error}).") from None
            threads.append((entry["meta"], records))
        return threads

    def _documents(
        self, meta: dict[str, Any], records: list[dict[str, Any]], source: str
    ) -> Iterator[Document]:
        # Same numbering as the extension's Markdown export: comments in tree order,
        # counted from 1, deleted and removed ones included.
        numbers: dict[str, int] = {}
        for record in records:
            if record["record_type"] == "comment":
                numbers[record["comment_id"]] = len(numbers) + 1

        post = next(record for record in records if record["record_type"] == "post")
        site = meta["source"]
        thread_id = meta["thread_id"]
        shared = {
            "source": source,
            "site": site,
            "thread_id": thread_id,
            "thread_title": meta["thread_title"],
            "thread_url": meta["permalink"],
            "subreddit": post["subreddit"],
            "fetched_at": meta["fetched_at"],
        }

        for record in records:
            is_post = record["record_type"] == "post"
            if is_post and not self.include_post:
                continue
            state = record["comment_state"]
            if is_post:
                parts = [meta["thread_title"].strip()]
                if state == "normal" and record["body"].strip():
                    parts.append(record["body"])
                page_content = "\n\n".join(parts)
            elif state == "normal":
                page_content = record["body"]
            elif self.include_deleted:
                page_content = f"({state})"
            else:
                continue

            comment_id = record["comment_id"]
            parent_id = record["parent_id"]
            yield Document(
                id=f"{site}:{thread_id}" if is_post else f"{site}:{thread_id}:{comment_id}",
                page_content=page_content,
                metadata={
                    **shared,
                    "record_type": record["record_type"],
                    "comment_id": comment_id,
                    "comment_number": None if is_post else numbers[comment_id],
                    "parent_id": parent_id,
                    # None for top-level comments, and for replies whose parent wasn't fetched.
                    "reply_to_number": numbers.get(parent_id) if parent_id else None,
                    "parent_missing": bool(record.get("parent_missing", False)),
                    "depth": record["depth"],
                    "score": record["score"],
                    "score_hidden": record["score_hidden"],
                    "created_utc": record["created_utc"],
                    "edited_utc": record["edited_utc"],
                    "is_op": record["is_op"],
                    "comment_state": state,
                    "permalink": record["permalink"],
                },
            )
