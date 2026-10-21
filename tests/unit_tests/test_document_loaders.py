import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_comment_scraper import CommentScraperLoader

FIXTURES = Path(__file__).parent.parent / "fixtures"
EXPORT_FILES = [
    "which-crm.json",
    "which-crm-nested.json",
    "pricing-nested.json",
    "hacker-news-two-threads.json",
]

ENVELOPE = {
    "schema_version": 1,
    "exported_at": "2026-10-10T12:00:00.000Z",
    "generator": "Comment Scraper",
}
META = {
    "source": "reddit",
    "thread_id": "t1",
    "thread_title": "Thread t1",
    "permalink": "https://www.reddit.com/r/test/comments/t1/",
    "fetch_path": "json",
    "task_state": "all_known_fetched",
    "stop_reason": None,
    "fetched": 1,
    "known_unfetched_at_least": 0,
    "reported_by_source": 1,
    "fetched_at": "2026-10-10T12:00:00.000Z",
}


def record(
    comment_id: str | None, parent_id: str | None, body: str = "text", **extra: Any
) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "source": "reddit",
        "record_type": "post" if comment_id is None else "comment",
        "thread_id": "t1",
        "thread_title": "Thread t1",
        "subreddit": "test",
        "comment_id": comment_id,
        "parent_id": parent_id,
        "depth": None if comment_id is None else 0,
        "score": 1,
        "score_hidden": False,
        "created_utc": "2026-10-09T12:00:00.000Z",
        "edited_utc": None,
        "is_op": False,
        "body": body,
        "permalink": f"https://www.reddit.com/r/test/comments/t1/_/{comment_id}/",
        "comment_state": "normal",
    }
    base.update(extra)
    return base


def write_export(tmp_path: Path, *threads: Any, name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({**ENVELOPE, "threads": list(threads)}))
    return path


def comments(docs: list[Document]) -> list[Document]:
    return [doc for doc in docs if doc.metadata["record_type"] == "comment"]


def test_flat_export_gives_the_post_then_one_document_per_comment() -> None:
    docs = CommentScraperLoader(FIXTURES / "which-crm.json").load()

    post = docs[0]
    assert post.id == "reddit:crm1"
    assert (
        post.page_content
        == "Which CRM do small teams actually stick with?\n\n# Which CRM?\nWe are four people."
    )
    assert post.metadata["record_type"] == "post"
    assert post.metadata["comment_number"] is None

    # a8 (removed) and a5 (deleted) have no text, so they are left out but keep their numbers.
    assert [doc.metadata["comment_id"] for doc in comments(docs)] == [
        "a1",
        "a2",
        "a3",
        "a4",
        "a6",
        "a7",
    ]
    assert [doc.metadata["comment_number"] for doc in comments(docs)] == [1, 2, 3, 5, 7, 8]

    reply = next(doc for doc in docs if doc.metadata["comment_id"] == "a3")
    assert reply.id == "reddit:crm1:a3"
    assert reply.page_content == "The export button was the first thing they hid."
    assert reply.metadata == {
        "source": str(FIXTURES / "which-crm.json"),
        "site": "reddit",
        "thread_id": "crm1",
        "thread_title": "Which CRM do small teams actually stick with?",
        "thread_url": "https://www.reddit.com/r/smallbusiness/comments/crm1/",
        "subreddit": "smallbusiness",
        "fetched_at": "2026-10-10T12:00:00.000Z",
        "record_type": "comment",
        "comment_id": "a3",
        "comment_number": 3,
        "parent_id": "a2",
        "reply_to_number": 2,
        "parent_missing": False,
        "depth": 2,
        "score": 31,
        "score_hidden": False,
        "created_utc": "2026-10-09T12:00:00.000Z",
        "edited_utc": "2026-10-09T16:00:00.000Z",
        "is_op": False,
        "comment_state": "normal",
        "permalink": "https://www.reddit.com/r/smallbusiness/comments/crm1/_/a3/",
    }


def test_edge_cases_keep_what_the_export_says() -> None:
    by_id = {
        doc.metadata["comment_id"]: doc
        for doc in CommentScraperLoader(FIXTURES / "which-crm.json").load()
    }
    assert by_id["a2"].metadata["is_op"] is True
    assert by_id["a4"].metadata["score"] is None
    assert by_id["a4"].metadata["score_hidden"] is True
    # A reply to a deleted comment still points at the deleted comment's number.
    assert by_id["a6"].metadata["reply_to_number"] == 6
    # Comment text is kept as written, including lines Markdown would read as headings.
    assert (
        by_id["a6"].page_content
        == "# Not a heading\nWe ended up on a spreadsheet.\n\nSecond paragraph."
    )
    orphan = by_id["a7"]
    assert orphan.metadata["parent_missing"] is True
    assert orphan.metadata["parent_id"] == "gone1"
    assert orphan.metadata["reply_to_number"] is None
    top_level = by_id["a1"]
    assert top_level.metadata["parent_id"] == "crm1"
    assert top_level.metadata["reply_to_number"] is None
    assert top_level.metadata["depth"] == 0


def markdown_line_parts(doc: Document) -> tuple[str, str]:
    """What the Markdown export prints before and after the relative age for this comment."""
    meta = doc.metadata
    if meta["reply_to_number"]:
        reply = f" reply to [c{meta['reply_to_number']}]"
    elif meta["parent_missing"]:
        reply = " reply to a comment not fetched"
    else:
        reply = ""
    score = "hidden" if meta["score"] is None else meta["score"]
    flags = (", OP" if meta["is_op"] else "") + (", edited" if meta["edited_utc"] else "")
    return f"[c{meta['comment_number']}]{reply} (score {score}, ", f"{flags})"


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_comment_numbers_match_the_extensions_markdown_export(name: str) -> None:
    golden = json.loads((FIXTURES / "markdown-numbers.json").read_text())[name]
    docs = CommentScraperLoader(FIXTURES / name, include_post=False, include_deleted=True).load()
    expected = [
        (thread["thread_id"], comment) for thread in golden for comment in thread["comments"]
    ]
    assert len(docs) == len(expected)
    for doc, (thread_id, comment) in zip(docs, expected):
        assert doc.metadata["thread_id"] == thread_id
        assert doc.metadata["comment_id"] == comment["comment_id"]
        assert doc.metadata["comment_number"] == comment["number"]
        start, end = markdown_line_parts(doc)
        assert comment["markdown_line"].startswith(start)
        assert comment["markdown_line"].endswith(end)


def test_flat_and_nested_exports_of_a_thread_give_the_same_documents() -> None:
    flat = CommentScraperLoader(FIXTURES / "which-crm.json", include_deleted=True).load()
    nested = CommentScraperLoader(FIXTURES / "which-crm-nested.json", include_deleted=True).load()
    # Only the file path differs.
    for doc in flat + nested:
        doc.metadata["source"] = None
    assert flat == nested


def test_options_for_the_post_and_deleted_comments() -> None:
    docs = CommentScraperLoader(
        FIXTURES / "which-crm.json", include_post=False, include_deleted=True
    ).load()
    assert all(doc.metadata["record_type"] == "comment" for doc in docs)
    by_id = {doc.metadata["comment_id"]: doc for doc in docs}
    assert by_id["a5"].page_content == "(deleted)"
    assert by_id["a8"].page_content == "(removed)"
    assert by_id["a8"].metadata["comment_state"] == "removed"


def test_hacker_news_batch_export_with_two_threads() -> None:
    docs = CommentScraperLoader(FIXTURES / "hacker-news-two-threads.json").load()
    assert [doc.id for doc in docs] == ["hn:40001", "hn:40001:40011", "hn:40001:40012", "hn:40002"]
    assert all(doc.metadata["site"] == "hn" and doc.metadata["subreddit"] is None for doc in docs)
    assert docs[1].metadata["score"] is None
    assert docs[2].metadata["thread_url"] == "https://news.ycombinator.com/item?id=40001"
    # A thread with no comments still gives its post; a link post with no text gives just the title.
    assert docs[3].page_content == "Show HN: A tiny invoice generator"


def test_several_files_load_in_the_order_given() -> None:
    loader = CommentScraperLoader(
        [FIXTURES / "pricing-nested.json", str(FIXTURES / "which-crm.json")]
    )
    docs = loader.load()
    assert [doc.metadata["thread_id"] for doc in docs][:3] == ["price1", "price1", "price1"]
    assert docs[3].id == "reddit:crm1"
    assert docs[0].page_content == "Raising prices as a freelancer"


def test_home_directory_in_the_path_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_export(tmp_path, {"meta": META, "records": [record(None, None), record("c1", "t1")]})
    assert len(CommentScraperLoader("~/export.json").load()) == 2


def test_metadata_values_are_plain_values_vector_stores_accept() -> None:
    for name in EXPORT_FILES:
        for doc in CommentScraperLoader(FIXTURES / name, include_deleted=True).load():
            assert all(
                value is None or isinstance(value, (str, int, float, bool))
                for value in doc.metadata.values()
            )


def test_document_ids_are_unique() -> None:
    docs = CommentScraperLoader(
        [FIXTURES / "which-crm.json", FIXTURES / "hacker-news-two-threads.json"],
        include_deleted=True,
    ).load()
    ids = [doc.id for doc in docs]
    assert len(ids) == len(set(ids))


def test_loader_follows_the_base_loader_contract() -> None:
    # langchain-tests has no standard suite for document loaders, so this checks the
    # BaseLoader contract directly: every entry point returns the same Documents.
    loader = CommentScraperLoader(FIXTURES / "which-crm.json")
    lazy = list(loader.lazy_load())
    assert all(isinstance(doc, Document) for doc in lazy)
    assert loader.load() == lazy
    assert asyncio.run(loader.aload()) == lazy

    async def collect() -> list[Document]:
        return [doc async for doc in loader.alazy_load()]

    assert asyncio.run(collect()) == lazy


def test_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_text(
        "﻿" + json.dumps({**ENVELOPE, "threads": [{"meta": META, "records": [record(None, None)]}]}),
        encoding="utf-8",
    )
    assert len(CommentScraperLoader(path).load()) == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "not-an-export"}',
        '{"schema_version": 1, "generator": "Comment Scraper", "threads": [',
        "comment_id,body\n1,hello\n",
        '{"schema_version": 1, "generator": "Comment Scraper", "threads": {}}',
        '{"schema_version": 1, "generator": "Comment Scraper", "threads": [], "x": NaN}',
    ],
)
def test_files_that_are_not_exports_are_refused(tmp_path: Path, content: str) -> None:
    path = tmp_path / "file.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="is not a Comment Scraper JSON export"):
        CommentScraperLoader(path).load()


@pytest.mark.parametrize(
    ("thread", "reason"),
    [
        (7, "not an object"),
        (
            {"meta": {**META, "source": "twitter"}, "records": [record(None, None)]},
            "meta field source",
        ),
        ({"meta": META}, "no records"),
        (
            {"meta": META, "records": [record(None, None), {"record_type": "comment"}]},
            "record field comment_id",
        ),
        (
            {"meta": META, "records": [record(None, None), record("c1", "t1", score="12")]},
            "record field score",
        ),
        ({"meta": META, "records": [record("c1", "t1")]}, "0 post records"),
        (
            {"meta": META, "records": [record(None, None), record("a", "t1"), record("a", "t1")]},
            "repeated comment id 'a'",
        ),
        (
            {"meta": META, "records": [record(None, None), record("", "t1")]},
            "empty or repeated comment id ''",
        ),
        (
            {
                "meta": META,
                "post": record(None, None),
                "comments": [{**record("c1", "t1"), "replies": None}],
            },
            "replies list is missing",
        ),
        (
            {
                "meta": META,
                "post": record(None, None),
                "comments": [{**record("c1", "t1"), "replies": [None]}],
            },
            "replies list is missing",
        ),
    ],
)
def test_damaged_threads_are_refused_with_the_file_and_reason(
    tmp_path: Path, thread: Any, reason: str
) -> None:
    path = write_export(tmp_path, {"meta": META, "records": [record(None, None)]}, thread)
    with pytest.raises(ValueError, match="thread 2 is damaged") as error:
        CommentScraperLoader(path).load()
    assert reason in str(error.value)
    assert str(path) in str(error.value)


def test_long_reply_chains_in_flat_exports_are_fine(tmp_path: Path) -> None:
    chain = [record(None, None)] + [
        record(f"c{index}", f"c{index - 1}" if index else "t1") for index in range(5_000)
    ]
    path = write_export(tmp_path, {"meta": META, "records": chain})
    docs = CommentScraperLoader(path, include_post=False).load()
    assert docs[-1].metadata["comment_number"] == 5_000
    assert docs[-1].metadata["reply_to_number"] == 4_999


def nested_chain(depth: int) -> str:
    """A nested export whose replies form one chain.

    Built as text because json.dumps recurses too."""
    opening = "".join(
        json.dumps(record(f"c{index}", f"c{index - 1}" if index else "t1"))[:-1] + ', "replies": ['
        for index in range(depth)
    )
    thread = (
        '{"meta": '
        + json.dumps(META)
        + ', "post": '
        + json.dumps(record(None, None))
        + ', "comments": ['
        + opening
        + "]}" * depth
        + "]}"
    )
    return json.dumps(ENVELOPE)[:-1] + ', "threads": [' + thread + "]}"


def test_deeply_nested_exports_load_or_point_to_the_flat_format(tmp_path: Path) -> None:
    shallow = tmp_path / "shallow.json"
    shallow.write_text(nested_chain(300))
    docs = CommentScraperLoader(shallow, include_post=False).load()
    assert docs[-1].metadata["comment_number"] == 300

    # How deep Python's JSON parser can go depends on the Python version and the stack size,
    # so a very deep file either loads in full or fails with a message that says what to do.
    deep = tmp_path / "deep.json"
    deep.write_text(nested_chain(20_000))
    try:
        docs = CommentScraperLoader(deep, include_post=False).load()
    except ValueError as error:
        assert "one row per comment" in str(error)
    else:
        assert docs[-1].metadata["comment_number"] == 20_000


def test_parser_recursion_limit_is_reported_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_deep(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError

    monkeypatch.setattr("langchain_comment_scraper.document_loaders.json.loads", too_deep)
    path = write_export(tmp_path, {"meta": META, "records": [record(None, None)]})
    with pytest.raises(ValueError, match="one row per comment"):
        CommentScraperLoader(path).load()
