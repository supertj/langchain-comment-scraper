from langchain_comment_scraper import __all__

EXPECTED_ALL = ["CommentScraperLoader", "__version__"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
