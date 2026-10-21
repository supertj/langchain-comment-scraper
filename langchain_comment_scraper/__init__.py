"""Load Comment Scraper exports of Reddit and Hacker News threads as LangChain Documents."""

from importlib import metadata

from langchain_comment_scraper.document_loaders import CommentScraperLoader

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    # Running from a source checkout that was never installed.
    __version__ = ""

__all__ = ["CommentScraperLoader", "__version__"]
