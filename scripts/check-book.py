#!/usr/bin/env python3
"""Check mdBook source includes and local links in its rendered HTML."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
INCLUDE = re.compile(r"\{\{#(?:rustdoc_)?include\s+([^}]+)\}\}")


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])


def check_sources(source, errors):
    summary = source / "SUMMARY.md"
    if not summary.is_file():
        errors.append(f"missing summary: {summary}")
        return
    chapters = set()
    for link in re.findall(r"\]\(([^)]+)\)", summary.read_text(encoding="utf-8")):
        target = source / unquote(urlsplit(link).path)
        if not target.is_file():
            errors.append(f"{summary}: missing chapter {link}")
        chapters.add(target.resolve())
    for path in source.rglob("*.md"):
        if path != summary and path.resolve() not in chapters:
            errors.append(f"{path}: chapter is absent from SUMMARY.md")
        for include in INCLUDE.findall(path.read_text(encoding="utf-8")):
            # This book includes whole tested files, without line-number excerpts.
            target = path.parent / include.strip()
            if not target.is_file():
                errors.append(f"{path}: missing include {include.strip()}")


def check_html(destination, errors):
    pages = {path.resolve(): Page(path) for path in destination.rglob("*.html")}
    if not pages:
        errors.append(f"no rendered HTML in {destination}; run mdbook build book first")
        return
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            if url.path.startswith("/"):
                target = (destination / unquote(url.path).lstrip("/")).resolve()
            else:
                target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(destination):
                errors.append(f"{path}: target leaves rendered book {link}")
            elif not target.is_file():
                errors.append(f"{path}: missing target {link}")
            elif url.fragment and target.suffix == ".html":
                target_page = pages.get(target)
                if target_page is None or unquote(url.fragment) not in target_page.ids:
                    errors.append(f"{path}: missing anchor {link}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--book", type=Path, default=ROOT / "book")
    parser.add_argument("--html", type=Path, default=ROOT / "target/book")
    args = parser.parse_args()
    errors = []
    check_sources(args.book.resolve() / "src", errors)
    if not args.source_only:
        check_html(args.html.resolve(), errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.source_only:
        print("Book includes and summary are valid.")
    else:
        print("Book includes and internal links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
