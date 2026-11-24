"""Check the Markdown in this repository for broken links.

Default (offline, runs on every pull request):
  - relative links and images point to files that exist
  - #anchors exist in the target page (GitHub heading slugs)
  - every image has alt text

--external (weekly): also requests every http(s) link and fails on 404/410,
which catches pages moved in other repositories. Other HTTP errors and timeouts
are reported as warnings, because they are usually temporary.

Standard library only. Usage: python .github/scripts/check_links.py [--external]
"""
import argparse
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "runs"}

TARGET_RE = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")  # ](target), also inside [![..](..)](..)
EMPTY_ALT_RE = re.compile(r"!\[\s*\]\(")
HTML_SRC_RE = re.compile(r"<(?:img|source)\b[^>]*?\b(?:src|srcset)=\"([^\"]+)\"", re.I)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def prose_lines(path):
    """(line number, text) for lines outside fenced code blocks."""
    in_code = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            yield number, line


def slugify(heading):
    """GitHub's anchor for a heading: rendered text, lower case, punctuation removed, spaces to hyphens."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*]", "", text).strip().lower()
    kept = (c for c in text if c in " -_" or unicodedata.category(c)[0] in "LN" or unicodedata.category(c) in ("Mn", "Mc"))
    return "".join(kept).replace(" ", "-")


_anchor_cache = {}


def anchors(path):
    if path not in _anchor_cache:
        seen, found = {}, set()
        for _, line in prose_lines(path):
            match = HEADING_RE.match(line)
            if match:
                slug = slugify(match.group(2))
                count = seen.get(slug, 0)
                seen[slug] = count + 1
                found.add(slug if count == 0 else f"{slug}-{count}")
            found.update(re.findall(r"<a\s+(?:id|name)=\"([^\"]+)\"", line))
        _anchor_cache[path] = found
    return _anchor_cache[path]


def check_internal():
    errors, external = [], {}
    for md in markdown_files():
        rel = md.relative_to(ROOT)
        for number, line in prose_lines(md):
            where = f"{rel}:{number}"
            if EMPTY_ALT_RE.search(line):
                errors.append((where, "image has no alt text"))
            for tag in IMG_TAG_RE.findall(line):
                if not re.search(r"\balt=\"[^\"]+\"", tag):
                    errors.append((where, "<img> has no alt text"))
            for target in TARGET_RE.findall(line) + HTML_SRC_RE.findall(line):
                if re.match(r"^(https?:)", target):
                    external.setdefault(target, where)
                    continue
                if target.startswith("mailto:"):
                    continue
                path_part, _, fragment = target.partition("#")
                dest = (md.parent / unquote(path_part)).resolve() if path_part else md
                if not dest.exists():
                    errors.append((where, f"missing file: {target}"))
                elif fragment and dest.suffix == ".md" and fragment not in anchors(dest):
                    errors.append((where, f"missing anchor: {target}"))
    return errors, external


def check_external(urls):
    errors, warnings = [], []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; repo-link-check)"}
    for url, where in sorted(urls.items()):
        status = None
        for method in ("HEAD", "GET"):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=20) as r:
                    status = r.status
                break
            except urllib.error.HTTPError as e:
                status = e.code
                if e.code not in (403, 405, 429, 501):  # some servers reject HEAD; retry those with GET
                    break
            except Exception as e:  # timeouts, DNS, TLS
                status = type(e).__name__
                break
        if status in (404, 410):
            errors.append((where, f"{status}: {url}"))
        elif status != 200 and not (isinstance(status, int) and 200 <= status < 400):
            warnings.append((where, f"{status}: {url}"))
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external", action="store_true", help="also request every http(s) link")
    args = parser.parse_args()

    errors, external = check_internal()
    warnings = []
    if args.external:
        ext_errors, warnings = check_external(external)
        errors += ext_errors
    for where, message in warnings:
        print(f"::warning file={where.split(':')[0]},line={where.split(':')[1]}::{message}")
    for where, message in errors:
        print(f"::error file={where.split(':')[0]},line={where.split(':')[1]}::{message}")
    checked = f"{len(list(markdown_files()))} Markdown files" + (f", {len(external)} external links" if args.external else "")
    print(f"{len(errors)} broken, {len(warnings)} warnings ({checked})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
