"""Check local README links and the repository-hosted visual assets."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\]\(([^)]+)\)|\bsrc="([^"]+)"')
IMAGE = re.compile(r'<img\b[^>]*\bsrc="([^"]+)"')
ANIMATIONS = (
    "assets/animations/neurovision-header.gif",
    "assets/animations/neurovision-footer.gif",
)


def headings(markdown: str) -> set[str]:
    titles = re.findall(r"^#{1,6}\s+(.+)$", markdown, re.MULTILINE)
    return {
        re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-") for title in titles
    }


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    errors = []
    lines = readme.strip().splitlines()
    if not lines or (IMAGE.findall(lines[0]), IMAGE.findall(lines[-1])) != (
        [ANIMATIONS[0]],
        [ANIMATIONS[1]],
    ):
        errors.append("The header and footer GIFs must open and close the README.")
    for markdown, source in LINK.findall(readme):
        url = urlsplit(markdown or source)
        if url.scheme or url.netloc:
            continue
        path = ROOT / unquote(url.path or "README.md")
        if not path.is_file():
            errors.append(f"Missing README target: {url.path}")
        elif url.fragment and path.suffix.lower() == ".md":
            if unquote(url.fragment) not in headings(path.read_text(encoding="utf-8")):
                errors.append(f"Missing README anchor: {url.path}#{url.fragment}")
    for relative in ANIMATIONS:
        path = ROOT / relative
        if path.is_file():
            with path.open("rb") as stream:
                header = stream.read(6)
            if header != b"GIF89a" or path.stat().st_size > 128 * 1024:
                errors.append(f"Expected a compact GIF89a banner: {relative}")
    for folder in ("animations", "branding", "diagrams"):
        for path in (ROOT / "assets" / folder).glob("*.svg"):
            try:
                document = ElementTree.parse(path)
                if document.getroot().tag != "{http://www.w3.org/2000/svg}svg":
                    errors.append(f"Expected SVG root: {path.relative_to(ROOT)}")
            except ElementTree.ParseError as error:
                errors.append(f"Invalid SVG: {path.relative_to(ROOT)} ({error})")
    if errors:
        raise SystemExit("\n".join(errors))
    print("README links, animation placement and SVG assets are valid.")


if __name__ == "__main__":
    main()
