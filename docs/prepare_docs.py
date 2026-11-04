"""Stage existing Markdown and images without importing DeSCORT or changing tutorials."""
from pathlib import Path
import os
import re
import shutil
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/_generated"
GITHUB = "https://github.com/wyywyy999/DeSCORT_script"
GUIDES = ["installation.md", "data_format.md", "model_architecture.md",
          "model_zoo.md", "reproducibility.md", "hosting.md", "api/index.md", "workflows/index.md"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


class ImageTag(HTMLParser):
    def handle_starttag(self, tag, attrs):
        self.attrs = dict(attrs)


def sphinx_image(match):
    """Register HTML images with Sphinx so local files are copied to HTML output."""
    tag = ImageTag()
    tag.feed(match.group(0))
    attrs = tag.attrs
    if "src" not in attrs:
        return match.group(0)
    options = []
    if attrs.get("alt"):
        options.append(f":alt: {attrs['alt']}")
    if attrs.get("width"):
        width = attrs["width"]
        options.append(f":width: {width}px" if width.isdigit() else f":width: {width}")
    return "\n```{image} " + attrs["src"] + "\n" + "\n".join(options) + "\n```\n"


def prepare():
    sources = {ROOT / "README.md", ROOT / "tutorials/README.md"}
    sources.update((ROOT / "tutorials").rglob("README.md"))
    sources.update(ROOT / "docs" / name for name in GUIDES)
    sources.update((ROOT / "result").rglob("README.md"))
    sources.update((ROOT / "environments").glob("*.md"))
    sources = {p.resolve() for p in sources if p.is_file()}
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    def rewrite(source, match):
        target = match.group(2)
        if urlsplit(target).scheme or target.startswith(("#", "//")):
            return match.group(0)
        path_part, separator, fragment = target.partition("#")
        target_path = (source.parent / unquote(path_part)).resolve()
        try:
            relative = target_path.relative_to(ROOT)
        except ValueError:
            return match.group(0)
        if target_path.is_dir() and (target_path / "README.md").resolve() in sources:
            target_path = target_path / "README.md"
            relative = target_path.relative_to(ROOT)
        if target_path in sources:
            link = os.path.relpath(target_path, source.parent)
        elif target_path.is_file() and target_path.suffix.lower() in IMAGE_SUFFIXES:
            output = DEST / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, output)
            link = os.path.relpath(target_path, source.parent)
        else:
            kind = "tree" if target_path.is_dir() else "blob"
            link = f"{GITHUB}/{kind}/main/{quote(relative.as_posix())}"
        return f"{match.group(1)}({link}{separator}{fragment})"

    # Rewrite prose links only; shell/Python code blocks remain byte-for-byte intact.
    link_pattern = re.compile(r"(!?\[[^]\n]*\])\(([^)\s]+)\)")
    for source in sorted(sources):
        text = source.read_text()
        parts = re.split(r"(^```[^\n]*\n.*?^```[^\n]*(?:\n|$))", text,
                         flags=re.MULTILINE | re.DOTALL)
        for i in range(0, len(parts), 2):
            parts[i] = link_pattern.sub(lambda match: rewrite(source, match), parts[i])
            parts[i] = re.sub(r"<img\b[^>]*>", sphinx_image, parts[i], flags=re.IGNORECASE)
        output = DEST / source.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(parts))

    primary = {ROOT / "README.md", ROOT / "tutorials/README.md"}
    primary.update(ROOT / "docs" / name for name in GUIDES)
    primary.update(p for p in sources if p.parent.parent == ROOT / "tutorials")
    extra = sorted(p.relative_to(ROOT).with_suffix("").as_posix()
                   for p in sources - primary)
    (DEST / "additional.rst").write_text(
        "Additional references\n=====================\n\n"
        ".. toctree::\n   :maxdepth: 1\n\n" + "".join(f"   {name}\n" for name in extra))
    return len(sources)


if __name__ == "__main__":
    print(f"Staged {prepare()} Markdown pages in {DEST}")
