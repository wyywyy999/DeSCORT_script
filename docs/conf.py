"""Build the documentation from the maintained repository Markdown files."""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_docs import prepare

prepare()
project = "DeSCORT"
author = "DeSCORT contributors"
release = re.search(r'__version__ = "([^"]+)"',
                    (HERE.parent / "src/descort/__init__.py").read_text()).group(1)
extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
language = "en"
# Only index.rst and the staged Markdown tree are Sphinx inputs.
exclude_patterns = ["_build", "__pycache__", "Thumbs.db", ".DS_Store"]
exclude_patterns += [str(p.relative_to(HERE)) for p in HERE.rglob("*.md")
                     if "_generated" not in p.relative_to(HERE).parts
                     and "_build" not in p.relative_to(HERE).parts]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 6
html_theme = "sphinx_rtd_theme"
html_title = "DeSCORT Documentation"
html_theme_options = {"navigation_depth": 4, "collapse_navigation": False}
html_show_sourcelink = False
html_context = {"display_github": True, "github_user": "wyywyy999",
                "github_repo": "DeSCORT_script", "github_version": "main"}


def version_document_links(app, exception):
    """Version local HTML links so navigation refreshes previously cached pages."""
    if exception is not None or app.builder.format != "html":
        return
    import hashlib
    from html import escape, unescape
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    digest = hashlib.sha256()
    for source in sorted([HERE / "index.rst", HERE / "conf.py"]
                         + list((HERE / "_generated").rglob("*.md"))):
        digest.update(source.relative_to(HERE).as_posix().encode())
        digest.update(source.read_bytes())
    revision = digest.hexdigest()[:12]
    pattern = re.compile(r"href=([\"'])([^\"']+)\1")

    def replace(match):
        target = urlsplit(unescape(match.group(2)))
        if target.scheme or target.netloc or not target.path.endswith(".html"):
            return match.group(0)
        query = [(key, value) for key, value in parse_qsl(target.query, keep_blank_values=True)
                 if key != "docs_rev"]
        query.append(("docs_rev", revision))
        link = urlunsplit((target.scheme, target.netloc, target.path,
                           urlencode(query), target.fragment))
        return "href=" + match.group(1) + escape(link, quote=True) + match.group(1)

    for page in Path(app.outdir).rglob("*.html"):
        page.write_text(pattern.sub(replace, page.read_text(encoding="utf-8")),
                        encoding="utf-8")


def setup(app):
    app.connect("build-finished", version_document_links)
