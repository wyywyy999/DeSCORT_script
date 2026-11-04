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
