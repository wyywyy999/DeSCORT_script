# Building and hosting the documentation

The Read the Docs project builds from `.readthedocs.yaml` in the repository root.
It installs only `docs/requirements.txt`; it does not install DeSCORT or run models.

For a local preview, create a separate documentation environment (Python 3.11 is
used online) and run these commands from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000/`. `docs/conf.py` stages existing Markdown tutorials
and result images into the ignored `docs/_generated` directory before building.
Edit the maintained source Markdown, not the generated copies. Script links lead
to GitHub; documentation and image links remain inside the built website.

In Read the Docs, set the default branch to `main` and the configuration file to
`.readthedocs.yaml`. The version named `latest` normally follows that branch; it
is not necessary to create a Git branch named `latest`. After pushing, check the
new build or click **Rebuild**. Use **View docs** after the build succeeds, then
copy the actual website address into `tutorials/README.md` and the root README.
