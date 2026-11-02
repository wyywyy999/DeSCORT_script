#!/usr/bin/env python3
"""Run a tutorial with an explicit JSON argv profile and optional CLI overrides."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(profile, extra=()):
    script = (ROOT / profile["script"]).resolve()
    if ROOT not in script.parents or not script.is_file() or script.suffix != ".py":
        raise ValueError("Profile script must be an existing Python file inside this project")
    launcher = profile.get("launcher", ["{python}"])
    arguments = profile.get("args", [])
    if not launcher or not all(isinstance(x, str) for x in launcher):
        raise ValueError("launcher must be a nonempty list of strings")
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError("args must be a list of literal argument strings")
    command = [x.replace("{python}", sys.executable) for x in launcher]
    command += [str(script), *arguments, *extra]
    command = [os.path.expandvars(os.path.expanduser(x)) for x in command]
    unresolved = sorted({v for x in command for v in re.findall(r"\$(?:\{[^}]+\}|[A-Za-z_]\w*)", x)})
    if unresolved:
        raise ValueError("Set path variables before running: " + ", ".join(unresolved))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without importing or running the workflow")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    try:
        profile = json.loads(args.config.read_text())
        command = build_command(profile, extra)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        return subprocess.call(command, cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
