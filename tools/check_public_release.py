"""Verify reviewed bytes and reject importing unrelated Git history.

The review manifest must be updated manually after content review. This check
enforces that review boundary; it does not detect every possible sensitive record.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

MANIFEST = "PUBLIC_FILES.json"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def validate(files):
    if MANIFEST not in files:
        raise ValueError("Missing reviewed file manifest")
    approved = json.loads(files[MANIFEST])
    if set(files) != set(approved) | {MANIFEST}:
        raise ValueError("Unreviewed file set")
    for name, data in files.items():
        if len(data) > 256_000:
            raise ValueError("Oversized public file")
        data.decode("utf-8")
        if b"\x00" in data:
            raise ValueError("Binary payload in public release")
        if name != MANIFEST and hashlib.sha256(data).hexdigest() != approved[name]:
            raise ValueError("File changed since content review")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tracked = git(root, "ls-files", "-z").decode().strip("\0").split("\0")
    if not tracked or tracked == [""]:
        raise ValueError("No staged public files")
    for name in tracked:
        if (root / name).is_symlink():
            raise ValueError("Symlink not permitted")
    validate({name: (root / name).read_bytes() for name in tracked})
    # Validate the index too, so an older/unreviewed staged file cannot be pushed.
    validate({name: git(root, "show", ":" + name) for name in tracked})
    if args.history:
        commits = git(root, "rev-list", "--all").decode().split()
        roots = git(root, "rev-list", "--all", "--max-parents=0").decode().split()
        if len(roots) != 1:
            raise ValueError("Expected exactly one clean history root")
        for commit in commits:
            rows = git(root, "ls-tree", "-r", "-z", commit).split(b"\0")
            files = {}
            for row in filter(None, rows):
                header, name = row.split(b"\t", 1)
                mode, kind, sha = header.split()
                if kind != b"blob" or mode not in (b"100644", b"100755"):
                    raise ValueError("Submodules/symlinks are not permitted")
                files[name.decode()] = git(root, "cat-file", "blob", sha.decode())
            validate(files)
        print(f"Reviewed file integrity verified for {len(commits)} clean-history commit(s).")
    else:
        print("Reviewed working tree and index verified.")


if __name__ == "__main__":
    main()
