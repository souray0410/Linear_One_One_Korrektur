import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("public_gate", Path(__file__).parents[1] / "tools/check_public_release.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def approved():
    content = b"synthetic example only\n"
    return {"example.txt": content, "PUBLIC_FILES.json": json.dumps(
        {"example.txt": hashlib.sha256(content).hexdigest()}).encode()}


def test_approved_bytes_pass():
    gate.validate(approved())


def test_unreviewed_file_and_modified_content_are_rejected():
    files = approved(); files["unreviewed.txt"] = b"unexpected"
    with pytest.raises(ValueError, match="file set"): gate.validate(files)
    files = approved(); files["example.txt"] = b"changed"
    with pytest.raises(ValueError, match="changed"): gate.validate(files)


def test_missing_manifest_and_binary_are_rejected():
    with pytest.raises(ValueError, match="Missing"): gate.validate({"x": b"y"})
    files = approved(); files["example.txt"] = b"\0"
    with pytest.raises(ValueError, match="Binary"): gate.validate(files)
