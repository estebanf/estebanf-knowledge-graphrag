import subprocess
import sys


def test_importing_cli_does_not_eagerly_import_heavy_modules():
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import rag.cli; "
            "print(json.dumps({"
            "'ingestion': 'rag.ingestion' in sys.modules, "
            "'community': 'rag.community' in sys.modules"
            "}))"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    payload = result.stdout.strip()
    assert payload == '{"ingestion": false, "community": false}'


def test_importing_parser_does_not_eagerly_import_docling():
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import rag.parser; "
            "print(json.dumps({'docling': any(name.startswith('docling') for name in sys.modules)}))"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    payload = result.stdout.strip()
    assert payload == '{"docling": false}'
