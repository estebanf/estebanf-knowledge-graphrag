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


def test_backend_entrypoints_do_not_import_docling():
    # The whole point of the refactor: the backend image runs without Docling.
    # Importing the API app and the worker must not pull it in — this subprocess
    # guard fails fast in CI before an image build would fail at runtime.
    cmd = [
        sys.executable,
        "-c",
        (
            "import os, json, sys; "
            "os.environ['RAG_DISABLE_MCP'] = '1'; "
            "import rag.api.main; "
            "import rag.worker; "
            "import rag.ingestion; "
            "print(json.dumps({'docling': any(n.startswith('docling') for n in sys.modules)}))"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    payload = result.stdout.strip()
    assert payload == '{"docling": false}', result.stdout + result.stderr
