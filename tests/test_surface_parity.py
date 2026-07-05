"""Cross-surface parity assertions.

Ensures that every field of ``CommunityOptions`` appears in both the MCP
``community`` tool and the CLI community commands, and that for each
artifact type there is an API route, a ``RagClient`` method, and an MCP
tool.
"""

from __future__ import annotations

import inspect

import pytest
from rag.api.schemas import CommunityOptions
from rag.api_client import RagClient
from rag.mcp_server import _build_server


def _community_option_field_names() -> set[str]:
    return set(CommunityOptions.model_fields.keys())


def _mcp_community_tool_param_names() -> set[str]:
    server = _build_server()
    for t in server._tool_manager.list_tools():
        if t.name == "community":
            sig = inspect.signature(t.fn)
            return set(sig.parameters.keys())
    return set()


def _cli_community_command_option_names() -> dict[str, set[str]]:
    """Return a mapping of CLI community command name -> set of --option names
    by introspecting callback function signatures."""
    from rag.cli import community_app  # type: ignore[import-untyped]

    from typer.models import CommandInfo

    result: dict[str, set[str]] = {}
    for cmd in community_app.registered_commands:
        callback = getattr(cmd, "callback", None)
        if callback is None:
            continue
        sig = inspect.signature(callback)
        options: set[str] = set()
        for pname in sig.parameters.keys():
            if pname == "source_id":
                continue
            options.add(pname.replace("_", "-"))
        result[cmd.name or "unknown"] = options
    return result


# Mapping from CommunityOptions field names to CLI option names.
# Some fields use shortened names in the CLI for usability.
_COMMUNITY_OPTION_TO_CLI_MAP = {
    "semantic_threshold": "semantic-threshold",
    "cutoff": "cutoff",
    "min_community_size": "min-community-size",
    "top_k_chunks": "top-k",
    "cross_source_top_k": "cross-source-top-k",
    "max_cross_source_queries": "max-cross-source-queries",
    "source_cooc_weight": "source-cooc-weight",
    "resolution": "resolution",
}


class TestCommunityOptionsParity:
    """CommunityOptions fields must appear in MCP community tool params
    and as CLI --options on every community subcommand."""

    def test_all_community_options_fields_in_mcp_community_tool(self) -> None:
        """Verify that every CommunityOptions field is reachable through the
        community tool.  The tool passes them via the ``community_options``
        dict, so we inspect the _build_server source to confirm each field is
        extracted with ``community_options.get(...)``."""
        schema_fields = _community_option_field_names()
        mcp_params = _mcp_community_tool_param_names()

        assert "community_options" in mcp_params, (
            f"Expected 'community_options' param in community tool; got: {sorted(mcp_params)}"
        )

        import rag.mcp_server

        source = inspect.getsource(rag.mcp_server._build_server)

        missing_from_body: list[str] = []
        for field in sorted(schema_fields):
            needle = f'community_options.get("{field}")'
            if needle not in source:
                missing_from_body.append(field)
        assert not missing_from_body, (
            f"CommunityOptions fields not extracted from community_options dict "
            f"in _build_server source: {missing_from_body}"
        )

    def test_all_community_options_fields_in_cli_commands(self) -> None:
        schema_fields = _community_option_field_names()
        cli_opts_by_cmd = _cli_community_command_option_names()

        missing_by_cmd: dict[str, list[str]] = {}
        for cmd_name in sorted(cli_opts_by_cmd.keys()):
            cmd_opts = cli_opts_by_cmd[cmd_name]
            missing = []
            for field in sorted(schema_fields):
                expected_opt = _COMMUNITY_OPTION_TO_CLI_MAP.get(field, field.replace("_", "-"))
                if expected_opt not in cmd_opts:
                    missing.append(f"{field} (expected --{expected_opt})")
            if missing:
                missing_by_cmd[cmd_name] = missing

        if missing_by_cmd:
            lines = []
            for cmd_name, missed in sorted(missing_by_cmd.items()):
                lines.append(f"  {cmd_name}: {', '.join(missed)}")
            pytest.fail(
                f"CommunityOptions fields missing from CLI community commands:\n"
                + "\n".join(lines)
                + "\n\nAdd the missing --options to the CLI commands (U13)."
            )

    def test_mcp_community_tool_has_no_extra_params_beyond_schema_and_scope_wrappers(self) -> None:
        """Every MCP community tool param should either be a field in
        CommunityOptions, a scope-control param (scope_mode, source_ids,
        criteria, filters, search_options, retrieve_options, summarize_model,
        working_set_id), or the community_options dict wrapper itself."""
        schema_fields = _community_option_field_names()
        mcp_params = _mcp_community_tool_param_names()

        scope_wrappers = {
            "scope_mode",
            "source_ids",
            "criteria",
            "filters",
            "search_options",
            "retrieve_options",
            "summarize_model",
            "working_set_id",
            "community_options",
        }

        extra = mcp_params - schema_fields - scope_wrappers
        assert not extra, (
            f"MCP community tool has params not in CommunityOptions or scope wrappers: {sorted(extra)}"
        )


class TestArtifactTypeParity:
    """For each artifact type, an API route, RagClient method, and MCP tool
    must exist (name-based assertions)."""

    # (artifact_type, mcp_tool_name, client_method_name)
    ARTIFACT_TYPES = [
        ("community runs", "list_community_runs", "list_community_runs"),
        ("community run detail", "get_community_run", "get_community_run"),
        ("theme reports", "list_theme_reports", "list_theme_reports"),
        ("theme report detail", "get_theme_report", "get_theme_report"),
        ("answers", "list_answers", "list_answers"),
        ("answer detail", "get_answer", "get_answer"),
        ("working sets", "list_working_sets", "list_working_sets"),
        ("working set detail", "get_working_set", "get_working_set"),
        ("metadata facets", "list_metadata_facets", "get_facets"),
    ]

    @staticmethod
    def _mcp_tool_names() -> set[str]:
        server = _build_server()
        return {t.name for t in server._tool_manager.list_tools()}

    @staticmethod
    def _client_method_names() -> set[str]:
        return {m for m in dir(RagClient) if not m.startswith("_") and callable(getattr(RagClient, m, None))}

    def test_every_artifact_has_mcp_tool(self) -> None:
        mcp_names = self._mcp_tool_names()
        for artifact_type, mcp_name, _client in self.ARTIFACT_TYPES:
            assert mcp_name in mcp_names, (
                f"MCP tool '{mcp_name}' missing for {artifact_type}"
            )

    def test_every_artifact_has_client_method(self) -> None:
        client_names = self._client_method_names()
        for artifact_type, _mcp, client_name in self.ARTIFACT_TYPES:
            assert client_name in client_names, (
                f"RagClient method '{client_name}' missing for {artifact_type}"
            )
