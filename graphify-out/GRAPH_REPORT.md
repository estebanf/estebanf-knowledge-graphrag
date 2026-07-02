# Graph Report - .  (2026-07-02)

## Corpus Check
- 157 files · ~84,433 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1385 nodes · 2683 edges · 93 communities (75 shown, 18 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 390 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Cli Ingest Rationale|Cli Ingest Rationale]]
- [[_COMMUNITY_Auth Mcp Server|Auth Mcp Server]]
- [[_COMMUNITY_Chunking Chunk Validation|Chunking Chunk Validation]]
- [[_COMMUNITY_Scripts Duplicates Assess|Scripts Duplicates Assess]]
- [[_COMMUNITY_Source Cross Load|Source Cross Load]]
- [[_COMMUNITY_Insight Extraction Insights|Insight Extraction Insights]]
- [[_COMMUNITY_Docs Plans Insight|Docs Plans Insight]]
- [[_COMMUNITY_Ingest Delete Endpoint|Ingest Delete Endpoint]]
- [[_COMMUNITY_Client Ragclient Job|Client Ragclient Job]]
- [[_COMMUNITY_Search Endpoint Returns|Search Endpoint Returns]]
- [[_COMMUNITY_Frontend Lib Unauthorizederror|Frontend Lib Unauthorizederror]]
- [[_COMMUNITY_Ingestion Stage Build|Ingestion Stage Build]]
- [[_COMMUNITY_Parser Image Parse|Parser Image Parse]]
- [[_COMMUNITY_Retrieval Query Variants|Retrieval Query Variants]]
- [[_COMMUNITY_Package Frontend Json|Package Frontend Json]]
- [[_COMMUNITY_Remediation Remediate Image|Remediation Remediate Image]]
- [[_COMMUNITY_Retrieval And Chunk|Retrieval And Chunk]]
- [[_COMMUNITY_Session Auth User|Session Auth User]]
- [[_COMMUNITY_Workers Supervisor Worker|Workers Supervisor Worker]]
- [[_COMMUNITY_Graph Extraction Entities|Graph Extraction Entities]]
- [[_COMMUNITY_Jobs Cli List|Jobs Cli List]]
- [[_COMMUNITY_Merge Semantic Duplicates|Merge Semantic Duplicates]]
- [[_COMMUNITY_Embedding Retrieval Get|Embedding Retrieval Get]]
- [[_COMMUNITY_Answering Answer Models|Answering Answer Models]]
- [[_COMMUNITY_Auth Routes Session|Auth Routes Session]]
- [[_COMMUNITY_Tsconfig Frontend Json|Tsconfig Frontend Json]]
- [[_COMMUNITY_Config Cli File|Config Cli File]]
- [[_COMMUNITY_Ingestion Ingest File|Ingestion Ingest File]]
- [[_COMMUNITY_Worker Supervisor List|Worker Supervisor List]]
- [[_COMMUNITY_Cli Mode Uses|Cli Mode Uses]]
- [[_COMMUNITY_Jobs Worker Supervisor|Jobs Worker Supervisor]]
- [[_COMMUNITY_Schemas Basemodel Communityoptions|Schemas Basemodel Communityoptions]]
- [[_COMMUNITY_Worker Logging Config|Worker Logging Config]]
- [[_COMMUNITY_Client Sends Error|Client Sends Error]]
- [[_COMMUNITY_Frontend Lib App|Frontend Lib App]]
- [[_COMMUNITY_Retrieval Seed Expand|Retrieval Seed Expand]]
- [[_COMMUNITY_Sources Routes Source|Sources Routes Source]]
- [[_COMMUNITY_Hybrid Search Retrieval|Hybrid Search Retrieval]]
- [[_COMMUNITY_Frontend App Answermodelsresponse|Frontend App Answermodelsresponse]]
- [[_COMMUNITY_Jobs Routes Job|Jobs Routes Job]]
- [[_COMMUNITY_Retrieval Related Expand|Retrieval Related Expand]]
- [[_COMMUNITY_Youtube Sources Delete|Youtube Sources Delete]]
- [[_COMMUNITY_Workers Stop Returns|Workers Stop Returns]]
- [[_COMMUNITY_Worker Supervisor Subprocess|Worker Supervisor Subprocess]]
- [[_COMMUNITY_Frontend Sourcesexplorer Components|Frontend Sourcesexplorer Components]]
- [[_COMMUNITY_Retrieval Variant Concepts|Retrieval Variant Concepts]]
- [[_COMMUNITY_Graph Delete Legacy|Graph Delete Legacy]]
- [[_COMMUNITY_Frontend Auth Lib|Frontend Auth Lib]]
- [[_COMMUNITY_Ingest Routes Multipart|Ingest Routes Multipart]]
- [[_COMMUNITY_Retrieval Run Insight|Retrieval Run Insight]]
- [[_COMMUNITY_Retrieval Insight Retrieve|Retrieval Insight Retrieve]]
- [[_COMMUNITY_Storage Store Images|Storage Store Images]]
- [[_COMMUNITY_Migration Columns Required|Migration Columns Required]]
- [[_COMMUNITY_Worker Supervisor Workersupervisor|Worker Supervisor Workersupervisor]]
- [[_COMMUNITY_Opencode Json Playwright|Opencode Json Playwright]]
- [[_COMMUNITY_Client Apierror Runtimeerror|Client Apierror Runtimeerror]]
- [[_COMMUNITY_Profile Retrieve Scripts|Profile Retrieve Scripts]]
- [[_COMMUNITY_Cli Ids Passes|Cli Ids Passes]]
- [[_COMMUNITY_Retrieve Cli Command|Retrieve Cli Command]]
- [[_COMMUNITY_Remediate Insights Script|Remediate Insights Script]]
- [[_COMMUNITY_Insightcard Frontend Components|Insightcard Frontend Components]]
- [[_COMMUNITY_Resultcard Frontend Components|Resultcard Frontend Components]]
- [[_COMMUNITY_Job Lifecycle Cancel|Job Lifecycle Cancel]]
- [[_COMMUNITY_Migration Creates Index|Migration Creates Index]]
- [[_COMMUNITY_Docker Compose Service|Docker Compose Service]]
- [[_COMMUNITY_Smoke E2e Scripts|Smoke E2e Scripts]]
- [[_COMMUNITY_Graph Linking Link|Graph Linking Link]]
- [[_COMMUNITY_Bucketpopover Frontend Components|Bucketpopover Frontend Components]]
- [[_COMMUNITY_Retrieve Routes Schemas|Retrieve Routes Schemas]]
- [[_COMMUNITY_Conftest Cli Config|Conftest Cli Config]]
- [[_COMMUNITY_Delete Legacy Edges|Delete Legacy Edges]]
- [[_COMMUNITY_Mcp Probe Scripts|Mcp Probe Scripts]]
- [[_COMMUNITY_Search Routes Searchresponse|Search Routes Searchresponse]]
- [[_COMMUNITY_Backup Scripts Script|Backup Scripts Script]]
- [[_COMMUNITY_Reset Scripts Script|Reset Scripts Script]]
- [[_COMMUNITY_Start Scripts Script|Start Scripts Script]]
- [[_COMMUNITY_Stop Scripts Script|Stop Scripts Script]]
- [[_COMMUNITY_Rest Package For|Rest Package For]]
- [[_COMMUNITY_Routes Route|Routes Route]]
- [[_COMMUNITY_Routes Route Modules|Routes Route Modules]]
- [[_COMMUNITY_Prompts All Llm|Prompts All Llm]]
- [[_COMMUNITY_Cli Http Path|Cli Http Path]]
- [[_COMMUNITY_Chunk Concepts|Chunk Concepts]]
- [[_COMMUNITY_Knowledge Graphrag Pkg|Knowledge Graphrag Pkg]]
- [[_COMMUNITY_Source Operations Readme|Source Operations Readme]]

## God Nodes (most connected - your core abstractions)
1. `get_connection()` - 38 edges
2. `RagClient` - 35 edges
3. `WorkerSupervisor` - 35 edges
4. `RetrievalCandidate` - 31 edges
5. `KeyStore` - 23 edges
6. `_get_client()` - 23 edges
7. `TraceLogger` - 23 edges
8. `WorkerInfo` - 23 edges
9. `execute_ingestion_pipeline()` - 21 edges
10. `_use_api()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `test_cleanup_from_insight_stage_removes_insight_artifacts()` --calls--> `cleanup_from_stage()`  [INFERRED]
  tests/test_job_lifecycle.py → src/rag/ingestion.py
- `test_execute_pipeline_from_insight_stage_skips_graph_linking()` --calls--> `execute_ingestion_pipeline()`  [INFERRED]
  tests/test_job_lifecycle.py → src/rag/ingestion.py
- `test_trace_logger_emits_only_when_enabled()` --calls--> `TraceLogger`  [INFERRED]
  tests/test_retrieval.py → src/rag/retrieval.py
- `test_chat_json_opencode_calls_opencode_endpoint()` --calls--> `_chat_json_opencode()`  [INFERRED]
  tests/test_retrieval.py → src/rag/retrieval.py
- `test_generate_query_variants_uses_opencode()` --calls--> `generate_query_variants()`  [INFERRED]
  tests/test_retrieval.py → src/rag/retrieval.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Retrieval Pipeline Concepts** — concepts_retrieval_pipeline, concepts_query_variant, concepts_dense_prefilter, concepts_graph_expansion, concepts_insight_expansion [EXTRACTED 1.00]
- **Server Deployment Surface** — docs_plans_2026_05_21_server_deployment_principal_auth_layer, docs_plans_2026_05_21_server_deployment_worker_process_supervision, docs_plans_2026_05_21_server_deployment_mcp_streamable_http, docs_plans_2026_05_21_server_deployment_server_deployment_architecture [EXTRACTED 1.00]
- **Retrieval Performance Fix** — docs_solutions_performance_issues_rag_retrieval_vector_prefilter_and_query_fanout_dense_scan_bottleneck, docs_solutions_performance_issues_rag_retrieval_vector_prefilter_and_query_fanout_binary_quantized_hnsw_prefilter, docs_solutions_performance_issues_rag_retrieval_vector_prefilter_and_query_fanout_query_fanout_reduction [EXTRACTED 1.00]

## Communities (93 total, 18 thin omitted)

### Community 0 - "Cli Ingest Rationale"
Cohesion: 0.07
Nodes (63): Argument, help, Option, cancel_job(), community_ids(), community_retrieve(), community_search(), configure_command() (+55 more)

### Community 1 - "Auth Mcp Server"
Cohesion: 0.06
Nodes (51): ASGIApp, FastMCP, Receive, Request, Scope, Send, get_default_keystore(), KeyRecord (+43 more)

### Community 2 - "Chunking Chunk Validation"
Cohesion: 0.06
Nodes (52): BaseSettings, RecursiveCharacterTextSplitter, Returns True if chunk passes quality check. Returns True on any error., Sample-based quality validation. Returns True if chunks pass, False to fail the, _score_chunk(), validate_chunks(), chunk_document(), ChunkData (+44 more)

### Community 3 - "Scripts Duplicates Assess"
Cohesion: 0.06
Nodes (38): main(), UnionFind, _length_band(), main(), _prefix_key(), _digit_sets_differ(), fetch_embeddings(), fuzzy_candidates() (+30 more)

### Community 4 - "Source Cross Load"
Cohesion: 0.11
Nodes (39): Graph, _build_igraph(), ChunkResult, Community, ContributingSource, _cosine_similarity(), detect_communities(), EntityNode (+31 more)

### Community 5 - "Insight Extraction Insights"
Cohesion: 0.10
Nodes (33): ProgressCallback, _cleanup_source_insights(), _has_existing_links(), _load_chunk_rows(), main(), _process_source(), _embedding_literal(), extract_and_store_insights() (+25 more)

### Community 6 - "Docs Plans Insight"
Cohesion: 0.07
Nodes (34): Auth and Server Invariants, Insight Extraction Behavior, MCP Server, Worker Supervisor, Binary-Quantized HNSW, Dense Prefilter, Entity Mention, Full-Precision Rerank (+26 more)

### Community 7 - "Ingest Delete Endpoint"
Cohesion: 0.09
Nodes (26): create_app(), FastAPI, _client(), _result(), test_community_endpoint_ids_mode(), test_community_endpoint_passes_summarize_model(), test_community_endpoint_rejects_invalid_scope_mode(), test_community_endpoint_retrieve_mode_passes_options() (+18 more)

### Community 8 - "Client Ragclient Job"
Cohesion: 0.11
Nodes (4): Any, Response, RagClient, Yield SSE events with ``event``/``data`` parsed.

### Community 9 - "Search Endpoint Returns"
Cohesion: 0.13
Nodes (28): HybridSearchResults, _answer_models(), Path, TestClient, _client(), _retrieve_results(), _search_results(), _source_detail() (+20 more)

### Community 10 - "Frontend Lib Unauthorizederror"
Cohesion: 0.10
Nodes (26): AuthErrorListener, authFetch(), community(), CommunityChunk, CommunityContributingSource, CommunityEntity, CommunityItem, ensureOk() (+18 more)

### Community 11 - "Ingestion Stage Build"
Cohesion: 0.17
Nodes (26): _build_completed_stage_entry(), _build_failed_stage_entry(), _build_processing_stage_entry(), cancel_job(), check_duplicate(), cleanup_from_stage(), _cleanup_graph_artifacts(), _cleanup_insight_artifacts_for_job() (+18 more)

### Community 12 - "Parser Image Parse"
Cohesion: 0.14
Nodes (23): Exception, describe_image(), _describe_docling_pictures(), _describe_markdown_images(), _get_docling_converter(), parse_document(), parse_to_markdown(), ParseError (+15 more)

### Community 13 - "Retrieval Query Variants"
Cohesion: 0.20
Nodes (25): aggregate_root_score(), _chat_json(), _chat_json_opencode(), finalize_insight_results(), finalize_root_results(), generate_insight_query_variants(), _generate_insight_sub_query(), generate_query_variants() (+17 more)

### Community 14 - "Package Frontend Json"
Cohesion: 0.08
Nodes (25): dependencies, @monaco-editor/react, react, react-dom, react-markdown, devDependencies, jsdom, tailwindcss (+17 more)

### Community 15 - "Remediation Remediate Image"
Cohesion: 0.16
Nodes (21): main(), main(), AffectedImageSource, AffectedSource, ensure_schema_ready(), get_affected_sources(), get_image_placeholder_sources(), get_preflight_counts() (+13 more)

### Community 16 - "Retrieval And Chunk"
Cohesion: 0.11
Nodes (22): _apply_expanded_chunk_text(), _expand_neighbor_contexts(), _load_chunk_ids_for_entity(), NullConnection, test_aggregate_root_score_uses_component_weights_and_bonus(), test_apply_expanded_chunk_text_updates_nested_results(), test_chat_json_opencode_calls_opencode_endpoint(), test_expand_neighbor_contexts_dedupes_chunk_lookups() (+14 more)

### Community 17 - "Session Auth User"
Cohesion: 0.17
Nodes (17): SessionResolver, Install the session-cookie resolver (called by Phase 4 wiring)., set_session_resolver(), app(), InMemoryUserAuthService, FastAPI, Tests for session-based authentication: login, logout, /me, cookie handling., In-memory user/session store usable in tests. (+9 more)

### Community 18 - "Workers Supervisor Worker"
Cohesion: 0.16
Nodes (15): get_log(), get_supervisor_dep(), launch_workers(), LaunchResponse, list_workers(), Worker management routes: launch, stop, list, log., stop_all_workers(), stop_worker() (+7 more)

### Community 19 - "Graph Extraction Entities"
Cohesion: 0.17
Nodes (22): extract_and_store_graph(), extract_entities(), extract_relationships(), _normalise_name(), store_entities_and_edges(), _validate_entity(), _mock_llm_response(), test_extract_and_store_graph_iterates_all_chunks() (+14 more)

### Community 20 - "Jobs Cli List"
Cohesion: 0.10
Nodes (3): _make_job_row(), test_jobs_list_shows_jobs(), test_jobs_status_shows_record()

### Community 21 - "Merge Semantic Duplicates"
Cohesion: 0.17
Nodes (16): _digit_sets_differ(), fetch_embeddings(), find_candidate_pairs(), _has_exclusive_keyword_conflict(), _length_band(), main(), merge_cluster(), merge_memgraph() (+8 more)

### Community 22 - "Embedding Retrieval Get"
Cohesion: 0.15
Nodes (19): embed_and_store_chunks(), get_embeddings(), Generate embeddings for chunks and update the DB.     chunk_rows: list of (chunk, Call OpenRouter embeddings API in batches. Returns list of embedding vectors., _build_chunk_filter_sql(), dense_retrieve(), _fetch_chunk_candidates_by_ids(), _fetch_same_source_neighbor_candidates() (+11 more)

### Community 23 - "Answering Answer Models"
Cohesion: 0.18
Nodes (16): get_supported_answer_models(), _build_answer_prompt(), _is_anthropic_model(), _is_supported_model(), _opencode_go_headers(), _opencode_go_key(), Response, _require_api_key() (+8 more)

### Community 24 - "Auth Routes Session"
Cohesion: 0.13
Nodes (14): ABC, APIRouter, build_router(), _clear_session_cookie(), install_default_session_resolver(), LoginRequest, LoginResponse, MeResponse (+6 more)

### Community 25 - "Tsconfig Frontend Json"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, allowSyntheticDefaultImports, esModuleInterop, forceConsistentCasingInFileNames, isolatedModules, jsx, lib (+10 more)

### Community 26 - "Config Cli File"
Cohesion: 0.21
Nodes (14): CliConfig, CliConfigError, _from_file(), load_cli_config(), Path, CLI config: ``~/.config/rag/config.toml`` with env-var overrides., Resolve precedence: env > file > none., require_config() (+6 more)

### Community 27 - "Ingestion Ingest File"
Cohesion: 0.19
Nodes (18): compute_md5(), ingest_file(), Path, submit_ingestion_job(), cleanup(), cleanup_existing_file(), ingested(), Path (+10 more)

### Community 28 - "Worker Supervisor List"
Cohesion: 0.19
Nodes (5): InMemoryWorkerStore, PostgresWorkerStore, Persists worker rows in the ``worker_processes`` table., WorkerInfo, WorkerStore

### Community 29 - "Cli Mode Uses"
Cohesion: 0.18
Nodes (16): CliRunner, fake_client(), Exercise the API-backed code paths in the CLI via a mocked RagClient., Make ``_get_client()`` return a MagicMock with context-manager semantics., runner(), test_configure_writes_config(), test_health_uses_api(), test_ingest_uses_api() (+8 more)

### Community 30 - "Jobs Worker Supervisor"
Cohesion: 0.19
Nodes (15): datetime, _dt_to_ts(), Background worker supervisor.  Spawns and tracks ``rag _worker-run`` subprocesse, _ts_to_dt(), _client(), TestClient, Tests for the jobs API: list, status, retry, cancel., test_cancel_job_endpoint() (+7 more)

### Community 31 - "Schemas Basemodel Communityoptions"
Cohesion: 0.22
Nodes (16): BaseModel, CommunityOptions, CommunityRequest, InsightResult, InsightSourceInfo, RetrieveOptions, SearchOptions, SearchRequest (+8 more)

### Community 32 - "Worker Logging Config"
Cohesion: 0.21
Nodes (13): _add_event_defaults(), configure_logging(), Configure structlog for JSON output with contextvars support.      Call once at, claim_next_job(), Connection, recover_stuck_jobs(), run_worker(), test_add_event_defaults_populates_required_logging_fields() (+5 more)

### Community 33 - "Client Sends Error"
Cohesion: 0.21
Nodes (14): BaseTransport, _mock(), Path, Tests for the RagClient HTTP wrapper using httpx.MockTransport., test_delete_source_sends_hard_flag(), test_error_response_raises_api_error(), test_health_returns_payload(), test_launch_and_stop_workers() (+6 more)

### Community 34 - "Frontend Lib App"
Cohesion: 0.17
Nodes (13): Authenticated(), BucketEntry, Mode, useAuth(), Login(), SourcePanel(), SourcePanelProps, AnswerModel (+5 more)

### Community 35 - "Retrieval Seed Expand"
Cohesion: 0.25
Nodes (16): EntityCandidate, expand_seed_candidate(), _generate_entity_query(), _load_entities_for_chunks(), _load_seed_entities(), RetrievalCandidate, SecondHopEntityCandidate, _select_second_hop_entities_from_chunks() (+8 more)

### Community 36 - "Sources Routes Source"
Cohesion: 0.19
Nodes (13): FileResponse, SourceDetail, SourceInsightsResponse, SourceListResponse, download_source(), get_source(), get_source_insights(), get_sources() (+5 more)

### Community 37 - "Hybrid Search Retrieval"
Cohesion: 0.27
Nodes (14): _expand_chunk_texts(), hybrid_search(), _token_count(), _candidate(), _empty_insights(), test_expand_chunk_texts_uses_uuid_ids_without_text_cast(), test_hybrid_search_filters_by_cosine_similarity(), test_hybrid_search_respects_limit() (+6 more)

### Community 38 - "Frontend App Answermodelsresponse"
Cohesion: 0.15
Nodes (9): App(), answerModelsResponse, retrieveResponse, searchResponse, secondSourcesResponse, sourceInsightsResponse, sourceResponse, sourcesResponse (+1 more)

### Community 39 - "Jobs Routes Job"
Cohesion: 0.24
Nodes (12): get_job(), get_job_stats(), JobListResponse, JobStatsItem, JobStatsResponse, JobStatus, JobSummary, list_jobs() (+4 more)

### Community 40 - "Retrieval Related Expand"
Cohesion: 0.19
Nodes (8): expand_seed_insight(), InsightSearchResult, _load_related_insights(), FakeInsightDriver, FakeInsightSession, test_expand_seed_insight_batches_second_hop_embeddings(), test_expand_seed_insight_returns_related_and_second_hop(), test_load_related_insights_uses_related_to()

### Community 41 - "Youtube Sources Delete"
Cohesion: 0.23
Nodes (9): list_youtube_sources(), purge_youtube_sources(), run(), _load_script_module(), test_list_youtube_sources_filters_active_kind_youtube(), test_purge_youtube_sources_dry_run_does_not_delete(), test_purge_youtube_sources_execute_deletes_matches(), test_script_main_passes_execute_and_filters() (+1 more)

### Community 42 - "Workers Stop Returns"
Cohesion: 0.24
Nodes (12): client(), Path, TestClient, Tests for /api/workers (launch, stop, list, log)., supervisor(), test_launch_workers_returns_ids(), test_list_workers(), test_list_workers_excludes_stopped_by_default() (+4 more)

### Community 44 - "Worker Supervisor Subprocess"
Cohesion: 0.24
Nodes (12): Path, Tests for WorkerSupervisor: subprocess spawn, reap, orphan reconciliation., supervisor(), test_launch_n_spawns_multiple(), test_launch_records_row_and_starts_subprocess(), test_orphan_reconciliation_marks_dead_pids_crashed(), test_reap_loop_marks_crashed_on_unexpected_exit(), test_shutdown_stops_all_workers() (+4 more)

### Community 45 - "Frontend Sourcesexplorer Components"
Cohesion: 0.18
Nodes (7): DetailTab, SourcesExplorer(), SourcesExplorerProps, sourceTitle(), MetadataFilter, SourceInsight, SourceSummary

### Community 46 - "Retrieval Variant Concepts"
Cohesion: 0.18
Nodes (11): Current Retrieval Behavior, Expanded Variant, HyDE Variant, Query Variant, Retrieval Pipeline, Step-Back Variant, OpenCode Query Variant Generation, Expanded Variant Gate (+3 more)

### Community 47 - "Graph Delete Legacy"
Cohesion: 0.24
Nodes (8): collect_edge_counts(), delete_legacy_edges(), main(), delete_source(), Soft- or hard-delete a source., get_graph_driver(), delete_stored_file(), test_get_graph_driver_yields_and_closes()

### Community 48 - "Frontend Auth Lib"
Cohesion: 0.29
Nodes (8): AuthContext, AuthContextValue, AuthState, onAuthError(), fetchMe(), login(), logout(), MeResponse

### Community 49 - "Ingest Routes Multipart"
Cohesion: 0.29
Nodes (9): ingest_multipart(), ingest_text(), IngestResponse, IngestTextRequest, _parse_metadata(), Ingest routes: multipart file upload and JSON text submission., Accept a multipart upload and submit an ingestion job., Accept inline text and submit an ingestion job as a markdown document. (+1 more)

### Community 50 - "Retrieval Run Insight"
Cohesion: 0.22
Nodes (9): Lock, _consume_llm_budget(), Atomically check and consume one LLM budget slot. Returns False if exhausted., run_insight_first_stage_retrieval(), should_include_expanded_variant(), test_run_insight_first_stage_retrieval_fuses_per_variant_results(), test_run_insight_first_stage_retrieval_omits_step_back_and_gates_expanded(), test_run_insight_first_stage_retrieval_searches_variants_concurrently() (+1 more)

### Community 51 - "Retrieval Insight Retrieve"
Cohesion: 0.36
Nodes (9): _fetch_insight_sources_and_topics(), insight_dense_retrieve(), insight_hybrid_search(), _insight_rows_to_candidates(), insight_sparse_retrieve(), _insight_weighted_reciprocal_rank_fusion(), InsightCandidate, InsightSourceRef (+1 more)

### Community 52 - "Storage Store Images"
Cohesion: 0.36
Nodes (7): Path, store_file(), store_markdown_images(), test_store_file_uses_versioned_original_name(), test_store_markdown_images_copies_referenced_images(), test_store_markdown_images_skips_missing_images(), test_store_markdown_images_skips_remote_urls()

### Community 54 - "Worker Supervisor Workersupervisor"
Cohesion: 0.32
Nodes (3): _pid_alive(), Stop every active worker. Returns the IDs that were stopped., Mark workers with dead PIDs as ``crashed`` (called on startup).

### Community 55 - "Opencode Json Playwright"
Cohesion: 0.29
Nodes (6): mcp, playwright, command, enabled, type, $schema

### Community 56 - "Client Apierror Runtimeerror"
Cohesion: 0.29
Nodes (4): RuntimeError, ApiError, Path, HTTP client used by the CLI to talk to the RAG API server.

### Community 57 - "Profile Retrieve Scripts"
Cohesion: 0.57
Nodes (6): main(), _patch_retrieval(), _print_report(), Any, _record(), _wrap()

### Community 58 - "Cli Ids Passes"
Cohesion: 0.48
Nodes (5): _result(), test_community_ids_basic(), test_community_ids_with_overrides(), test_community_retrieve_passes_options(), test_community_search_passes_criteria()

### Community 59 - "Retrieve Cli Command"
Cohesion: 0.38
Nodes (3): _retrieval_response(), test_retrieve_command_passes_filters_and_overrides(), test_retrieve_command_prints_json_response()

### Community 61 - "Remediate Insights Script"
Cohesion: 0.52
Nodes (6): _load_script(), test_remediate_insights_batch_size_limits_total_sources(), test_remediate_insights_no_pending_sources(), test_remediate_insights_source_id_force_cleans_before_rebuild(), test_remediate_insights_source_id_processes_when_no_existing_links(), test_remediate_insights_source_id_skips_existing_without_force()

### Community 62 - "Insightcard Frontend Components"
Cohesion: 0.53
Nodes (5): InsightCard(), InsightCardProps, primarySource(), sourceLabel(), InsightResult

### Community 63 - "Resultcard Frontend Components"
Cohesion: 0.47
Nodes (5): badgeLabel(), ResultCard(), ResultCardProps, sourceLabel(), SearchResult

### Community 66 - "Docker Compose Service"
Cohesion: 0.70
Nodes (5): Local Docker Stack, Backend Service, Frontend Service, Memgraph Service, Postgres Service

### Community 67 - "Smoke E2e Scripts"
Cohesion: 0.50
Nodes (4): note(), RAG_API_KEY, RAG_SERVER_URL, smoke_e2e.sh script

### Community 68 - "Graph Linking Link"
Cohesion: 0.40
Nodes (3): link_graph(), Compatibility stage kept for queued jobs; no graph mutations., test_link_graph_is_noop_and_skips_memgraph_writes()

### Community 70 - "Retrieve Routes Schemas"
Cohesion: 0.50
Nodes (3): retrieve_route(), AnswerRequest, RetrieveRequest

### Community 71 - "Conftest Cli Config"
Cohesion: 0.50
Nodes (3): _isolate_cli_config(), Shared pytest fixtures., Point the CLI config at a non-existent temp path and clear env overrides.      W

### Community 72 - "Delete Legacy Edges"
Cohesion: 0.83
Nodes (3): _load_script_module(), test_collect_edge_counts_reads_related_to_and_mentioned_in(), test_delete_legacy_edges_executes_for_both_types()

## Knowledge Gaps
- **95 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RagClient` connect `Client Ragclient Job` to `Client Apierror Runtimeerror`, `Client Sends Error`, `Config Cli File`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `create_app()` connect `Ingest Delete Endpoint` to `Auth Mcp Server`, `Scripts Duplicates Assess`, `Search Endpoint Returns`, `Auth Routes Session`, `Jobs Worker Supervisor`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `get_connection()` connect `Scripts Duplicates Assess` to `Cli Ingest Rationale`, `Worker Logging Config`, `Source Cross Load`, `Insight Extraction Insights`, `Hybrid Search Retrieval`, `Jobs Routes Job`, `Youtube Sources Delete`, `Ingestion Stage Build`, `Graph Delete Legacy`, `Remediation Remediate Image`, `Merge Semantic Duplicates`, `Worker Supervisor Workersupervisor`, `Ingestion Ingest File`, `Worker Supervisor List`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `get_connection()` (e.g. with `main()` and `main()`) actually correct?**
  _`get_connection()` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `WorkerSupervisor` (e.g. with `LaunchResponse` and `WorkerInfoModel`) actually correct?**
  _`WorkerSupervisor` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `RetrievalCandidate` (e.g. with `_search_results()` and `_search_results()`) actually correct?**
  _`RetrievalCandidate` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _202 weakly-connected nodes found - possible documentation gaps or missing edges._