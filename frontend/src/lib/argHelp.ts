const argHelp: Record<string, string> = {
  resolution: "Clustering resolution parameter. Higher values produce more, smaller communities. Default 1.0.",
  semantic_threshold: "Minimum cosine similarity for entities to be considered related (0–1). Default 0.85.",
  source_cooc_weight: "Weight applied to cross-source entity co-occurrence edges relative to within-source MENTIONS edges. Default 0.5.",
  cross_source_top_k: "Top-K cross-source entity pairs to materialize per source pair. Higher values increase recall at cost of performance. Default 5.",
  max_cross_source_queries: "Maximum number of cross-source pairs to evaluate. Caps total work per run. Default 50.",
  cutoff: "Edge weight cutoff for pruning the entity graph before community detection (0–1). Default 0.5.",
  min_community_size: "Minimum number of entities required for a cluster to be reported as a community. Default 3.",
  top_k_chunks: "Maximum chunks returned per community for evidence display. Default 5.",
  search_limit: "Maximum number of results returned per type (chunks and insights). Default 10.",
  search_min_score: "Minimum similarity score a result must meet to be included (0–1). Default 0.7.",
  retrieve_source_ids: "Restrict retrieval to these source IDs only (comma-separated). Leave empty to search all sources.",
  retrieve_filters: "Key=value metadata filters applied to retrieval, comma-separated (e.g. author=Jane,kind=article). Leave empty for no filtering.",
  retrieve_seed_count: "Maximum number of seed chunks used to start graph expansion. Default 10.",
  retrieve_result_count: "Maximum number of final results returned. Default 5.",
  retrieve_rrf_k: "Reciprocal Rank Fusion constant; higher values give more weight to dense (semantic) results over sparse (keyword) results. Default 60.",
  retrieve_entity_confidence_threshold: "Minimum confidence required for an entity to be included in graph expansion (0–1). Default 0.75.",
  retrieve_first_hop_similarity_threshold: "Maximum similarity distance allowed for first-hop entity expansion (0–1). Default 0.5.",
  retrieve_second_hop_similarity_threshold: "Maximum similarity distance allowed for second-hop entity expansion (0–1). Default 0.5.",
  retrieve_trace: "Include a detailed step-by-step retrieval trace in the response, useful for debugging.",
};

export default argHelp;
