const argHelp: Record<string, string> = {
  resolution: "Clustering resolution parameter. Higher values produce more, smaller communities. Default 1.0.",
  semantic_threshold: "Minimum cosine similarity for entities to be considered related (0–1). Default 0.85.",
  source_cooc_weight: "Weight applied to cross-source entity co-occurrence edges relative to within-source MENTIONS edges. Default 0.5.",
  cross_source_top_k: "Top-K cross-source entity pairs to materialize per source pair. Higher values increase recall at cost of performance. Default 5.",
  max_cross_source_queries: "Maximum number of cross-source pairs to evaluate. Caps total work per run. Default 50.",
  cutoff: "Edge weight cutoff for pruning the entity graph before community detection (0–1). Default 0.5.",
  min_community_size: "Minimum number of entities required for a cluster to be reported as a community. Default 3.",
  top_k_chunks: "Maximum chunks returned per community for evidence display. Default 5.",
};

export default argHelp;
