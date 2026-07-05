export type SourceMetadata = Record<string, string>;

export type MetadataFilter = {
  key: string;
  value: string;
};

export type SearchResult = {
  score: number;
  chunk: string;
  chunk_id: string;
  source_id: string;
  source_path: string;
  source_metadata: SourceMetadata;
};

export type InsightSourceInfo = {
  source_id: string;
  source_path: string;
  source_metadata: SourceMetadata;
};

export type InsightResult = {
  score: number;
  insight: string;
  insight_id: string;
  topics?: string[];
  sources?: InsightSourceInfo[];
};

export type InsightRelatedGroup = {
  type: "first_hop" | "second_hop";
  sub_query?: string;
  insights: InsightResult[];
};

export type RetrieveInsightResult = {
  insight_id: string;
  insight: string;
  score: number;
  topics?: string[];
  sources?: InsightSourceInfo[];
  related: InsightRelatedGroup[];
};

export type SearchResponse = {
  results: {
    chunks: SearchResult[];
    insights: InsightResult[];
  };
};

export type RetrieveRelated = {
  entity: string;
  chunks: SearchResult[];
  second_level_related: Array<{
    entity: string;
    relationship: {
      label: string;
      metadata: Record<string, string | number | boolean | null>;
    };
    chunks: SearchResult[];
  }>;
};

export type RetrieveResult = SearchResult & {
  related: RetrieveRelated[];
};

export type RetrieveResponse = {
  retrieval_results: RetrieveResult[];
  insights: RetrieveInsightResult[];
};

export type AnswerModel = {
  id: string;
  label: string;
  default: boolean;
};

export type SourceDetail = {
  source_id: string;
  name?: string | null;
  file_name?: string | null;
  file_type?: string | null;
  storage_path: string;
  metadata: SourceMetadata;
  markdown_content: string;
};

export type SourceSummary = {
  source_id: string;
  name?: string | null;
  file_name?: string | null;
  file_type?: string | null;
  metadata: SourceMetadata;
  created_at: string;
  insight_count: number;
};

export type SourceInsight = {
  insight_id: string;
  insight: string;
  topics: string[];
  chunk_id: string;
  chunk_index?: number | null;
  chunk_preview: string;
};

export type SourceListResponse = {
  sources: SourceSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type SourceInsightsResponse = {
  insights: SourceInsight[];
};

export type StreamAnswerOptions = {
  query: string;
  model: string;
  source_ids?: string[];
  seed_count?: number;
  result_count?: number;
  rrf_k?: number;
  onAnswerDelta: (delta: string) => void;
  onResults: (results: RetrieveResponse["retrieval_results"]) => void;
};

export type FacetValue = { value: string; count: number };

export type FacetsResponse = { facets: Record<string, FacetValue[]> };

export type WorkingSet = {
  id: string;
  name: string;
  source_ids: string[];
  source_count: number;
  created_at: string;
};

export type CommunityRunRequest = {
  scope_mode: "working_set" | "ids" | "search" | "retrieve";
  working_set_id?: string;
  source_ids?: string[];
  criteria?: string[];
  search_options?: { limit: number; min_score: number };
  retrieve_options?: {
    seed_count?: number;
    result_count?: number;
    rrf_k?: number;
  };
  resolution?: number;
  semantic_threshold?: number;
  source_cooc_weight?: number;
  cross_source_top_k?: number;
  max_cross_source_queries?: number;
  cutoff?: number;
  min_community_size?: number;
  top_k_chunks?: number;
  summarize?: boolean;
  summarize_model?: string;
};

export type CommunityRunSummary = {
  run_id: string;
  status: "running" | "completed" | "failed";
  params_summary: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
};

export type CommunityRunDetail = CommunityRunSummary & {
  params: CommunityRunRequest;
  result?: CommunityResponse;
  error?: string;
};

export type StageEntry = {
  stage: string;
  message: string;
  timestamp?: string;
};

export type StreamRunCallbacks = {
  onStage?: (entry: StageEntry) => void;
  onResult?: (data: CommunityResponse) => void;
  onError?: (error: { message: string }) => void;
};

export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
    this.name = "UnauthorizedError";
  }
}

type AuthErrorListener = () => void;
let authErrorListener: AuthErrorListener | null = null;

export function onAuthError(fn: AuthErrorListener | null): void {
  authErrorListener = fn;
}

function ensureOk(response: Response): void {
  if (response.status === 401) {
    if (authErrorListener) authErrorListener();
    throw new UnauthorizedError();
  }
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
}

export async function authFetch(input: RequestInfo, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(input, { credentials: "include", ...init });
  if (response.status === 401 && authErrorListener) {
    authErrorListener();
  }
  return response;
}

async function postJson<T>(url: string, body: object): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  ensureOk(response);
  return response.json() as Promise<T>;
}

export function search(query: string, limit: number, minScore: number): Promise<SearchResponse> {
  return postJson<SearchResponse>("/api/search", {
    query,
    limit,
    min_score: minScore,
  });
}

export function retrieve(query: string): Promise<RetrieveResponse> {
  return postJson<RetrieveResponse>("/api/retrieve", { query });
}

export async function getAnswerModels(): Promise<AnswerModel[]> {
  const response = await fetch("/api/answer/models", { credentials: "include" });
  ensureOk(response);
  const payload = (await response.json()) as { models: AnswerModel[] };
  return payload.models ?? [];
}

export async function streamAnswer(options: StreamAnswerOptions): Promise<void> {
  const response = await fetch("/api/answer/stream", {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: options.query,
      model: options.model,
      ...(options.source_ids?.length ? { source_ids: options.source_ids } : {}),
      ...(options.seed_count != null ? { seed_count: options.seed_count } : {}),
      ...(options.result_count != null ? { result_count: options.result_count } : {}),
      ...(options.rrf_k != null ? { rrf_k: options.rrf_k } : {}),
    }),
  });

  if (!response.body) {
    ensureOk(response);
    throw new Error(`Request failed: ${response.status}`);
  }
  ensureOk(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const eventBlock of events) {
      const eventName = eventBlock
        .split("\n")
        .find((line) => line.startsWith("event: "))
        ?.slice(7)
        .trim();
      const data = eventBlock
        .split("\n")
        .find((line) => line.startsWith("data: "))
        ?.slice(6);

      if (!eventName || !data) {
        continue;
      }

      const payload = JSON.parse(data);
      if (eventName == "answer_delta") {
        options.onAnswerDelta(payload.delta ?? "");
      } else if (eventName == "results") {
        options.onResults(payload.retrieval_results ?? []);
      }
    }
  }
}

export async function getSource(sourceId: string): Promise<SourceDetail> {
  const response = await fetch(`/api/sources/${sourceId}`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<SourceDetail>;
}

export async function listSources(limit = 20, offset = 0, metadataFilters: MetadataFilter[] = [], q = ""): Promise<SourceListResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  metadataFilters.forEach((filter) => params.append("metadata", `${filter.key}:${filter.value}`));
  if (q.trim()) params.set("q", q.trim());
  const response = await fetch(`/api/sources?${params.toString()}`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<SourceListResponse>;
}

export async function getSourceInsights(sourceId: string): Promise<SourceInsightsResponse> {
  const response = await fetch(`/api/sources/${sourceId}/insights`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<SourceInsightsResponse>;
}

export async function getFacets(): Promise<FacetsResponse> {
  const response = await fetch("/api/sources/facets", { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<FacetsResponse>;
}

export async function listWorkingSets(): Promise<{ working_sets: WorkingSet[] }> {
  const response = await fetch("/api/working-sets", { credentials: "include" });
  ensureOk(response);
  const payload = (await response.json()) as { working_sets: WorkingSet[] };
  return payload.working_sets ? payload : { working_sets: [] };
}

export function createWorkingSet(name: string, source_ids: string[]): Promise<{ id: string }> {
  return postJson("/api/working-sets", { name, source_ids });
}

export function updateWorkingSet(id: string, payload: { name?: string; source_ids?: string[] }): Promise<WorkingSet> {
  return postJson(`/api/working-sets/${id}`, payload);
}

export async function deleteWorkingSet(id: string): Promise<{ deleted: string }> {
  const response = await fetch(`/api/working-sets/${id}`, { method: "DELETE", credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<{ deleted: string }>;
}

export function startCommunityRun(request: CommunityRunRequest): Promise<{ run_id: string }> {
  return postJson<{ run_id: string }>("/api/community/runs", request);
}

export async function listCommunityRuns(): Promise<{ runs: CommunityRunSummary[] }> {
  const response = await fetch("/api/community/runs", { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<{ runs: CommunityRunSummary[] }>;
}

export async function getCommunityRun(id: string): Promise<CommunityRunDetail> {
  const response = await fetch(`/api/community/runs/${id}`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<CommunityRunDetail>;
}

export async function streamRunEvents(
  runId: string,
  callbacks: StreamRunCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/community/runs/${runId}/events`, {
    credentials: "include",
    signal,
  });

  if (!response.body) {
    ensureOk(response);
    throw new Error(`Request failed: ${response.status}`);
  }
  ensureOk(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const eventBlock of events) {
      const eventName = eventBlock
        .split("\n")
        .find((line) => line.startsWith("event: "))
        ?.slice(7)
        .trim();
      const data = eventBlock
        .split("\n")
        .find((line) => line.startsWith("data: "))
        ?.slice(6);

      if (!eventName || !data) {
        continue;
      }

      const payload = JSON.parse(data);
      if (eventName === "stage") {
        callbacks.onStage?.(payload);
      } else if (eventName === "result") {
        callbacks.onResult?.(payload);
      } else if (eventName === "error") {
        callbacks.onError?.(payload);
      }
    }
  }
}

export type CommunityEntity = {
  entity_id: string;
  canonical_name: string;
  entity_type: string;
};

export type CommunityContributingSource = {
  source_id: string;
  source_name: string;
};

export type CommunityChunk = {
  chunk_id: string;
  source_id: string;
  source_name: string;
  entity_overlap_count: number;
  score: number;
  content: string;
};

export type CommunityItem = {
  community_id: string;
  is_cross_source: boolean;
  entity_count: number;
  entities: CommunityEntity[];
  contributing_sources: CommunityContributingSource[];
  chunks: CommunityChunk[];
  summary: string;
};

export type CommunityResponse = {
  metadata: {
    scope_mode: string;
    source_count: number;
    sources_excluded: Array<{ source_id: string; reason: string }>;
    parameters: {
      semantic_threshold: number;
      source_cooc_weight: number;
      cutoff: number;
      min_community_size: number;
      top_k_chunks: number;
    };
  };
  communities: CommunityItem[];
  virtual_merges?: number;
};

export type CommunityRequestOptions = {
  scope_mode: "ids" | "search" | "retrieve";
  source_ids?: string[];
  criteria?: string[];
  search_options?: { limit: number; min_score: number };
  retrieve_options?: {
    seed_count?: number;
    result_count?: number;
    rrf_k?: number;
  };
  community_options?: {
    semantic_threshold?: number;
    source_cooc_weight?: number;
    cross_source_top_k?: number;
    max_cross_source_queries?: number;
    resolution?: number;
    cutoff?: number;
    min_community_size?: number;
    top_k_chunks?: number;
  };
  summarize_model?: string;
};

export function community(options: CommunityRequestOptions): Promise<CommunityResponse> {
  return postJson<CommunityResponse>("/api/community", options);
}

export type ThemeReportSummary = {
  id: string;
  run_id: string;
  status: "completed" | "partial" | "failed";
  model: string;
  created_at: string;
};

export type ThemeReportCommunity = {
  label: string;
  type: string;
  confidence: number;
  cross_source: boolean;
  summary: string;
  key_entities: string[];
  key_sources: string[];
};

export type ThemeReportBucket = {
  label: string;
  communities: ThemeReportCommunity[];
};

export type ThemeReportDetail = ThemeReportSummary & {
  failed_community_ids: number[];
  report: {
    communities: ThemeReportCommunity[];
    buckets: ThemeReportBucket[];
    narrative: string;
    cleanup_recommendations: string[];
  };
};

export type EvidenceItem = {
  source_id: string;
  source_name: string;
  text: string;
};

export type SavedAnswer = {
  id: string;
  question: string;
  answer: string;
  model: string;
  params: Record<string, unknown>;
  evidence_snapshot: EvidenceItem[];
  created_at: string;
};

export function generateTheme(runId: string, model?: string): Promise<{ id: string }> {
  return postJson<{ id: string }>("/api/themes", {
    run_id: runId,
    ...(model ? { model } : {}),
  });
}

export async function listThemes(
  limit = 20,
  offset = 0,
): Promise<{ reports: ThemeReportSummary[]; total: number; limit: number; offset: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const response = await fetch(`/api/themes?${params.toString()}`, { credentials: "include" });
  ensureOk(response);
  return response.json();
}

export async function getTheme(id: string): Promise<ThemeReportDetail> {
  const response = await fetch(`/api/themes/${id}`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<ThemeReportDetail>;
}

export function regenerateTheme(id: string, model?: string): Promise<{ id: string }> {
  return postJson<{ id: string }>(`/api/themes/${id}/regenerate`, {
    ...(model ? { model } : {}),
  });
}

export function saveAnswer(
  question: string,
  answer: string,
  model: string,
  params: Record<string, unknown>,
  evidence: EvidenceItem[],
): Promise<{ id: string }> {
  return postJson<{ id: string }>("/api/answers", { question, answer, model, params, evidence });
}

export async function listAnswers(
  limit = 20,
  offset = 0,
): Promise<{ answers: SavedAnswer[]; total: number; limit: number; offset: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const response = await fetch(`/api/answers?${params.toString()}`, { credentials: "include" });
  ensureOk(response);
  return response.json();
}

export async function getAnswer(id: string): Promise<SavedAnswer> {
  const response = await fetch(`/api/answers/${id}`, { credentials: "include" });
  ensureOk(response);
  return response.json() as Promise<SavedAnswer>;
}

export async function deleteAnswer(id: string): Promise<{ deleted: string }> {
  const response = await fetch(`/api/answers/${id}`, {
    method: "DELETE",
    credentials: "include",
  });
  ensureOk(response);
  return response.json() as Promise<{ deleted: string }>;
}
