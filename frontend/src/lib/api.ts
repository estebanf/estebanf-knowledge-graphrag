export type SourceMetadata = Record<string, string>;

export type SearchResult = {
  score: number;
  chunk: string;
  chunk_id: string;
  source_id: string;
  source_path: string;
  source_metadata: SourceMetadata;
};

export type SearchResponse = {
  results: SearchResult[];
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

export type StreamAnswerOptions = {
  query: string;
  model: string;
  onAnswerDelta: (delta: string) => void;
  onResults: (results: RetrieveResponse["retrieval_results"]) => void;
};

async function postJson<T>(url: string, body: object): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

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
  const response = await fetch("/api/answer/models");
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  const payload = (await response.json()) as { models: AnswerModel[] };
  return payload.models;
}

export async function streamAnswer(options: StreamAnswerOptions): Promise<void> {
  const response = await fetch("/api/answer/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: options.query,
      model: options.model,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

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
  const response = await fetch(`/api/sources/${sourceId}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<SourceDetail>;
}
