import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";


const searchResponse = {
  results: {
    chunks: [
      {
        score: 0.706,
        chunk: "The shift towards agentic workflows fundamentally alters the marginal cost of intelligence.",
        chunk_id: "chunk-1",
        source_id: "source-1",
        source_path: "/tmp/economics.md",
        source_metadata: {
          kind: "report",
          source: "Gartner",
          title: "Economics of GenAI",
        },
      },
    ],
    insights: [],
  },
};


const retrieveResponse = {
  retrieval_results: [
    {
      score: 0.91,
      chunk: "Root chunk about economics.",
      chunk_id: "chunk-root",
      source_id: "source-1",
      source_path: "/tmp/economics.md",
      source_metadata: {
        kind: "report",
        source: "Gartner",
      },
      related: [
        {
          entity: "Economics of GenAI",
          chunks: [
            {
              score: 0.82,
              chunk: "Related chunk about margin structure.",
              chunk_id: "chunk-related",
              source_id: "source-1",
              source_path: "/tmp/economics.md",
              source_metadata: {
                kind: "report",
                source: "Gartner",
              },
            },
          ],
          second_level_related: [],
        },
      ],
    },
  ],
};


const sourceResponse = {
  source_id: "source-1",
  name: "Economics of GenAI",
  file_name: "economics.md",
  file_type: "text/markdown",
  storage_path: "/tmp/economics.md",
  metadata: {
    kind: "report",
    source: "Gartner",
  },
  markdown_content: "# Economics\n\nRendered body",
};

const sourcesResponse = {
  sources: [
    {
      source_id: "source-1",
      name: "Economics of GenAI",
      file_name: "economics.md",
      file_type: "text/markdown",
      metadata: {
        kind: "report",
        source: "Gartner",
        domain: "strategy",
      },
      created_at: "2026-05-01T12:30:00Z",
      insight_count: 1,
    },
  ],
  total: 21,
  limit: 20,
  offset: 0,
};

const secondSourcesResponse = {
  sources: [
    {
      source_id: "source-2",
      name: "Warehouse Metrics",
      file_name: "warehouse.md",
      file_type: "text/markdown",
      metadata: {
        kind: "youtube",
        source: "Breham Group",
      },
      created_at: "2026-05-02T12:30:00Z",
      insight_count: 0,
    },
  ],
  total: 21,
  limit: 20,
  offset: 20,
};

const sourceInsightsResponse = {
  insights: [
    {
      insight_id: "insight-1",
      insight: "Agentic workflows lower marginal analysis cost.",
      topics: ["economics", "automation"],
      chunk_id: "chunk-1",
      chunk_index: 2,
      chunk_preview: "The shift towards agentic workflows fundamentally alters cost.",
    },
  ],
};

const answerModelsResponse = {
  models: [
    { id: "google/gemma-4-31b-it", label: "google/gemma-4-31b-it", default: true },
    { id: "deepseek/deepseek-v3.2", label: "deepseek/deepseek-v3.2", default: false },
  ],
};

function createStreamResponse(chunks: string[]) {
  const encoder = new TextEncoder();
  let index = 0;
  return {
    ok: true,
    body: {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) {
              return { done: true, value: undefined };
            }
            const value = encoder.encode(chunks[index]);
            index += 1;
            return { done: false, value };
          },
          releaseLock() {},
        };
      },
    },
  };
}

function installClipboard() {
  const writeText = vi.fn(async () => undefined);
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
  return writeText;
}


describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("renders the simplified shell with the new title", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "estebanf's RAG" })).toBeInTheDocument();
    expect(screen.queryByText(/Workspaces/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New Session/i })).not.toBeInTheDocument();
  });

  test("submits search and renders shared result cards", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.clear(screen.getByLabelText(/minimum score/i));
    await userEvent.type(screen.getByLabelText(/minimum score/i), "0.55");
    await userEvent.clear(screen.getByLabelText(/result count/i));
    await userEvent.type(screen.getByLabelText(/result count/i), "7");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));

    expect(await screen.findByText(/Top Results/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Economics of GenAI/i })).toBeInTheDocument();
    expect(screen.getByText(/Gartner/i)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/search",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          query: "economics of agents",
          limit: 7,
          min_score: 0.55,
        }),
      }),
    );
  });

  test("submits search with the enter key", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents{enter}");

    expect(await screen.findByRole("heading", { name: /Economics of GenAI/i })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("shows score and count controls only in search mode", async () => {
    render(<App />);

    expect(screen.getByLabelText(/minimum score/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/result count/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /retrieve/i }));

    expect(screen.queryByLabelText(/minimum score/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/result count/i)).not.toBeInTheDocument();
  });

  test("submits retrieve with the enter key", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/retrieve") {
        return {
          ok: true,
          json: async () => retrieveResponse,
        };
      }
      return {
        ok: true,
        json: async () => sourceResponse,
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /retrieve/i }));
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents{enter}");

    expect(await screen.findByText(/Economics of GenAI/i, { selector: ".related-group__label" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/retrieve",
      expect.objectContaining({ method: "POST" }),
    );
  });

  test("loads answer models and streams answer before showing evidence", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/answer/models") {
        return {
          ok: true,
          json: async () => answerModelsResponse,
        };
      }
      if (url === "/api/answer/stream") {
        return createStreamResponse([
          'event: answer_delta\ndata: {"delta":"Thoughtful"}\n\n',
          'event: answer_delta\ndata: {"delta":" answer"}\n\n',
          'event: results\ndata: {"retrieval_results":[{"score":0.91,"chunk":"Root chunk about economics.","chunk_id":"chunk-root","source_id":"source-1","source_path":"/tmp/economics.md","source_metadata":{"kind":"report","source":"Gartner"},"related":[]}]}\n\n',
        ]);
      }
      return {
        ok: true,
        json: async () => sourceResponse,
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /answer/i }));
    expect(await screen.findByLabelText(/answer model/i)).toBeInTheDocument();
    expect(screen.getByDisplayValue("google/gemma-4-31b-it")).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /^answer$/i }));

    expect(await screen.findByText(/Thoughtful answer/i)).toBeInTheDocument();
    expect(await screen.findByText(/Root chunk about economics\./i)).toBeInTheDocument();
  });

  test("switches to retrieve and renders nested related sections", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/retrieve") {
        return {
          ok: true,
          json: async () => retrieveResponse,
        };
      }
      return {
        ok: true,
        json: async () => sourceResponse,
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /retrieve/i }));
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /retrieve/i }));

    expect(await screen.findByText(/Economics of GenAI/i, { selector: ".related-group__label" })).toBeInTheDocument();
    expect(screen.getByText(/Related chunk about margin structure\./i)).toBeInTheDocument();
  });

  test("opens rendered markdown in the side panel", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/search") {
        return {
          ok: true,
          json: async () => searchResponse,
        };
      }
      return {
        ok: true,
        json: async () => sourceResponse,
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    await screen.findByText(/The shift towards agentic workflows/i);

    await userEvent.click(screen.getByRole("button", { name: /view/i }));

    expect(await screen.findByRole("complementary", { name: /source preview/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Economics" })).toBeInTheDocument();
    expect(screen.getByText(/Rendered body/i)).toBeInTheDocument();
  });

  test("opens sources explorer and renders recent source markdown with insights", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/sources?limit=20&offset=0") {
        return {
          ok: true,
          json: async () => sourcesResponse,
        };
      }
      if (url === "/api/sources/source-1/insights") {
        return {
          ok: true,
          json: async () => sourceInsightsResponse,
        };
      }
      if (url === "/api/sources/source-1") {
        return {
          ok: true,
          json: async () => sourceResponse,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));

    expect(await screen.findByRole("button", { name: /Economics of GenAI/i })).toBeInTheDocument();
    expect(screen.getByText(/May 1, 2026/i)).toBeInTheDocument();
    expect(screen.getByText("kind")).toBeInTheDocument();
    expect(screen.getByText("report")).toBeInTheDocument();
    expect(screen.getByText("domain")).toBeInTheDocument();
    expect(screen.getByText("strategy")).toBeInTheDocument();
    expect(await screen.findByText(/Agentic workflows lower marginal analysis cost/i)).toBeInTheDocument();
    expect(screen.getByText(/economics/i, { selector: ".source-insight-topic" })).toBeInTheDocument();
    expect(screen.getByText(/chunk 2/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Economics" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /content/i }));

    expect(await screen.findByRole("heading", { name: "Economics" })).toBeInTheDocument();
    expect(screen.getByText(/Rendered body/i)).toBeInTheDocument();
  });

  test("copies source insights and individual insight text from the sources explorer", async () => {
    const writeText = installClipboard();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/sources?limit=20&offset=0") {
        return {
          ok: true,
          json: async () => sourcesResponse,
        };
      }
      if (url === "/api/sources/source-1/insights") {
        return {
          ok: true,
          json: async () => sourceInsightsResponse,
        };
      }
      if (url === "/api/sources/source-1") {
        return {
          ok: true,
          json: async () => sourceResponse,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));

    const sourceButton = await screen.findByRole("button", { name: /Economics of GenAI/i });
    expect(sourceButton).toHaveTextContent(/May 1, 2026/i);
    expect(sourceButton).toHaveTextContent(/source-1/i);

    await userEvent.click(await screen.findByRole("button", { name: /copy source insights/i }));
    expect(writeText).toHaveBeenCalledWith(JSON.stringify(sourceInsightsResponse.insights, null, 2));

    await userEvent.click(screen.getByRole("button", { name: /copy insight text/i }));
    expect(writeText).toHaveBeenCalledWith("Agentic workflows lower marginal analysis cost.");
  });

  test("defaults source detail to content when there are no insights", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/sources?limit=20&offset=0") {
        return {
          ok: true,
          json: async () => sourcesResponse,
        };
      }
      if (url === "/api/sources/source-1/insights") {
        return {
          ok: true,
          json: async () => ({ insights: [] }),
        };
      }
      if (url === "/api/sources/source-1") {
        return {
          ok: true,
          json: async () => sourceResponse,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));

    expect(await screen.findByRole("heading", { name: "Economics" })).toBeInTheDocument();
  });

  test("paginates the sources lane", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/sources?limit=20&offset=0") {
        return {
          ok: true,
          json: async () => sourcesResponse,
        };
      }
      if (url === "/api/sources?limit=20&offset=20") {
        return {
          ok: true,
          json: async () => secondSourcesResponse,
        };
      }
      if (url === "/api/sources/source-1/insights" || url === "/api/sources/source-2/insights") {
        return {
          ok: true,
          json: async () => ({ insights: [] }),
        };
      }
      if (url === "/api/sources/source-1") {
        return {
          ok: true,
          json: async () => sourceResponse,
        };
      }
      if (url === "/api/sources/source-2") {
        return {
          ok: true,
          json: async () => ({ ...sourceResponse, source_id: "source-2", name: "Warehouse Metrics" }),
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));
    expect(await screen.findByText(/Page 1 of 2/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /next page/i }));

    expect(await screen.findByRole("button", { name: /Warehouse Metrics/i })).toBeInTheDocument();
    expect(screen.getByText(/Page 2 of 2/i)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/sources?limit=20&offset=20");
  });

  test("filters sources by clicked metadata attributes and removes filter chips", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url === "/api/sources?limit=20&offset=0" ||
        url === "/api/sources?limit=20&offset=0&metadata=kind%3Areport" ||
        url === "/api/sources?limit=20&offset=0&metadata=kind%3Areport&metadata=source%3AGartner"
      ) {
        return {
          ok: true,
          json: async () => sourcesResponse,
        };
      }
      if (url === "/api/sources/source-1/insights") {
        return {
          ok: true,
          json: async () => sourceInsightsResponse,
        };
      }
      if (url === "/api/sources/source-1") {
        return {
          ok: true,
          json: async () => sourceResponse,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));
    const kindFilter = await screen.findByRole("button", { name: /filter by kind report/i });
    await userEvent.click(kindFilter);

    expect(await screen.findByRole("button", { name: /remove filter kind report/i })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/sources?limit=20&offset=0&metadata=kind%3Areport");

    await userEvent.click(screen.getByRole("button", { name: /filter by source gartner/i }));

    expect(fetchMock).toHaveBeenCalledWith("/api/sources?limit=20&offset=0&metadata=kind%3Areport&metadata=source%3AGartner");

    await userEvent.click(screen.getByRole("button", { name: /remove filter kind report/i }));

    expect(fetchMock).toHaveBeenCalledWith("/api/sources?limit=20&offset=0&metadata=source%3AGartner");
  });

  test("clear resets the query and removes rendered results", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    const input = screen.getByLabelText(/semantic query/i);
    await userEvent.type(input, "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    await userEvent.click(screen.getByRole("button", { name: /clear/i }));

    expect(input).toHaveValue("");
    expect(screen.queryByRole("heading", { name: /Economics of GenAI/i })).not.toBeInTheDocument();
  });

  test("download action points to the source download endpoint", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    await screen.findByText(/The shift towards agentic workflows/i);

    const card = screen.getByText(/The shift towards agentic workflows/i).closest("article");
    expect(card).not.toBeNull();
    const downloadLink = within(card as HTMLElement).getByRole("link", { name: /download/i });

    await waitFor(() => {
      expect(downloadLink).toHaveAttribute("href", "/api/sources/source-1/download");
    });
  });

  test("copies all visible results as json", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    const writeText = installClipboard();
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    await userEvent.click(screen.getByRole("button", { name: /copy results/i }));

    expect(writeText).toHaveBeenCalledWith(JSON.stringify(searchResponse.results, null, 2));
    expect(screen.getByText(/results copied/i)).toBeInTheDocument();
  });

  test("add to bucket button appears on result cards", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    expect(screen.getByRole("button", { name: /add to bucket/i })).toBeInTheDocument();
  });

  test("adding same source twice keeps only one bucket entry", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    const addBtn = screen.getByRole("button", { name: /add to bucket/i });
    await userEvent.click(addBtn);
    await userEvent.click(addBtn);

    await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));
    const entries = screen.getAllByTestId("bucket-entry");
    expect(entries).toHaveLength(1);
  });

  test("bucket popover lists collected sources and copies IDs one per line", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    const writeText = installClipboard();
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    await userEvent.click(screen.getByRole("button", { name: /add to bucket/i }));
    await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));

    expect(screen.getByRole("region", { name: /collected sources/i })).toBeInTheDocument();
    expect(screen.getByTestId("bucket-entry")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /^copy$/i }));
    expect(writeText).toHaveBeenCalledWith("source-1");
    expect(screen.getByText(/ids copied/i)).toBeInTheDocument();
  });

  test("bucket clear empties the collected sources", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
    await screen.findByRole("heading", { name: /Economics of GenAI/i });

    await userEvent.click(screen.getByRole("button", { name: /add to bucket/i }));
    await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));

    const dialog = screen.getByRole("region", { name: /collected sources/i });
    await userEvent.click(within(dialog).getByRole("button", { name: /clear/i }));

    // Popover closes after clear; reopen to verify empty state
    await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));
    expect(screen.getByText(/no sources collected/i)).toBeInTheDocument();
    expect(screen.queryByTestId("bucket-entry")).not.toBeInTheDocument();
  });

  test("copies an individual chunk from the result card", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => searchResponse,
    }));
    const writeText = installClipboard();
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText(/semantic query/i), "economics of agents");
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    await screen.findByText(/The shift towards agentic workflows/i);

    await userEvent.click(screen.getByRole("button", { name: /copy chunk/i }));

    expect(writeText).toHaveBeenCalledWith(
      "The shift towards agentic workflows fundamentally alters the marginal cost of intelligence.",
    );
    expect(screen.getByText(/chunk copied/i)).toBeInTheDocument();
  });
});
