import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";


const searchResponse = {
  results: [
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
