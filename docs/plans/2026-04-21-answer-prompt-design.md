# Answer Prompt Design

## Goal

Improve answer quality by making the answer prompt question-shaped instead of retrieval-shaped.

## Problem

The current prompt encourages broad coverage of retrieved themes. That causes the model to:

- summarize adjacent ideas instead of answering the exact question,
- overuse high-scoring but tangential chunks,
- produce retrieval-shaped prose instead of a crisp recommendation,
- mention evidence in a mechanical way rather than as a clean narrative.

## Approved Design

Use a relevance-first structured prompt.

The prompt should instruct the model to:

- answer the user's exact question as directly as possible,
- internally select the 2 to 4 most relevant pieces of evidence before writing,
- prioritize narrow domain cues in the question over broader adjacent themes,
- ignore interesting but tangential material even if it is highly scored,
- organize the answer around the question rather than the retrieval structure,
- separate direct support from synthesis,
- state evidence gaps precisely,
- write clean narrative prose with no chunk IDs, scores, or retrieval-language.

## Output Shape

The answer should always use this structure:

1. One-sentence conclusion.
2. Two short narrative paragraphs with only the strongest support.
3. One sentence on what the evidence does not establish.

## Non-Goals

- No citations or retrieval identifiers in the final answer.
- No mention of chunk IDs, scores, or "retrieved results."
- No attempt to cover every major theme in the retrieval.

## Verification

- Add a focused prompt test that checks the new instructions are present.
- Run the prompt test and the API suite after the change.
