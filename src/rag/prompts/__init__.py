"""All LLM prompt constants for the RAG system."""

DOCUMENT_PROFILING = """\
You are a document analyst. Analyze the document excerpt below and classify it.

Return ONLY a valid JSON object with exactly these fields:
{
  "structure_type": "well-structured|loosely-structured|unstructured",
  "heading_consistency": "consistent|inconsistent|none",
  "content_density": "uniform|variable",
  "primary_content_type": "prose|tabular|mixed|qa_pairs|code|transcript",
  "avg_section_length": "short|medium|long",
  "has_tables": true|false,
  "has_code_blocks": true|false,
  "domain": "legal|financial|technical|general|medical|policy"
}

Document excerpt:
"""

CHUNK_VALIDATION = """\
Evaluate this document chunk for retrieval quality. Score it on:
- completeness: does it contain complete thoughts?
- coherence: is it internally coherent?
- appropriate_length: is it neither too short nor too long?

Return ONLY: {"pass": true} or {"pass": false, "reason": "brief explanation"}

Chunk:
"""

ENTITY_EXTRACTION = """Extract named entities from the following text. Return ONLY a JSON array.
Each item must have: "canonical_name" (string), "entity_type" (one of: {types}), "aliases" (list of strings).
Return [] if no entities found.

Text:
{text}"""

RELATIONSHIP_EXTRACTION = """Given these entities: {entity_names}

Extract relationships from the text. Return ONLY a JSON array.
Each item must have: "source" (canonical_name), "target" (canonical_name), "type" (string), "confidence" (0.0-1.0).
Use concise relationship types like OWNS, EMPLOYS, GOVERNS, REQUIRES, IS_PART_OF, RELATED_TO.
Only include relationships grounded in the text.
Return [] if none found.

Text:
{text}"""

PROPOSITION_DECOMPOSITION = (
    "Break the following text into atomic, self-contained propositions. "
    "Each must be a single factual statement understandable without context.\n"
    "Return ONLY a JSON array of strings.\n\nText:\n"
)

ANSWER_GENERATION = (
    "Use only the retrieved results below.\n\n"
    "Write a comprehensive narrative answer that clearly answers the question first, "
    "then elaborates on the supporting evidence.\n\n"
    "Requirements:\n"
    "1. Begin with a direct thesis that answers the exact question.\n"
    "2. Write 3 to 5 cohesive paragraphs, not bullets.\n"
    "3. Build the answer around the most relevant evidence, not around all major themes in the retrieval.\n"
    "4. Prioritize evidence that matches the narrow angle of the question. If the question mentions insurance, "
    "foreground insurance and cyber-risk evidence over general AI strategy.\n"
    "5. Use broader AI-native themes only when they strengthen the insurance positioning.\n"
    "6. Synthesize the evidence into a clear recommendation rather than listing retrieved points.\n"
    "7. Be explicit about what is directly supported by the retrieval, and avoid adding claims that are not grounded in it.\n"
    "8. If the retrieval contains conflicting signals, state the conflict instead of forcing a clean answer.\n"
    "9. End with a brief statement of what the retrieval does not establish, if there is an important gap.\n"
    "10. Do not mention chunk IDs, scores, or retrieval mechanics.\n"
    "11. Do not elevate a concept into the main answer just because it appears in a high-scoring chunk. "
    "Relevance to the exact question matters more than retrieval score.\n"
    "12. Avoid generic AI phrasing and avoid filler.\n\n"
    "Style guidance:\n"
    "- Sound analytical and well structured.\n"
    "- Be specific and concrete.\n"
    "- Prefer a strong narrative arc: answer, reasoning, implications, limitations.\n\n"
    "Retrieved results:\n"
)

COMMUNITY_SUMMARIZATION = (
    "Craft a compelling narrative summarizing these chunks of related information"
)

QUERY_VARIANTS = """You are generating bounded retrieval query variants.
Return ONLY a JSON object with keys: original, hyde, expanded, step_back, decomposed.
- original must be the exact input query
- hyde must be a short hypothetical answer passage for dense retrieval
- expanded must add synonyms, aliases, abbreviations, and related terms
- step_back must be a more general background query
- decomposed must contain at most {max_decomposed} focused sub-queries
- avoid near-duplicate variants

Query:
{query}
"""

INSIGHT_QUERY_VARIANTS = """You are generating retrieval query variants for insight search.
Return ONLY a JSON object with keys: original, hyde, expanded, step_back.
- original must be the exact input query
- hyde must be a short hypothetical insight passage
- expanded must add synonyms, aliases, and related concepts
- step_back must be a more general background query
- avoid near-duplicate variants

Query:
{query}
"""

ENTITY_SELECTION = """You are selecting graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {max_entities} entity names.

User query:
{query}

Seed chunk:
{seed_chunk}

Entities:
{entities}
"""

ENTITY_QUERY_GENERATION = """You are generating a retrieval sub-query.
Return ONLY a JSON object with key query.

Original user query:
{query}

Seed chunk:
{seed_chunk}

Entity:
{entity_name}
"""

SECOND_HOP_ENTITY_SELECTION = """You are selecting second-hop graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {max_entities} entity names.

User query:
{query}

Seed chunk:
{seed_chunk}

Current entity:
{entity_name}

Candidates:
{candidates}
"""

INSIGHT_EXTRACTION = """\
From the chunk below, extract insights that are specific, meaningful, and useful. Prioritize information that reveals causes, consequences, patterns, tradeoffs, risks, opportunities, decisions, assumptions, changes, tensions, or implications. Exclude generic summaries, obvious statements, repeated points, and details that do not affect understanding or action.

Focus only on insights that are related to one or more of these topics:
AI Adoption, AI Opportunity Strategy, AI Governance, AI Readiness, AI Use Case Prioritization, Workflow Automation, Workflow Intelligence, Human-in-the-Loop Design, Productized Services, Offer Design, Solution Architecture, Commercial Solutioning, Delivery Governance, Operating Models, B2B Transformation, Vertical Solutions, Cross-Functional Alignment, Technical Leadership, Executive Communication, Business Outcomes, Product Management, Product Strategy, Product Discovery, Product Roadmapping, Product Prioritization, Product Requirements, Product-Market Fit, Product Positioning, Product Packaging, Platform Strategy, Workflow Product Design, AI Product Management, AI Feature Strategy, Customer Research, Voice of Customer, Jobs-to-Be-Done, MVP Definition, Experimentation, Product Analytics, Go-to-Market Alignment, Buyer Enablement, Sales Enablement, Customer Onboarding, Customer Success Strategy, Product Operations, Stakeholder Alignment, Delivery Readiness, Acceptance Criteria, Product Governance, Outcome Management

Return your findings in a JSON document that follows this structure:
{{"insights": [{{"insight":"","topics":[""]}}]}}

Return {{"insights": []}} if no relevant insights are found.

Chunk
```
{chunk}
```"""
