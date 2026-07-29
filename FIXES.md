# Repair summary

## Root cause of the reported 500

Open WebUI's attachment button processes files inside Open WebUI. It does not call this API's `/upload` route. The old `/v1/chat/completions` implementation discarded the context Open WebUI supplied and always searched the separate local Qdrant collection, even when that collection had never been created.

## Main fixes

- Added `RAG_MODE=auto|openwebui|qdrant` and made `auto` the safe default.
- Preserved Open WebUI-injected `<source>` context instead of replacing it.
- Missing/empty Qdrant collections now return no results rather than throwing an unhandled exception.
- Optional Qdrant failure in `auto` mode falls back to upstream chat instead of returning 500.
- Added proper OpenAI-compatible streaming via server-sent events.
- Forwarded common generation and tool-calling parameters.
- Removed Open WebUI-only message metadata before forwarding to Groq/OpenAI-compatible APIs.
- Added clear HTTP errors for invalid PDFs, empty custom RAG, Qdrant outages, bad configuration, and embedding dimension mismatches.
- Added file-size, file-type, encrypted-PDF, and scanned-PDF validation.
- Added lazy embedding-model loading.
- Added Nomic's required `search_document:` and `search_query:` prefixes.
- Replaced one-Qdrant-request-per-chunk insertion with one batched upsert.
- Added document IDs, filenames, page numbers, and chunk metadata to payloads.
- Added optional per-document retrieval filtering.
- Fixed long-paragraph chunking and prevented empty chunks.
- Cleaned duplicate and unused dependencies; added the missing `python-multipart` dependency.
- Added `/health`, complete setup docs, `.env.example`, and unit tests.
