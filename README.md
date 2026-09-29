# genkitx-goodmem

[GoodMem](https://docs.goodmem.ai) memory for [Genkit](https://genkit.dev):
a native retriever and indexer, plus agent tools. Documents are chunked,
embedded and searched server-side; this plugin wraps the official
`@pairsystems/goodmem` SDK.

**Version 0.3.0.** Verified against GoodMem server **v1.0.320**.

> **Upgrading from 0.1.2.** 0.1.2 talked to GoodMem over hand-written `fetch`.
> Two defects stand out: `get_memory` with `includeContent` called `.json()`
> on a body the server returns as `text/plain`, so it **failed for every
> memory**; and a retrieval that failed — a space whose embedder was
> unavailable, say — returned `success: true` with zero results and no
> indication anything was wrong. See [Changes in 0.2.0](#changes-in-020).

> **Upgrading from 0.2.0.** Every id is now required to be a UUID, checked
> before any request is made: a model-supplied `memoryId` of `..` sent
> `DELETE /v1/`. A non-UUID in `spaceIds` or `rerankerId` now throws when the
> plugin is created. See [Ids are UUIDs](#ids-are-uuids).
>
> 0.2.0 also failed **every** retrieval when `rerankerId` was set (HTTP 400
> `Unrecognized field "rerankerId"`); 0.2.1 sends the reranker as a
> post-processor. See [Changes in 0.2.1](#changes-in-021).

> **New in 0.3.0.** An opt-in `llmId` has GoodMem write a grounded answer from
> the retrieved chunks: `abstractReply` in `goodmem/search` output and
> `goodmem_abstract_reply` on each retrieved document. Nothing changes unless
> you set it. See [Grounded answers from an LLM](#grounded-answers-from-an-llm).

## Install

```bash
npm install genkitx-goodmem
```

## Use

```ts
import { genkit } from 'genkit';
import { goodmem } from 'genkitx-goodmem';

const ai = genkit({
  plugins: [
    goodmem({
      baseUrl: process.env.GOODMEM_BASE_URL!,
      apiKey: process.env.GOODMEM_API_KEY!,
      spaceIds: ['<space-uuid>'],
    }),
  ],
});
```

`spaceIds` is required: the model never chooses a space.

### Retrieve and index

```ts
import { Document } from 'genkit';

const docs = await ai.retrieve({
  retriever: 'goodmem/memories',
  query: 'What is the main finding?',
  options: { k: 5 },
});

await ai.index({ indexer: 'goodmem/memories', documents: [Document.fromText('...')] });
```

0.1.2 registered neither, so GoodMem could not be used with Genkit's RAG paths
at all — it was tools only.

Each document carries the memory's own metadata plus its GoodMem provenance:

| Key | Value |
| --- | --- |
| `goodmem_chunk_id`, `goodmem_memory_id`, `goodmem_space_id` | where this hit came from |
| `goodmem_score` | higher is better |
| `goodmem_raw_score` | exactly what the server sent |
| `goodmem_score_kind` | `'vector'` or `'reranker'` — not the same scale |
| `goodmem_partial` | true when the server reported a problem |
| `goodmem_statuses` | present when partial |
| `goodmem_abstract_reply` | present when `llmId` is set and the LLM answered; the same answer on every document of one retrieval |

The `goodmem_*` prefix is reserved. These keys always describe *this*
retrieval: a memory's own metadata cannot override them, and the indexer
drops them before storing a document, so a retrieved document indexed again
(a copy, a cache, a migration) reports the copy's ids, not the original's.

### Tools

By default the model sees exactly two:

| Tool | What the model may pass |
| --- | --- |
| `goodmem/search` | `query`, `topK` |
| `goodmem/remember` | `text` |

Everything else is opt-in, because a model does not need to administer a
memory server in order to use one. 0.1.2 exposed eleven tools, including
`goodmem/delete_space`.

| Option | Adds |
| --- | --- |
| `uploadDir: '<dir>'` | `goodmem/upload_file`, confined to that directory |
| `allowAdminTools: true` | space/embedder management, `get_memory`, `list_memories` |
| `allowDelete: true` | `delete_memory`, `delete_space` |
| `allowWrite: false` | removes `goodmem/remember` |

The id each opt-in tool takes — `memoryId`, `spaceId`, `embedderId` — is
declared to the model as a UUID (`format: uuid`).

## Ids are UUIDs

Every GoodMem id this plugin sends — a memory, space, embedder, reranker or LLM id,
whether a model, your code or the plugin configuration supplied it — must be a
canonical UUID; anything else is refused with a `GoodMemError` naming the
field, before any request is made, because ids are part of the URL path
(`/v1/memories/{id}`) and a value such as `../spaces/<id>` could otherwise
address a different resource than the one named. An upper-case UUID is
accepted and sent lower-cased. The check sits at the call that sends the id,
so the tools, the retriever and indexer, and `GoodMemConnection`'s own methods
are all covered; the model-visible schema only tells the model.

## When retrieval goes wrong

`partial` means exactly one thing: **the server reported a real problem during
this retrieval.** It is independent of whether hits came back. A degraded
search still returns whatever arrived, flagged; when nothing usable arrives
the result is empty, `partial` is set, a `warning` is included, and the plugin
logs at WARNING with the server's own reason. A failed search is never
presented as an empty one.

A request that never yields a single event — a dead connection, a refused
handshake — **throws**, rather than being reported as a search that found
nothing.

### Scores

GoodMem produces two kinds of score and they are not comparable. **Vector**
scores are negative distances, so `goodmem_score` is the flipped value with
`goodmem_raw_score` kept beside it. **Reranker** scores are already
higher-is-better, on a **provider-dependent** scale — measured live on the
same five documents, Voyage `rerank-2.5` returned `0.27..0.93` and Jina
`jina-reranker-v3` returned `-0.14..0.43`.

So there is **no default threshold**. `minScore` applies only to reranker
scores, and warns naming the observed range if it removes everything:

```ts
goodmem({
  baseUrl: process.env.GOODMEM_BASE_URL!,
  apiKey: process.env.GOODMEM_API_KEY!,
  spaceIds: ['<space-uuid>'],
  rerankerId: '<reranker-uuid>',
  minScore: 0.5,
});
```

Whether hits were reranked is read from the response, not from the
configuration. When the reranker fails the server reports `RERANKING_FAILED`
(or `NOT_FOUND` for the reranker) and still returns the vector-scored hits it
had; those come back as `vector` scores, flipped, **not** filtered by
`minScore`, with `partial` set and the statuses attached.

## Grounded answers from an LLM

GoodMem can run an LLM over the chunks a retrieval found and send back an
answer grounded in them. It is **off** unless you set `llmId` — the id of an
LLM registered in GoodMem, as a UUID. Like `rerankerId` it is set by you, the
developer, never by the model: `goodmem/search` still takes only `query` and
`topK`.

```ts
goodmem({
  baseUrl: process.env.GOODMEM_BASE_URL!,
  apiKey: process.env.GOODMEM_API_KEY!,
  spaceIds: ['<space-uuid>'],
  llmId: '<llm-uuid>',
  rerankerId: '<reranker-uuid>', // optional; the two combine
});
```

It is sent as `llm_id` in the retrieval's post-processor config, next to
`reranker_id`. A value that is not a UUID — an LLM's display name, say — throws
a `GoodMemError` when the plugin is created, and again at any call, before a
request is made.

Where the answer appears:

| Path | Field |
| --- | --- |
| `goodmem/search` tool | `abstractReply` — a string, beside `results` |
| `goodmem/memories` retriever | `goodmem_abstract_reply` in every document's metadata |
| `GoodMemConnection.retrieve()` | `abstractReply` on the returned `RetrievalOutcome` |

A Genkit retriever returns documents and nothing else, so the answer for the
retrieval rides on each document, the same way `goodmem_partial` does; like
every `goodmem_*` key, the indexer drops it. A retrieval that finds no
documents has nothing to carry it: use `goodmem/search` or
`GoodMemConnection.retrieve()` when you need the answer on its own.

**When the LLM fails**, the server reports `SUMMARIZATION_FAILED` (with
`NOT_FOUND` too when no LLM has that id; a provider error such as HTTP 429
arrives in the message). That is a problem status like any other: the hits are
kept, `partial` is set, the statuses and a `warning` are included, there is no
`abstractReply`, and nothing is raised.

An LLM does not rerank. Scores are labelled exactly as they would be without
one — `vector` unless a reranker was applied — and `minScore` is unaffected.

## Metadata filters

Filters are expressions evaluated server-side, not SQL. They are set by the
developer, never by the model:

```ts
import { filters, goodmem } from 'genkitx-goodmem';

goodmem({
  baseUrl: process.env.GOODMEM_BASE_URL!,
  apiKey: process.env.GOODMEM_API_KEY!,
  spaceIds: ['<space-uuid>'],
  metadataFilter: { tenant: 'acme', active: true },
});

const expression = filters.allOf(
  filters.equals('tenant', 'acme'),
  filters.compare('year', '>=', 2026),
);
```

The helper applies the escaping the server accepts (`'` → `\'`, `\` → `\\`;
SQL-style `''` doubling is rejected with HTTP 400), refuses control
characters, restricts field names, and casts each value to the type GoodMem
stored — a boolean compared as `TEXT` is accepted with HTTP 200 and matches
nothing.

## Uploads

Uploads are **off** unless you set `uploadDir`. When set, every path is
resolved (symlinks included) and refused if it lands outside that directory,
so a model-supplied path cannot read arbitrary host files.

## Changes in 0.3.0

A new, opt-in setting; nothing changes unless it is set. Measured with
`tests/goodmem_llm_test.ts` and `tests/goodmem_ids_test.ts` against 0.2.1 and
0.3.0, and live against a GoodMem server with an OpenRouter `qwen/qwen3-8b`
LLM.

| Was | Now |
| --- | --- |
| No way to ask GoodMem for an LLM answer: `llmId` was not a setting, so passing it did nothing — no `llm_id` reached the server and no answer came back | `llmId` is sent as `llm_id` in the post-processor config; the answer is `abstractReply` in `goodmem/search` output and `goodmem_abstract_reply` on retrieved documents |
| `llmId: 'qwen3-8b'` (a name, not an id) was accepted silently | Refused with `llmId must be a UUID` at plugin creation and at every call; nothing sent |
| A failing LLM could not be observed | `SUMMARIZATION_FAILED` / `NOT_FOUND` surface as `partial` + `statuses` + `warning`, with the hits kept |


Measured by driving the real SDK against a local server that records every
request it receives (`tests/goodmem_ids_test.ts` and
`tests/goodmem_rerank_test.ts`, run against 0.2.0 and 0.2.1). `<U>` is a
space id.

| Was | Now |
| --- | --- |
| `goodmem/delete_memory` with `memoryId: "../spaces/<U>"`, called by a model through `ai.generate`, sent `DELETE /v1/memories/..%2Fspaces%2F<U>` and the tool answered `success: true` | Refused: `memoryId must be a UUID`; the server receives nothing |
| `memoryId: ".."` sent `DELETE /v1/` and `"."` sent `DELETE /v1/memories/` — the SDK's `encodeURIComponent` turns `/` into `%2F` but leaves a bare dot segment for the URL parser to resolve | Refused; nothing sent |
| `list_memories` with `spaceId: ".."` sent `GET /v1/memories` | Refused; nothing sent |
| Any string reached the URL, percent-encoded: `" <U>"`, `"<U>?x=1"`, `"<U>#frag"`, `"%2e%2e/spaces/<U>"`, `"urn:uuid:<U>"` — across `get_memory`, `delete_memory`, `get_space`, `update_space`, `delete_space`, `list_memories` and the matching `GoodMemConnection` methods | Only a canonical UUID reaches a request |
| A non-UUID in `spaceIds` was sent in every retrieval and write body; `create_space` sent any `embedderId` | Refused when the plugin is created, and again at every call |
| `rerankerId: ""` was silently read as "no reranker" | Refused; leave it unset instead |
| Tool id arguments were declared as a bare `string` | Declared `format: uuid` |
| With any `rerankerId` set, **every** retrieval — `ai.retrieve`, `goodmem/search`, `minScore` — failed with HTTP 400 `Unrecognized field "rerankerId" (class com.goodmem.rest.dto.RetrieveMemoryRequest)`: the SDK translates `rerankerId` only in its `(message, options)` form | Sent as `postProcessor: {name: "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory", config: {reranker_id}}`; per-space filters stay in `spaceKeys` |
| With a reranker configured, a failed rerank (`RERANKING_FAILED`, reranker `NOT_FOUND`) returned the server's vector fallback hits labelled `reranker` and unflipped (`-0.5846`), and any `minScore` discarded all of them | Labelled `vector`, flipped (`0.5846`), kept, `partial` with the statuses |
| Memory metadata overrode `goodmem_*` provenance, and the indexer stored it: a re-indexed copy reported the original's `goodmem_memory_id` (so `delete_memory` with it deleted the original), and a stored `goodmem_partial: false` hid a real `RERANKING_FAILED` | Provenance written last and `goodmem_*` stripped from memory metadata and by the indexer |

## Changes in 0.2.0

Reproduced against the published 0.1.2 package, live against GoodMem v1.0.320.

| Was | Now |
| --- | --- |
| Hand-written `fetch` client | Official `@pairsystems/goodmem` SDK |
| `get_memory({includeContent:true})` called `.json()` on `text/plain` and **failed for every memory**, returning `success: true` with a `contentError` string | Decoded by content type: text as text, anything else base64. A failed fetch raises |
| A space with a failing embedder returned `success: true, totalResults: 0` — `EMBEDDER_FAILED` was dropped | `partial` + `statuses` + a warning carrying the server's reason |
| No retriever and no indexer — unusable with Genkit RAG | `goodmem/memories` registered as both |
| `publicRead` was a tool argument; the server answers `400 Unrecognized field "publicRead"` | Not offered anywhere |
| Eleven tools including `delete_space` | `goodmem/search` + `goodmem/remember`; the rest opt-in |
| `filePath` was an unrestricted tool argument; it read `/etc/hostname` and uploaded it | Confined to `uploadDir`; `..` and symlink escapes refused |
| Empty search took **23.4 s** — `waitForIndexing` on by default | **under a second**; the read path never polls |
| **No request carried a timeout** — no `AbortController` anywhere | `timeoutMs`, default 30s |
| Chunks and memories joined by positional `memoryIndex` | Joined by UUID, de-duplicated by chunk id |
| Raw negative scores | oriented score + raw + kind |
| Reusing a space name accepted any embedder | Reuse requires a match; a mismatch names both |
| `nextToken` appeared nowhere — listings returned one page | Paginated, bounded by `maxListItems` |
| 34 tests over hand-built `Response` objects; no CI | 40 offline + 14 live; CI on Node 20 and 22 |

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/goodmem_test.ts` | 43 | nothing — the real SDK over a mocked `fetch`, fed NDJSON captured from a live server |
| `tests/goodmem_ids_test.ts` | 63 | nothing — the real SDK over real HTTP to a local server that records every request; every id-taking entry point, fourteen malformed ids each |
| `tests/goodmem_rerank_test.ts` | 17 | nothing — the same, with a server that rejects undeclared retrieve fields; the reranker request, failed-rerank fallback hits, `goodmem_*` provenance |
| `tests/goodmem_llm_test.ts` | 19 | nothing — the same, fed NDJSON captured live with an LLM configured, a nonexistent LLM, and a provider 429; the `llm_id` request, the abstract reply, `SUMMARIZATION_FAILED` |
| `tests/goodmem_live_test.ts` | 20 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them |

```bash
npm install
npm test          # offline
npm run test:live # live; GOODMEM_TEST_EMBEDDER_ID pins an embedder, GOODMEM_TEST_RERANKER_ID and GOODMEM_TEST_LLM_ID enable the reranker and LLM tests
npx tsc --noEmit  # types, as CI runs them
```

The live suite creates one space per run and asserts, against a fresh
listing, that it is gone afterwards.

## License

Apache-2.0.
