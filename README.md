# genkitx-goodmem

[GoodMem](https://goodmem.ai) plugin for
[Genkit](https://genkit.dev).

GoodMem gives AI agents retrieval-augmented generation (RAG) memory. Store documents in a space and GoodMem chunks, embeds, and indexes them so your agent can pull back the most relevant passages on any question.

This package exposes the GoodMem API as 11 Genkit tools under the `goodmem/` namespace, callable from any Genkit agent or flow.

## Installation

```bash
npm install genkitx-goodmem
# or
pnpm add genkitx-goodmem
# or
yarn add genkitx-goodmem
```

`genkit` is a peer dependency. If you do not already have it installed:

```bash
npm install genkit
```

## Quickstart

```ts
import { genkit } from 'genkit';
import { goodmem } from 'genkitx-goodmem';

const ai = genkit({
  plugins: [
    goodmem({
      baseUrl: process.env.GOODMEM_BASE_URL || 'https://localhost:8080',
      apiKey: process.env.GOODMEM_API_KEY!,
    }),
  ],
});
```

Once the plugin is loaded, 11 tools are registered and available to any Genkit agent or flow.

## Tool naming

All tools sit under the `goodmem/` namespace and use snake_case action names. For example: `goodmem/create_space`, `goodmem/list_embedders`. The `<plugin>/<action>` shape follows Genkit's convention.

## Available tools

| Tool | Description |
|---|---|
| `goodmem/list_embedders` | List embedder models available on the server. |
| `goodmem/list_spaces` | List spaces visible to the API key. |
| `goodmem/get_space` | Fetch a space by UUID. |
| `goodmem/create_space` | Create a space (idempotent by name). |
| `goodmem/update_space` | Rename, retag, or change the public-read flag on a space. |
| `goodmem/delete_space` | Delete a space and every memory in it. |
| `goodmem/create_memory` | Store text or a file as a memory. |
| `goodmem/list_memories` | List memories in a space, with status/sort filters. |
| `goodmem/retrieve_memories` | Semantic retrieval across one or more spaces. |
| `goodmem/get_memory` | Fetch a memory by UUID, optionally with content. |
| `goodmem/delete_memory` | Delete a memory. |

### Retrieval options

`goodmem/retrieve_memories` accepts the following parameters in addition to `query`, `spaceIds`, and `maxResults`:

| Parameter | Type | Description |
|---|---|---|
| `metadataFilter` | string | SQL-style JSONPath filter applied server-side to every space key. Example: `CAST(val('$.category') AS TEXT) = 'feat'` |
| `waitForIndexing` | boolean | Poll for results when none come back on the first call (default `true`). |
| `maxWaitSeconds` | number | Polling budget in seconds (default `10`). |
| `pollInterval` | number | Seconds between polls (default `2`). |
| `rerankerId` | string | Reranker model UUID to refine result ordering. |
| `llmId` | string | LLM UUID that generates a contextual abstract reply. |
| `relevanceThreshold` | number | Minimum score (0-1) for inclusion. Only used with a reranker or LLM. |
| `llmTemperature` | number | Creativity (0-2) for the LLM post-processor. Only used with `llmId`. |
| `chronologicalResort` | boolean | Reorder results by creation time instead of relevance. |
| `includeMemoryDefinition` | boolean | Fetch full memory metadata alongside matched chunks (default `true`). |

### `list_memories` options

| Parameter | Type | Description |
|---|---|---|
| `statusFilter` | `"PENDING" \| "PROCESSING" \| "COMPLETED" \| "FAILED"` | Restrict results to one processing status. |
| `includeContent` | boolean | Include original document content alongside metadata (default `false`). |
| `sortBy` | `"created_at" \| "updated_at"` | Field used to sort the returned memories. |
| `sortOrder` | `"ASCENDING" \| "DESCENDING"` | Sort direction. |

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server. Default in examples: `https://localhost:8080`. |
| `GOODMEM_API_KEY` | API key sent as the `X-API-Key` header. |
| `NODE_TLS_REJECT_UNAUTHORIZED` | Set to `0` for local dev with a self-signed certificate. Node's `fetch` reads it directly. |

## Helper functions

The plugin also exports two helper functions for direct programmatic use. The `goodmem/list_spaces` and `goodmem/list_embedders` tools call into these:

```ts
import { listSpaces, listEmbedders } from 'genkitx-goodmem';

const spaces = await listSpaces({
  baseUrl: 'https://localhost:8080',
  apiKey: process.env.GOODMEM_API_KEY!,
});

const embedders = await listEmbedders({
  baseUrl: 'https://localhost:8080',
  apiKey: process.env.GOODMEM_API_KEY!,
});
```

## Example: full workflow

```ts
import { genkit, z } from 'genkit';
import { goodmem } from 'genkitx-goodmem';

const ai = genkit({
  plugins: [
    goodmem({
      baseUrl: 'https://localhost:8080',
      apiKey: process.env.GOODMEM_API_KEY!,
    }),
  ],
});

const memoryFlow = ai.defineFlow(
  { name: 'memoryFlow', inputSchema: z.string(), outputSchema: z.any() },
  async (query) => {
    const listEmbedders = await ai.registry.lookupAction(
      '/tool/goodmem/list_embedders'
    );
    const { embedders } = await listEmbedders({});
    const embedderId = embedders[0].embedderId;

    const createSpace = await ai.registry.lookupAction(
      '/tool/goodmem/create_space'
    );
    const space = await createSpace({
      name: 'my-knowledge-base',
      embedderId,
    });

    const createMemory = await ai.registry.lookupAction(
      '/tool/goodmem/create_memory'
    );
    await createMemory({
      spaceId: space.spaceId,
      textContent: 'The capital of France is Paris.',
      source: 'manual',
    });

    const retrieve = await ai.registry.lookupAction(
      '/tool/goodmem/retrieve_memories'
    );
    return retrieve({
      query,
      spaceIds: [space.spaceId],
      maxResults: 5,
    });
  }
);
```

## End-to-end demo

[`examples/example_usage.ts`](examples/example_usage.ts) walks through three scenarios: persistent project context, a scribe and analyst pipeline, and metadata-driven retrieval. The answering step uses OpenAI via `@genkit-ai/compat-oai`. Set `OPENAI_API_KEY` before running.

```bash
# from a checkout of this repo:
pnpm install
pnpm run demo
# or to inspect traces in the Genkit Developer UI:
genkit start -- tsx examples/example_usage.ts
```

## Testing

Run the unit test suite (mocked `fetch`, exercises every tool):

```bash
pnpm install
pnpm run test
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
