/**
 * LLM post-processing, over real HTTP.
 *
 * GoodMem can run an LLM over the chunks a retrieval found and stream back an
 * `abstractReply` -- a grounded answer. 0.2.1 had no way to ask for one:
 * `llmId` was not a setting, so passing it was ignored and nothing reached the
 * server, and the stream parser's `abstractReply` branch never fired.
 *
 * These drive the real `@pairsystems/goodmem` SDK against a local server that
 * records every request and, like GoodMem, rejects a retrieve body carrying a
 * field `RetrieveMemoryRequest` does not declare. The streams are NDJSON
 * captured from a live GoodMem server with the LLM configured, with a
 * nonexistent LLM id, and with an LLM whose provider answered 429.
 *
 * Pinned here:
 *
 *  - `llmId` reaches the server as `llm_id` in the post-processor config, next
 *    to `reranker_id`, and only when it is set;
 *  - the answer is `abstractReply` in `goodmem/search` output and
 *    `goodmem_abstract_reply` on every document the retriever returns;
 *  - an LLM that fails (`SUMMARIZATION_FAILED`, `NOT_FOUND` for the LLM) is a
 *    problem status: the hits are kept, `partial` is set, the statuses are
 *    surfaced, and nothing is raised;
 *  - an LLM does not rerank, so scores are labelled exactly as without one;
 *  - the model still passes only `query` and `topK`.
 *
 * They import only what the plugin exported before the change, so the same
 * file runs against the old source and shows what it did.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { join } from 'node:path';
import { after, afterEach, before, beforeEach, describe, it } from 'node:test';

import { genkit } from 'genkit';

import { GoodMemConnection, goodmem } from '../src/index.js';

const FIXTURES = join(import.meta.dirname ?? __dirname, 'goodmem_fixtures');
const fixture = (name: string) => readFileSync(join(FIXTURES, name), 'utf8');

const KEY = 'gm_offline_test_key';
const SPACE_ID = '01a0d44b-746f-775b-b91e-bc73d4058e27';
const RERANKER_ID = '019cfd1c-c033-7517-b7de-f73941a0464c';
const LLM_ID = '019cfd9f-0963-76f9-b069-4cde19a64ba8';

const POST_PROCESSOR = 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory';

/** The answer in `retrieve_llm_ok.ndjson`, as the live LLM wrote it. */
const ANSWER = JSON.parse(
  fixture('retrieve_llm_ok.ndjson')
    .trim()
    .split('\n')
    .find((l) => l.startsWith('{"abstractReply"'))!
).abstractReply.text as string;
/** The vector score every captured non-reranked fixture carries. */
const VECTOR_RAW = -0.6840693950653076;
/** The reranker score the captured reranked fixtures carry. */
const RERANKER_RAW = 0.8671875;

/** The fields GoodMem's `RetrieveMemoryRequest` declares; anything else is a 400. */
const RETRIEVE_FIELDS = new Set([
  'message', 'context', 'spaceKeys', 'requestedSize', 'outputBudget',
  'fetchMemory', 'fetchMemoryContent', 'hnsw', 'postProcessor', 'logging',
]);

// ---- a local GoodMem that records every request ----------------------------

interface Recorded {
  method: string;
  url: string;
  body: string;
}

let server: Server;
let baseUrl: string;
let requests: Recorded[] = [];
/** The NDJSON the next retrieve is answered with. */
let stream = fixture('retrieve_ok.ndjson');

function route(req: Recorded, res: any) {
  const json = (status: number, body: unknown) => {
    res.writeHead(status, { 'content-type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  const path = req.url.split('?')[0];
  if (req.method === 'POST' && path === '/v1/memories:retrieve') {
    const unknown = Object.keys(JSON.parse(req.body)).find((k) => !RETRIEVE_FIELDS.has(k));
    if (unknown) {
      return json(400, {
        error:
          `Unrecognized field "${unknown}" (class com.goodmem.rest.dto.RetrieveMemoryRequest), ` +
          'not marked as ignorable',
      });
    }
    res.writeHead(200, { 'content-type': 'application/x-ndjson' });
    return res.end(stream);
  }
  return json(404, { error: `no route for ${req.method} ${req.url}` });
}

before(async () => {
  server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const recorded = {
        method: req.method ?? '',
        url: req.url ?? '',
        body: Buffer.concat(chunks).toString('utf8'),
      };
      requests.push(recorded);
      route(recorded, res);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

let warn: typeof console.warn;
let warnings: string[] = [];
beforeEach(() => {
  requests = [];
  stream = fixture('retrieve_ok.ndjson');
  warnings = [];
  warn = console.warn;
  console.warn = (...args: unknown[]) => void warnings.push(args.join(' '));
});
afterEach(() => {
  console.warn = warn;
});

const retrieveBodies = () =>
  requests.filter((r) => r.url === '/v1/memories:retrieve').map((r) => JSON.parse(r.body));

function connection(extra: Record<string, unknown> = {}) {
  return new GoodMemConnection({ baseUrl, apiKey: KEY, spaceIds: [SPACE_ID], ...extra } as any);
}

function plugin(extra: Record<string, unknown> = {}) {
  return genkit({
    plugins: [goodmem({ baseUrl, apiKey: KEY, spaceIds: [SPACE_ID], ...extra } as any)],
  });
}

async function callTool(ai: any, name: string, input: any) {
  const action = await ai.registry.lookupAction(`/tool/goodmem/${name}`);
  const result = await action(input);
  return result.result ?? result;
}

// ---- (1) the request ----------------------------------------------------------

describe('a configured llmId', () => {
  it('is sent as llm_id in the post-processor config', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    await connection({ llmId: LLM_ID }).retrieve('What is the fixture canary?', 3);
    assert.deepEqual(retrieveBodies(), [
      {
        message: 'What is the fixture canary?',
        spaceKeys: [{ spaceId: SPACE_ID }],
        requestedSize: 3,
        fetchMemory: true,
        postProcessor: { name: POST_PROCESSOR, config: { llm_id: LLM_ID } },
      },
    ]);
  });

  it('sits next to reranker_id when both are set', async () => {
    stream = fixture('retrieve_llm_reranked.ndjson');
    await connection({ rerankerId: RERANKER_ID, llmId: LLM_ID }).retrieve('q', 3);
    const [body] = retrieveBodies();
    assert.deepEqual(body.postProcessor, {
      name: POST_PROCESSOR,
      config: { reranker_id: RERANKER_ID, llm_id: LLM_ID },
    });
    assert.ok(!('llmId' in body) && !('llm_id' in body), 'an LLM id was sent outside the post-processor');
  });

  it('is sent lower-cased, as every id is', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    await connection({ llmId: LLM_ID.toUpperCase() }).retrieve('q', 3);
    assert.equal(retrieveBodies()[0].postProcessor.config.llm_id, LLM_ID);
  });

  it('leaves the request unchanged when unset', async () => {
    await connection().retrieve('q', 1);
    await connection({ rerankerId: RERANKER_ID }).retrieve('q', 1);
    const [plain, reranked] = retrieveBodies();
    assert.ok(!('postProcessor' in plain));
    assert.deepEqual(reranked.postProcessor, { name: POST_PROCESSOR, config: { reranker_id: RERANKER_ID } });
  });

  it('reaches the server through goodmem/search and the retriever', async () => {
    const ai = plugin({ llmId: LLM_ID });
    stream = fixture('retrieve_llm_ok.ndjson');
    await callTool(ai, 'search', { query: 'q', topK: 3 });
    stream = fixture('retrieve_llm_ok.ndjson');
    await ai.retrieve({ retriever: 'goodmem/memories', query: 'q' });
    assert.deepEqual(
      retrieveBodies().map((b) => b.postProcessor?.config),
      [{ llm_id: LLM_ID }, { llm_id: LLM_ID }]
    );
  });

  it('that is not a UUID is refused before any request', async () => {
    for (const bad of ['', 'qwen3-8b', `../llms/${LLM_ID}`, `${LLM_ID}?x=1`]) {
      await assert.rejects(
        () => connection({ llmId: bad }).retrieve('q', 1),
        (e: any) => e.name === 'GoodMemError' && /llmId must be a UUID/.test(e.message),
        JSON.stringify(bad)
      );
      assert.throws(() => plugin({ llmId: bad }), /llmId must be a UUID/, JSON.stringify(bad));
    }
    assert.deepEqual(requests, []);
  });

  it('is not something the model can set: goodmem/search still takes query and topK', async () => {
    // A model that records the tools it is shown, then answers.
    const ai = plugin({ llmId: LLM_ID });
    const shown: any[] = [];
    ai.defineModel({ name: 'test/recorder' }, async (request) => {
      shown.push(...(request.tools ?? []));
      return { message: { role: 'model', content: [{ text: 'done' }] }, finishReason: 'stop' };
    });
    await ai.generate({ model: 'test/recorder', prompt: 'recall', tools: ['goodmem/search'] });
    assert.deepEqual(Object.keys(shown[0].inputSchema.properties).sort(), ['query', 'topK']);
    assert.ok(!JSON.stringify(shown).includes(LLM_ID), 'the LLM id was shown to the model');
  });
});

// ---- (2) the answer -------------------------------------------------------------

describe('the abstract reply', () => {
  it('is the outcome abstractReply, with the hits as they were', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const outcome = await connection({ llmId: LLM_ID }).retrieve('q', 3);
    assert.equal(outcome.abstractReply, ANSWER);
    assert.match(outcome.abstractReply!, /ORYX-2290/);
    assert.equal(outcome.partial, false);
    assert.equal(outcome.hits.length, 1);
  });

  it('is abstractReply in goodmem/search output', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const out = await callTool(plugin({ llmId: LLM_ID }), 'search', { query: 'q', topK: 3 });
    assert.equal(out.success, true);
    assert.equal(out.abstractReply, ANSWER);
    assert.equal(out.partial, false);
    assert.equal(out.totalResults, 1);
    assert.ok(!('warning' in out));
  });

  it('is goodmem_abstract_reply on every retrieved document', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const docs = await plugin({ llmId: LLM_ID }).retrieve({ retriever: 'goodmem/memories', query: 'q' });
    assert.equal(docs.length, 1);
    assert.equal(docs[0].metadata?.goodmem_abstract_reply, ANSWER);
    assert.equal(docs[0].metadata?.goodmem_partial, false);
  });

  it('is absent without an LLM', async () => {
    const out = await callTool(plugin(), 'search', { query: 'q', topK: 3 });
    assert.ok(!('abstractReply' in out));
    const docs = await plugin().retrieve({ retriever: 'goodmem/memories', query: 'q' });
    assert.ok(!('goodmem_abstract_reply' in (docs[0].metadata ?? {})));
  });

  it('is dropped by the indexer, like every goodmem_* key', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const ai = plugin({ llmId: LLM_ID });
    const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: 'q' });
    requests = [];
    // The local server has no create route; the body that was sent is what matters.
    await ai.index({ indexer: 'goodmem/memories', documents: docs }).catch(() => undefined);
    const created = requests.filter((r) => r.method === 'POST' && r.url === '/v1/memories');
    assert.equal(created.length, 1);
    assert.ok(!JSON.stringify(JSON.parse(created[0].body)).includes('abstract_reply'));
  });
});

// ---- (3) an LLM that fails --------------------------------------------------------

describe('when the LLM fails', () => {
  it('a nonexistent LLM: hits kept, partial, NOT_FOUND and SUMMARIZATION_FAILED, no raise', async () => {
    stream = fixture('retrieve_llm_not_found.ndjson');
    const out = await callTool(plugin({ llmId: LLM_ID }), 'search', { query: 'q', topK: 3 });
    assert.equal(out.success, true);
    assert.equal(out.partial, true);
    assert.deepEqual(out.statuses.map((s: any) => s.code), ['NOT_FOUND', 'SUMMARIZATION_FAILED']);
    assert.equal(out.totalResults, 1, 'the hits were dropped');
    assert.match(out.results[0].text, /ORYX-2290/);
    assert.match(out.warning, /LLM not found/);
    assert.ok(!('abstractReply' in out));
    assert.ok(warnings.some((w) => w.includes('SUMMARIZATION_FAILED')), 'nothing was logged');
  });

  it("a provider error (429): the server's reason is surfaced, hits kept", async () => {
    stream = fixture('retrieve_llm_rate_limited.ndjson');
    const outcome = await connection({ llmId: LLM_ID }).retrieve('q', 3);
    assert.equal(outcome.partial, true);
    assert.deepEqual(outcome.statuses.map((s) => s.code), ['SUMMARIZATION_FAILED']);
    assert.match(outcome.statuses[0].message, /429/);
    assert.equal(outcome.hits.length, 1);
    assert.equal(outcome.abstractReply, undefined);
  });

  it('the retriever returns the documents, flagged', async () => {
    stream = fixture('retrieve_llm_not_found.ndjson');
    const docs = await plugin({ llmId: LLM_ID }).retrieve({ retriever: 'goodmem/memories', query: 'q' });
    assert.equal(docs.length, 1);
    assert.equal(docs[0].metadata?.goodmem_partial, true);
    assert.deepEqual(
      (docs[0].metadata?.goodmem_statuses as any[]).map((s) => s.code),
      ['NOT_FOUND', 'SUMMARIZATION_FAILED']
    );
    assert.ok(!('goodmem_abstract_reply' in (docs[0].metadata ?? {})));
  });
});

// ---- (4) an LLM does not rerank ------------------------------------------------------

describe('scores with an LLM', () => {
  it('are vector scores without a reranker', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const outcome = await connection({ llmId: LLM_ID }).retrieve('q', 3);
    assert.deepEqual(
      outcome.hits.map((h) => [h.scoreKind, h.rawScore, h.score]),
      [['vector', VECTOR_RAW, -VECTOR_RAW]]
    );
  });

  it('are reranker scores with a working reranker', async () => {
    stream = fixture('retrieve_llm_reranked.ndjson');
    const outcome = await connection({ rerankerId: RERANKER_ID, llmId: LLM_ID }).retrieve('q', 3);
    assert.equal(outcome.partial, false);
    assert.match(outcome.abstractReply!, /ORYX-2290/);
    assert.deepEqual(
      outcome.hits.map((h) => [h.scoreKind, h.rawScore, h.score]),
      [['reranker', RERANKER_RAW, RERANKER_RAW]]
    );
  });

  it("an LLM's NOT_FOUND does not unlabel a real rerank, or trip minScore", async () => {
    stream = fixture('retrieve_llm_not_found_reranked.ndjson');
    const outcome = await connection({ rerankerId: RERANKER_ID, llmId: LLM_ID, minScore: 0.5 }).retrieve('q', 3);
    assert.equal(outcome.partial, true);
    assert.deepEqual(outcome.statuses.map((s) => s.code), ['NOT_FOUND', 'SUMMARIZATION_FAILED']);
    assert.deepEqual(
      outcome.hits.map((h) => [h.scoreKind, h.score]),
      [['reranker', RERANKER_RAW]]
    );
  });

  it('minScore still applies only to reranker scores', async () => {
    stream = fixture('retrieve_llm_ok.ndjson');
    const outcome = await connection({ llmId: LLM_ID, minScore: 0.99 }).retrieve('q', 3);
    assert.equal(outcome.hits.length, 1, 'a vector hit met a reranker threshold');
  });
});
