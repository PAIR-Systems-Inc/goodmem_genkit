/**
 * Reranking and provenance, over real HTTP.
 *
 * These drive the real `@pairsystems/goodmem` SDK against a local server that
 * records every request and -- like GoodMem -- rejects a retrieve body that
 * carries a field `RetrieveMemoryRequest` does not declare, with the server's
 * own 400 wording. 0.2.1 sent `rerankerId` as a top-level field, which the SDK
 * only translates in its `(message, options)` form; the object form goes out
 * verbatim, so every retrieval with a reranker configured failed.
 *
 * Three behaviours are pinned here:
 *
 *  - a configured reranker reaches the server as a `postProcessor`;
 *  - whether hits were reranked is read from the response, not the
 *    configuration: under `RERANKING_FAILED` (or a reranker `NOT_FOUND`) the
 *    server falls back to vector hits, which are oriented as vector scores
 *    and are never dropped by a reranker `minScore` (Q4a);
 *  - `goodmem_*` document metadata always describes this retrieval: stored
 *    memory metadata cannot override it, and the indexer does not store it.
 *
 * They import only what the plugin exported before the fix, so the same file
 * runs against the old source and shows what it did.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { join } from 'node:path';
import { after, afterEach, before, beforeEach, describe, it } from 'node:test';

import { Document, genkit } from 'genkit';

import { GoodMemConnection, goodmem } from '../src/index.js';

const FIXTURES = join(import.meta.dirname ?? __dirname, 'goodmem_fixtures');
const fixture = (name: string) => readFileSync(join(FIXTURES, name), 'utf8');

const KEY = 'gm_offline_test_key';
const SPACE_ID = '01a0d44b-746f-775b-b91e-bc73d4058e27';
const RERANKER_ID = '019cfd1c-c033-7517-b7de-f73941a0464c';
/** The memory every captured fixture returns. */
const ORIGINAL_ID = '01a0d44b-748d-72eb-b54e-c3ea2d956927';
const ORIGINAL_CHUNK = '01a0d44b-7bb4-75a3-b7a1-c7b8cda5859d';
/** The id the local server gives the copy the indexer creates. */
const COPY_ID = '01a0d44b-c0b1-7000-8000-00000000c0b1';
const COPY_SPACE = '01a0d44b-96ae-7081-bc16-5644e701222a';
const COPY_CHUNK = '01a0d44b-c0b1-7000-8000-0000000c4c01';

const POST_PROCESSOR = 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory';
const FIXTURE_METADATA = '"metadata":{"year":2026.0,"active":true,"tenant":"acme"}';

/**
 * The fields GoodMem's `RetrieveMemoryRequest` declares -- the SDK's own
 * `RetrieveMemoryRequest` interface in @pairsystems/goodmem 0.1.7. The server
 * rejects anything else.
 */
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
  if (req.method === 'POST' && path === '/v1/memories') {
    const body = JSON.parse(req.body);
    return json(200, {
      memoryId: COPY_ID,
      spaceId: body.spaceId,
      contentType: body.contentType,
      processingStatus: 'PENDING',
      metadata: body.metadata ?? {},
    });
  }
  if (req.method === 'DELETE') {
    res.writeHead(204);
    return res.end();
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

// ---- streams ----------------------------------------------------------------

function line(event: unknown): string {
  return JSON.stringify(event) + '\n';
}

/**
 * A successfully reranked stream: two chunks of the captured memory with
 * reranker scores. The memory definition is the captured one; the scores are
 * Voyage `rerank-2.5` values measured live (README, "Scores").
 */
function rerankedStream(scores: number[] = [0.93, 0.27]): string {
  const ok = fixture('retrieve_ok.ndjson').trim().split('\n').map((l) => JSON.parse(l));
  const [begin, memory, item, end] = ok;
  const items = scores.map((score, i) => {
    const copy = structuredClone(item);
    copy.retrievedItem.chunk.chunk.chunkId = ORIGINAL_CHUNK.slice(0, -1) + String(i);
    copy.retrievedItem.chunk.relevanceScore = score;
    return copy;
  });
  return [begin, memory, ...items, end].map(line).join('');
}

/** The captured degraded stream: NOT_FOUND + RERANKING_FAILED, one vector hit. */
const DEGRADED = () => fixture('retrieve_degraded_hits.ndjson');

/** The degraded stream with only the reranker NOT_FOUND status, no RERANKING_FAILED. */
const NOT_FOUND_ONLY = () =>
  DEGRADED()
    .split('\n')
    .filter((l) => !l.includes('"RERANKING_FAILED"'))
    .join('\n');

// ---- (1) the reranker reaches the server --------------------------------------

describe('a configured reranker', () => {
  it('is sent as a postProcessor, never as a rerankerId field', async () => {
    stream = rerankedStream();
    await connection({ rerankerId: RERANKER_ID }).retrieve('canary', 3);
    assert.deepEqual(retrieveBodies(), [
      {
        message: 'canary',
        spaceKeys: [{ spaceId: SPACE_ID }],
        requestedSize: 3,
        fetchMemory: true,
        postProcessor: { name: POST_PROCESSOR, config: { reranker_id: RERANKER_ID } },
      },
    ]);
  });

  it('keeps the per-space metadata filter beside the post-processor', async () => {
    stream = rerankedStream();
    await connection({ rerankerId: RERANKER_ID, metadataFilter: { tenant: 'acme' } }).retrieve('q', 1);
    const [body] = retrieveBodies();
    assert.deepEqual(body.spaceKeys, [
      { spaceId: SPACE_ID, filter: "CAST(val('$.tenant') AS TEXT) = 'acme'" },
    ]);
    assert.deepEqual(body.postProcessor, { name: POST_PROCESSOR, config: { reranker_id: RERANKER_ID } });
    assert.ok(!('rerankerId' in body));
  });

  it('sends no postProcessor without a reranker', async () => {
    await connection().retrieve('q', 1);
    const [body] = retrieveBodies();
    assert.ok(!('postProcessor' in body) && !('rerankerId' in body));
  });

  it('works through ai.retrieve, goodmem/search and minScore', async () => {
    stream = rerankedStream();
    const ai = plugin({ rerankerId: RERANKER_ID, minScore: 0.5 });
    const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: 'canary' });
    assert.equal(docs.length, 1, 'minScore 0.5 keeps the 0.93 hit and drops the 0.27 one');
    assert.equal(docs[0].metadata?.goodmem_score, 0.93);
    assert.equal(docs[0].metadata?.goodmem_score_kind, 'reranker');

    const out = await callTool(ai, 'search', { query: 'canary', topK: 5 });
    assert.equal(out.partial, false);
    assert.deepEqual(out.results.map((r: any) => r.score), [0.93]);
    assert.equal(retrieveBodies().length, 2);
  });

  it('labels a successful rerank as reranker scores, not flipped', async () => {
    stream = rerankedStream([0.93, 0.27]);
    const outcome = await connection({ rerankerId: RERANKER_ID }).retrieve('canary', 3);
    assert.equal(outcome.partial, false);
    assert.deepEqual(
      outcome.hits.map((h) => [h.scoreKind, h.rawScore, h.score]),
      [['reranker', 0.93, 0.93], ['reranker', 0.27, 0.27]]
    );
  });

  it('that does not exist is reported in statuses, not raised', async () => {
    stream = DEGRADED();
    const out = await callTool(plugin({ rerankerId: RERANKER_ID }), 'search', { query: 'canary', topK: 5 });
    assert.equal(out.success, true);
    assert.equal(out.partial, true);
    assert.deepEqual(out.statuses.map((s: any) => s.code), ['NOT_FOUND', 'RERANKING_FAILED']);
    assert.match(out.warning, /Reranker not found/);
    assert.equal(out.totalResults, 1, 'the fallback hit was discarded');
  });
});

