/**
 * Offline tests for genkitx-goodmem.
 *
 * These drive the *real* GoodMem SDK over a mocked `fetch`, fed with NDJSON
 * and JSON captured from a live GoodMem server (v1.0.320). 0.1.2's suite
 * hand-built Response objects whose `.json()` always worked, which is exactly
 * why its 34 passing tests were green against a content path that failed on
 * every memory.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, it } from 'node:test';

import { genkit } from 'genkit';

import * as filters from '../src/filters.js';
import { GoodMemFilterError } from '../src/filters.js';
import {
  MALFORMED_STREAM_CODE,
  UNKNOWN_CODE,
  classifyStatus,
  orientScore,
} from '../src/results.js';
import { GoodMemUploadError, resolveUploadPath } from '../src/uploads.js';
import { requireUuid } from '../src/ids.js';
import { decodeContent, GoodMemError, goodmem } from '../src/index.js';

const FIXTURES = join(import.meta.dirname ?? __dirname, 'goodmem_fixtures');
const fixture = (name: string) => readFileSync(join(FIXTURES, name));

const BASE = 'https://goodmem.test';
const KEY = 'gm_offline_test_key';
const SPACE_ID = '01a0d44b-746f-775b-b91e-bc73d4058e27';

let originalFetch: typeof globalThis.fetch;
beforeEach(() => {
  originalFetch = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** Serve captured server bytes for the retrieve endpoint. */
function serveRetrieve(payload: Buffer, capture?: { body?: any }) {
  globalThis.fetch = (async (url: any, init: any) => {
    const href = String(url);
    if (href.includes(':retrieve')) {
      if (capture && init?.body) capture.body = JSON.parse(String(init.body));
      return new Response(payload, {
        status: 200,
        headers: { 'content-type': 'application/x-ndjson' },
      });
    }
    return new Response('{}', { status: 404, headers: { 'content-type': 'application/json' } });
  }) as any;
}

function makeAi(overrides: Record<string, unknown> = {}) {
  return genkit({
    plugins: [
      goodmem({ baseUrl: BASE, apiKey: KEY, spaceIds: [SPACE_ID], ...overrides } as any),
    ],
  });
}

async function callTool(ai: any, name: string, input: any) {
  const action = await ai.registry.lookupAction(`/tool/goodmem/${name}`);
  const result = await action(input);
  return result.result ?? result;
}

describe('fixtures', () => {
  it('are real server bytes', () => {
    const stream = fixture('retrieve_ok.ndjson').toString();
    const events = stream.trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
    assert.ok(events.some((e) => e.resultSetBoundary));
    assert.ok(events.some((e) => e.retrievedItem));
    assert.match(stream, /[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-/i);
  });

  it('carry no credential', () => {
    for (const name of ['retrieve_ok.ndjson', 'spaces_page1.json', 'memory_get.json']) {
      assert.doesNotMatch(fixture(name).toString(), /gm_[a-z0-9]{20,}/);
    }
  });
});

describe('retrieval status contract', () => {
  it('Q4a: a degraded retrieval still returns its hits', async () => {
    serveRetrieve(fixture('retrieve_degraded_hits.ndjson'));
    const out = await callTool(makeAi(), 'search', { query: 'canary', topK: 5 });
    assert.ok(out.totalResults > 0, 'hits were discarded');
    assert.equal(out.partial, true);
    const codes = out.statuses.map((s: any) => s.code);
    assert.ok(codes.includes('NOT_FOUND') || codes.includes('RERANKING_FAILED'));
    assert.ok(out.warning);
  });

  it('Q4b: a degraded retrieval with no hits is flagged, not empty-looking', async () => {
    serveRetrieve(fixture('retrieve_degraded_empty.ndjson'));
    const out = await callTool(makeAi(), 'search', { query: 'nothing', topK: 5 });
    assert.equal(out.totalResults, 0);
    assert.equal(out.partial, true);
    assert.ok(out.statuses.length > 0);
  });

  it('Q1: the two informational codes are noise, by code alone', () => {
    assert.equal(classifyStatus('FEATURE_DISABLED', 'x').informational, true);
    assert.equal(classifyStatus('LLM_CAPABILITY_INFERRED', 'x').informational, true);
    assert.equal(classifyStatus('EMBEDDER_FAILED', 'x').informational, false);
  });

  it('Q3: an unrecognised code becomes UNKNOWN and keeps the original', () => {
    const { status, informational } = classifyStatus('SOME_FUTURE_CODE', 'new');
    assert.equal(status.code, UNKNOWN_CODE);
    assert.equal(status.message, 'new');
    assert.equal(status.details?.serverCode, 'SOME_FUTURE_CODE');
    assert.equal(informational, false);
  });

  it('a clean stream is not partial', async () => {
    serveRetrieve(fixture('retrieve_ok.ndjson'));
    const out = await callTool(makeAi(), 'search', { query: 'canary', topK: 5 });
    assert.equal(out.partial, false);
    assert.deepEqual(out.statuses, []);
    assert.ok(out.totalResults >= 1);
  });

  it('a truncated stream keeps what arrived and says so', async () => {
    const whole = fixture('retrieve_ok.ndjson');
    serveRetrieve(whole.subarray(0, Math.floor(whole.length * 0.6)));
    const out = await callTool(makeAi(), 'search', { query: 'canary', topK: 5 });
    assert.equal(out.partial, true);
    assert.ok(out.statuses.some((s: any) => s.code === MALFORMED_STREAM_CODE));
  });
});

describe('scores', () => {
  it('flips vector scores to higher-is-better', () => {
    assert.equal(orientScore(-0.51, false), 0.51);
  });

  it('never flips a reranker score', () => {
    assert.equal(orientScore(0.93, true), 0.93);
    assert.equal(orientScore(-0.14, true), -0.14);
  });

  it('keeps the raw value beside the oriented one', async () => {
    serveRetrieve(fixture('retrieve_ok.ndjson'));
    const out = await callTool(makeAi(), 'search', { query: 'canary', topK: 1 });
    const hit = out.results[0];
    assert.ok(hit.rawScore < 0);
    assert.equal(hit.score, -hit.rawScore);
    assert.equal(hit.scoreKind, 'vector');
  });

  it('sends no relevance threshold by default', async () => {
    const capture: { body?: any } = {};
    serveRetrieve(fixture('retrieve_ok.ndjson'), capture);
    await callTool(makeAi(), 'search', { query: 'canary', topK: 1 });
    assert.doesNotMatch(JSON.stringify(capture.body), /relevanceThreshold/);
  });
});

describe('native retriever and indexer', () => {
  it('registers a retriever Genkit RAG can use', async () => {
    serveRetrieve(fixture('retrieve_ok.ndjson'));
    const ai = makeAi();
    const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: 'canary', options: { k: 2 } });
    assert.ok(docs.length >= 1);
    const md = docs[0].metadata ?? {};
    for (const key of ['goodmem_chunk_id', 'goodmem_memory_id', 'goodmem_score', 'goodmem_score_kind']) {
      assert.ok(key in md, `${key} missing from document metadata`);
    }
  });

  it('flags degraded documents', async () => {
    serveRetrieve(fixture('retrieve_degraded_hits.ndjson'));
    const ai = makeAi();
    const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: 'canary' });
    assert.equal(docs[0].metadata?.goodmem_partial, true);
    assert.ok(docs[0].metadata?.goodmem_statuses);
  });

  it('registers an indexer', async () => {
    const ai = makeAi();
    const action = await ai.registry.lookupAction('/indexer/goodmem/memories');
    assert.ok(action, 'no indexer registered');
  });
});

describe('tool surface', () => {
  it('exposes only a search and a write by default', async () => {
    const ai = makeAi();
    const actions = await ai.registry.listActions();
    const names = [
      ...new Set(
        Object.keys(actions)
          .filter((k) => k.startsWith('/tool/goodmem/'))
          .map((k) => k.slice('/tool/goodmem/'.length))
      ),
    ];
    assert.deepEqual(names.sort(), ['remember', 'search']);
  });

  it('keeps space management and deletion opt-in', async () => {
    const ai = makeAi();
    const actions = await ai.registry.listActions();
    const names = Object.keys(actions).join(' ');
    for (const banned of ['delete_space', 'delete_memory', 'update_space', 'create_space']) {
      assert.ok(!names.includes(banned), `${banned} is exposed by default`);
    }
  });

  it('adds management under allowAdminTools, without deletion', async () => {
    const ai = makeAi({ allowAdminTools: true });
    const names = Object.keys(await ai.registry.listActions()).join(' ');
    assert.ok(names.includes('create_space'));
    assert.ok(!names.includes('delete_space'));
  });

  it('adds deletion only under allowDelete', async () => {
    const ai = makeAi({ allowDelete: true });
    const names = Object.keys(await ai.registry.listActions()).join(' ');
    assert.ok(names.includes('delete_space') && names.includes('delete_memory'));
  });

  it('registers no upload tool without an uploadDir', async () => {
    const ai = makeAi();
    const names = Object.keys(await ai.registry.listActions()).join(' ');
    assert.ok(!names.includes('upload_file'));
  });

  it('the model never chooses a space', async () => {
    serveRetrieve(fixture('retrieve_ok.ndjson'));
    const ai = makeAi();
    const action: any = await ai.registry.lookupAction('/tool/goodmem/search');
    const keys = Object.keys(action.__action.inputSchema?.shape ?? {});
    assert.deepEqual(keys.sort(), ['query', 'topK']);
  });

  it('refuses to start without spaceIds', () => {
    assert.throws(
      () => goodmem({ baseUrl: BASE, apiKey: KEY, spaceIds: [] } as any),
      /spaceIds/
    );
  });
});

describe('publicRead is gone', () => {
  it('is absent from the plugin source', () => {
    const src = readFileSync(join(import.meta.dirname ?? __dirname, '..', 'src', 'index.ts'), 'utf8');
    const code = src
      .split('\n')
      .filter((l) => !l.trim().startsWith('*') && !l.trim().startsWith('//'))
      .join('\n');
    assert.doesNotMatch(code, /publicRead/);
  });

  it('update_space accepts only name and labels', async () => {
    const ai = makeAi({ allowAdminTools: true });
    const action: any = await ai.registry.lookupAction('/tool/goodmem/update_space');
    const keys = Object.keys(action.__action.inputSchema?.shape ?? {});
    assert.ok(!keys.includes('publicRead'));
    assert.deepEqual(keys.sort(), ['labels', 'name', 'replaceLabels', 'spaceId']);
  });
});

describe('content decoding', () => {
  it('returns text as text', () => {
    const { content, encoding } = decodeContent(new TextEncoder().encode('hello'), 'text/plain');
    assert.equal(content, 'hello');
    assert.equal(encoding, 'text');
  });

  it('returns binary as base64, never mangled', () => {
    const pdf = Uint8Array.from([0x25, 0x50, 0x44, 0x46, 0x00, 0xff]);
    const { content, encoding } = decodeContent(pdf, 'application/pdf');
    assert.equal(encoding, 'base64');
    assert.deepEqual(Uint8Array.from(Buffer.from(content, 'base64')), pdf);
  });

  it('falls back to base64 when text does not decode', () => {
    const bad = Uint8Array.from([0xff, 0xfe, 0xfd]);
    assert.equal(decodeContent(bad, 'text/plain').encoding, 'base64');
  });
});

describe('uploads', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'gk-upload-'));
    writeFileSync(join(dir, 'ok.txt'), 'allowed');
  });

  it('refuses an absolute host path', () => {
    assert.throws(() => resolveUploadPath('/etc/hostname', dir), GoodMemUploadError);
  });

  it('refuses a .. escape', () => {
    assert.throws(() => resolveUploadPath('../../etc/hostname', dir), GoodMemUploadError);
  });

  it('refuses a symlink pointing outside', () => {
    symlinkSync('/etc/hostname', join(dir, 'escape.txt'));
    assert.throws(() => resolveUploadPath('escape.txt', dir), GoodMemUploadError);
  });

  it('allows a file inside the directory', () => {
    assert.ok(resolveUploadPath('ok.txt', dir).endsWith('ok.txt'));
  });

  it('is disabled entirely without an uploadDir', () => {
    assert.throws(() => resolveUploadPath('/etc/hostname', undefined), /disabled/);
  });
});

describe('ids', () => {
  it('accepts a canonical UUID and returns it lower-cased', () => {
    assert.equal(requireUuid(SPACE_ID.toUpperCase(), 'spaceId'), SPACE_ID);
  });

  it('refuses anything else with a GoodMemError naming the field', () => {
    for (const bad of [`../spaces/${SPACE_ID}`, `${SPACE_ID} `, `${SPACE_ID}\n`, '', 'space-1', undefined, null, 42]) {
      assert.throws(() => requireUuid(bad, 'memoryId'), (err: any) => {
        assert.ok(err instanceof GoodMemError, 'not a GoodMemError');
        assert.match(err.message, /^memoryId must be a UUID/);
        return true;
      });
    }
  });

  it('bounds how much of a refused value it repeats', () => {
    assert.throws(() => requireUuid('x'.repeat(10_000), 'spaceId'), (err: any) => err.message.length < 400);
  });
});

describe('filters', () => {
  it('escapes an apostrophe with a backslash, not by doubling', () => {
    assert.equal(filters.equals('n', "o'brien"), "CAST(val('$.n') AS TEXT) = 'o\\'brien'");
  });

  it('refuses control characters', () => {
    assert.throws(() => filters.equals('f', 'a\nb'), GoodMemFilterError);
  });

  it('casts a boolean as BOOLEAN, never TEXT', () => {
    assert.equal(filters.equals('a', true), "CAST(val('$.a') AS BOOLEAN) = true");
  });

  it('casts a number as NUMERIC', () => {
    assert.match(filters.equals('year', 2026), /AS NUMERIC/);
  });

  it('refuses an unsafe field name', () => {
    assert.throws(() => filters.equals("a' OR '1", 'x'), GoodMemFilterError);
  });

  it('builds comparisons and sets', () => {
    assert.match(filters.compare('year', '>=', 2000), />= 2000$/);
    assert.match(filters.oneOf('tag', ['a', 'b']), /IN \(/);
  });

  it('refuses mixed types in oneOf', () => {
    assert.throws(() => filters.oneOf('tag', ['a', 1]), GoodMemFilterError);
  });

  it('reaches the request as a space key filter', async () => {
    const capture: { body?: any } = {};
    serveRetrieve(fixture('retrieve_ok.ndjson'), capture);
    await callTool(makeAi({ metadataFilter: { tenant: 'acme' } }), 'search', { query: 'q', topK: 1 });
    assert.equal(capture.body.spaceKeys[0].filter, "CAST(val('$.tenant') AS TEXT) = 'acme'");
  });
});
