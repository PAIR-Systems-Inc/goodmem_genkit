/**
 * Live tests for genkitx-goodmem, against a running GoodMem server.
 *
 * These skip entirely unless GOODMEM_API_KEY and GOODMEM_BASE_URL are set,
 * which is also the check that no credential is baked into the package.
 *
 *   GOODMEM_API_KEY=... GOODMEM_BASE_URL=https://localhost:8080 \
 *     GOODMEM_TEST_EMBEDDER_ID=... npm run test:live
 */

import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, it } from 'node:test';

import { genkit } from 'genkit';

import { GoodMemConnection, GoodMemUploadError, goodmem } from '../src/index.js';

const BASE = process.env.GOODMEM_BASE_URL;
const KEY = process.env.GOODMEM_API_KEY;
const EMBEDDER = process.env.GOODMEM_TEST_EMBEDDER_ID;
const FAILING_EMBEDDER = process.env.GOODMEM_TEST_FAILING_EMBEDDER_ID;
const skip = !(BASE && KEY) ? 'GOODMEM_API_KEY and GOODMEM_BASE_URL are not set' : false;

if (!skip && process.env.GOODMEM_VERIFY_SSL !== 'true') {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
}

const RUN = Math.random().toString(36).slice(2, 10);
const H = { 'X-API-Key': KEY ?? '' };

function makeAi(spaceIds: string[], extra: Record<string, unknown> = {}) {
  return genkit({
    plugins: [goodmem({ baseUrl: BASE!, apiKey: KEY!, spaceIds, ...extra } as any)],
  });
}
async function tool(ai: any, name: string, input: any) {
  const action = await ai.registry.lookupAction(`/tool/goodmem/${name}`);
  const result = await action(input);
  return result.result ?? result;
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('live', { skip }, () => {
  let conn: GoodMemConnection;
  let embedderId: string;
  let spaceId: string;
  let canary: string;
  let memoryId: string;

  before(async () => {
    conn = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: ['bootstrap'] });
    embedderId = EMBEDDER ?? (await conn.listEmbedders())[0].embedderId;
    const created = await conn.createSpace(`gk-live-${RUN}`, embedderId);
    spaceId = created.spaceId;

    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId] });
    canary = `ORYX-${RUN.toUpperCase()}`;
    const memory: any = await scoped.createFromText(`The Genkit live canary is ${canary}.`, {
      tenant: 'acme',
    });
    memoryId = String(memory.memoryId);
    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      if ((await scoped.retrieve(canary, 3)).hits.length > 0) break;
      await sleep(2000);
    }
  });

  after(async () => {
    if (memoryId) await fetch(`${BASE}/v1/memories/${memoryId}`, { method: 'DELETE', headers: H });
    if (spaceId) await fetch(`${BASE}/v1/spaces/${spaceId}`, { method: 'DELETE', headers: H });
    // teardown verified against a fresh listing, not the delete's status
    const left = (await conn.listSpaces()).filter((s) => s.spaceId === spaceId);
    assert.deepEqual(left, [], 'the space survived teardown');
  });

  it('an exact identifier round-trips through the tool', async () => {
    const out = await tool(makeAi([spaceId]), 'search', { query: canary, topK: 5 });
    assert.equal(out.partial, false);
    assert.ok(out.results.some((r: any) => r.text.includes(canary)));
    assert.equal(out.results[0].memoryId, memoryId);
  });

  it('hits carry an oriented score, ids and the memory metadata', async () => {
    const out = await tool(makeAi([spaceId]), 'search', { query: canary, topK: 1 });
    const hit = out.results[0];
    assert.ok(hit.rawScore < 0, 'GoodMem vector scores are negative');
    assert.ok(hit.score > 0, 'not flipped to higher-is-better');
    assert.equal(hit.scoreKind, 'vector');
    assert.equal((hit.metadata as any).tenant, 'acme');
    assert.ok(hit.chunkId && hit.memoryId && hit.spaceId);
  });

  it('the native retriever returns Genkit documents', async () => {
    const ai = makeAi([spaceId]);
    const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: canary, options: { k: 3 } });
    assert.ok(docs.length >= 1);
    assert.equal(docs[0].metadata?.goodmem_partial, false);
    assert.ok((docs[0].metadata?.goodmem_score as number) > 0);
  });

  it('the indexer stores a document that becomes retrievable', async () => {
    const ai = makeAi([spaceId]);
    const token = `IBEX-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;
    const { Document } = await import('genkit');
    await ai.index({ indexer: 'goodmem/memories', documents: [Document.fromText(`Indexer canary ${token}.`)] });
    let found = false;
    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline && !found) {
      const out = await tool(ai, 'search', { query: token, topK: 3 });
      found = out.results.some((r: any) => r.text.includes(token));
      if (!found) await sleep(2000);
    }
    assert.ok(found, 'the indexed document never became searchable');
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId] });
    for (const m of await scoped.listMemories(spaceId)) {
      if (m.memoryId !== memoryId) await scoped.deleteMemory(m.memoryId);
    }
  });

  it('a failing embedder is reported, not silently empty', async (t) => {
    if (!FAILING_EMBEDDER) return t.skip('GOODMEM_TEST_FAILING_EMBEDDER_ID is not set');
    const bad = await conn.createSpace(`gk-live-bad-${RUN}`, FAILING_EMBEDDER);
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [bad.spaceId] });
    try {
      await scoped.createFromText('Doomed canary.');
      await sleep(6000);
      const outcome = await scoped.retrieve('doomed canary', 3);
      assert.equal(outcome.partial, true);
      assert.ok(outcome.statuses.length > 0, "the server's status was dropped");
    } finally {
      for (const m of await scoped.listMemories(bad.spaceId)) await scoped.deleteMemory(m.memoryId);
      await conn.deleteSpace(bad.spaceId);
    }
  });

  it('the read path does not poll an empty space', async () => {
    const empty = await conn.createSpace(`gk-live-fast-${RUN}`, embedderId);
    try {
      const started = Date.now();
      const out = await tool(makeAi([empty.spaceId]), 'search', { query: 'nothing at all', topK: 3 });
      const elapsed = (Date.now() - started) / 1000;
      assert.equal(out.totalResults, 0);
      assert.ok(elapsed < 3, `an empty search took ${elapsed.toFixed(1)}s`);
    } finally {
      await conn.deleteSpace(empty.spaceId);
    }
  });

  it('renames a space without publicRead', async () => {
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId] });
    const renamed = `gk-live-${RUN}-renamed`;
    assert.equal((await scoped.updateSpace(spaceId, { name: renamed })).success, true);
    assert.equal((await scoped.getSpace(spaceId)).name, renamed);
    await scoped.updateSpace(spaceId, { name: `gk-live-${RUN}` });
  });

  it('refuses to reuse a space name with another embedder', async () => {
    const others = (await conn.listEmbedders()).filter((e) => e.embedderId !== embedderId);
    if (others.length === 0) return;
    await assert.rejects(
      () => conn.createSpace(`gk-live-${RUN}`, others[0].embedderId),
      /cannot be changed/
    );
  });

  it("carries the server's own message on a rejected create", async () => {
    // A well-formed id that names no embedder, so the server -- not the
    // plugin's UUID check -- is what rejects it.
    let error: any;
    let created: any;
    try {
      created = await conn.createSpace(`gk-live-bad-${RUN}`, '00000000-0000-4000-8000-000000000000');
    } catch (err) {
      error = err;
    }
    if (created) await conn.deleteSpace(created.spaceId);
    assert.ok(error, 'a space was created with an embedder that does not exist');
    assert.ok(error.statusCode >= 400 && error.statusCode < 500, `HTTP ${error.statusCode}`);
    assert.match(String(error.message).toLowerCase(), /embedder/);
  });

  it('refuses a malformed id without sending it', async () => {
    await assert.rejects(() => conn.createSpace(`gk-live-bad-${RUN}`, 'not-a-uuid'), /embedderId must be a UUID/);
    await assert.rejects(() => conn.deleteSpace(`../spaces/${spaceId}`), /spaceId must be a UUID/);
    assert.equal((await conn.getSpace(spaceId)).spaceId, spaceId, 'the space is gone');
  });

  it('decodes text content as text', async () => {
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId] });
    const out: any = await scoped.getMemory(memoryId, true);
    assert.equal(out.contentEncoding, 'text');
    assert.ok(String(out.content).includes(canary));
  });

  it('round-trips binary content as base64, JSON-serialisable', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gk-live-'));
    const pdf = Buffer.from('%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n');
    writeFileSync(join(dir, 's.pdf'), pdf);
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId], uploadDir: dir });
    const memory: any = await scoped.createFromFile('s.pdf');
    try {
      await sleep(3000);
      const out: any = await scoped.getMemory(String(memory.memoryId), true);
      assert.equal(out.contentEncoding, 'base64');
      JSON.stringify(out);
      assert.ok(Buffer.from(out.content, 'base64').equals(pdf));
    } finally {
      await scoped.deleteMemory(String(memory.memoryId));
    }
  });

  it('refuses a host path even with uploads enabled', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gk-live-'));
    const scoped = new GoodMemConnection({ baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId], uploadDir: dir });
    await assert.rejects(() => scoped.createFromFile('/etc/hostname'), GoodMemUploadError);
  });

  it('applies a metadata filter server-side and refuses injection', async () => {
    const matching = new GoodMemConnection({
      baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId], metadataFilter: { tenant: 'acme' },
    });
    assert.ok((await matching.retrieve(canary, 5)).hits.length > 0);

    const injected = new GoodMemConnection({
      baseUrl: BASE!, apiKey: KEY!, spaceIds: [spaceId], metadataFilter: { tenant: "x' OR '1'='1" },
    });
    assert.equal((await injected.retrieve(canary, 5)).hits.length, 0, 'filter injection succeeded');
  });

  it('lists spaces without duplicates', async () => {
    const spaces = await conn.listSpaces();
    assert.equal(new Set(spaces.map((s) => s.spaceId)).size, spaces.length);
    assert.ok(spaces.some((s) => s.name.startsWith(`gk-live-${RUN}`)));
  });
});
