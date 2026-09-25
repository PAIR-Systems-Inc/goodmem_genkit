/**
 * Every id that reaches a GoodMem URL path is a UUID, checked before any
 * request is made.
 *
 * GoodMem ids are interpolated into URL paths -- `/v1/memories/{id}`,
 * `/v1/spaces/{id}`, `/v1/spaces/{id}/memories`. The Python SDK interpolates
 * them raw and its HTTP client resolves `..`, so a memory id of
 * `../spaces/<id>` deletes a whole space; the GoodMem server was also seen to
 * normalise `%2e%2e/spaces/<id>` into the same traversal. The TypeScript SDK
 * this plugin uses percent-encodes with `encodeURIComponent`, which turns `/`
 * into `%2F` -- but leaves a bare `..` alone, so `..` and `.` still climb the
 * path -- and how the server treats `%2F` is unknown. Neither the client
 * encoding nor the server is relied on: anything that is not a canonical UUID
 * is refused before it is sent.
 *
 * These tests drive the real `@pairsystems/goodmem` SDK over real HTTP
 * against a local server that records every request it receives. For each
 * entry point, every payload must be refused *and* the server must have
 * received nothing; a valid UUID must still reach exactly the intended path.
 *
 * They import only what the plugin exported before the fix, so the same file
 * runs against the old source and shows what it sent.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, beforeEach, describe, it } from 'node:test';

import { Document, genkit } from 'genkit';

import { GoodMemConnection, goodmem } from '../src/index.js';

const FIXTURES = join(import.meta.dirname ?? __dirname, 'goodmem_fixtures');
const fixture = (name: string) => readFileSync(join(FIXTURES, name));

const KEY = 'gm_offline_test_key';

/** The space a traversal aims at. */
const U = '01a0d44b-96ae-7081-bc16-5644e701222a';
/** Ids the plugin is legitimately configured with or asked about. */
const SPACE_ID = '01a0d44b-746f-775b-b91e-bc73d4058e27';
const EMBEDDER_ID = '019cfd1c-c033-7517-b7de-f73941a0464b';

/** The shared payload list, identical across every GoodMem integration. */
const SHARED_PAYLOADS = [
  `../spaces/${U}`,
  `a/../../spaces/${U}`,
  `%2e%2e/spaces/${U}`,
  `..%2Fspaces%2F${U}`,
  `${U}/../../spaces/${U}`,
  '',
  ` ${U}`,
  `${U}?x=1`,
  `${U}#frag`,
];

/**
 * Beyond the shared list. `encodeURIComponent` leaves `.` alone, so a bare
 * `..` or `.` is a dot segment the URL parser resolves even through this SDK;
 * `urn:uuid:` is accepted by JSON-schema's `uuid` format, so it shows the
 * call-boundary check -- not the model-visible schema -- is the real guard.
 */
const EXTRA_PAYLOADS = ['..', '.', `${U}\n`, `urn:uuid:${U}`, `{${U}}`];

const PAYLOADS = [...SHARED_PAYLOADS, ...EXTRA_PAYLOADS];

// ---- a local GoodMem that records every request ---------------------------

interface Recorded {
  method: string;
  /** The request target exactly as it arrived on the wire. */
  url: string;
  body: string;
}

const MEMORY = JSON.parse(fixture('memory_get.json').toString());
const SPACE = JSON.parse(fixture('spaces_page1.json').toString()).spaces[1];

let server: Server;
let baseUrl: string;
let requests: Recorded[] = [];

function route(method: string, target: string, res: any) {
  const path = target.split('?')[0];
  const json = (status: number, body: unknown) => {
    res.writeHead(status, { 'content-type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  // Answer as a server that honoured whatever path it was sent would, so the
  // old code's success is visible rather than masked by a 404.
  if (method === 'DELETE') {
    res.writeHead(204);
    return res.end();
  }
  if (method === 'POST' && path === '/v1/memories:retrieve') {
    res.writeHead(200, { 'content-type': 'application/x-ndjson' });
    return res.end(fixture('retrieve_ok.ndjson'));
  }
  if (method === 'POST' && path === '/v1/memories') return json(200, MEMORY);
  if (method === 'POST' && path === '/v1/spaces') return json(200, { ...SPACE, name: 'new-space' });
  if (method === 'GET' && path === '/v1/spaces') return json(200, { spaces: [] });
  if (method === 'GET' && /^\/v1\/spaces\/[^/]+\/memories$/.test(path)) return json(200, { memories: [] });
  if ((method === 'GET' || method === 'PUT') && /^\/v1\/spaces\/[^/]+$/.test(path)) return json(200, SPACE);
  if (method === 'GET' && /^\/v1\/memories\/[^/]+\/content$/.test(path)) {
    res.writeHead(200, { 'content-type': 'text/plain' });
    return res.end(fixture('memory_content.txt'));
  }
  if (method === 'GET' && /^\/v1\/memories\/[^/]+$/.test(path)) return json(200, MEMORY);
  return json(404, { error: `no route for ${method} ${target}` });
}

before(async () => {
  server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      requests.push({
        method: req.method ?? '',
        url: req.url ?? '',
        body: Buffer.concat(chunks).toString('utf8'),
      });
      route(req.method ?? '', req.url ?? '', res);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  requests = [];
});

const sent = () => requests.map((r) => `${r.method} ${r.url}`);

// ---- entry points -----------------------------------------------------------

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

let uploadDir: string;
before(() => {
  uploadDir = mkdtempSync(join(tmpdir(), 'gk-ids-'));
  writeFileSync(join(uploadDir, 'note.txt'), 'upload canary');
});

interface Entry {
  /** The entry point as a developer or a model sees it. */
  name: string;
  /** The field the id is carried in, which the refusal must name. */
  field: RegExp;
  call: (id: string) => Promise<unknown>;
  /** What a valid id `v` must send, and nothing else. */
  expect: (v: string) => string[];
  /** For a body id: where in the body it must appear. */
  inBody?: boolean;
}

/** Model-facing tools: the id is chosen by a language model. */
const MODEL_TOOLS: Entry[] = [
  {
    name: 'tool goodmem/get_memory',
    field: /memoryId/,
    call: (id) =>
      callTool(plugin({ allowAdminTools: true }), 'get_memory', { memoryId: id, includeContent: true }),
    expect: (v) => [`GET /v1/memories/${v}`, `GET /v1/memories/${v}/content`],
  },
  {
    name: 'tool goodmem/delete_memory',
    field: /memoryId/,
    call: (id) => callTool(plugin({ allowDelete: true }), 'delete_memory', { memoryId: id }),
    expect: (v) => [`DELETE /v1/memories/${v}`],
  },
  {
    name: 'tool goodmem/get_space',
    field: /spaceId/,
    call: (id) => callTool(plugin({ allowAdminTools: true }), 'get_space', { spaceId: id }),
    expect: (v) => [`GET /v1/spaces/${v}`],
  },
  {
    name: 'tool goodmem/update_space',
    field: /spaceId/,
    call: (id) =>
      callTool(plugin({ allowAdminTools: true }), 'update_space', { spaceId: id, name: 'renamed' }),
    expect: (v) => [`PUT /v1/spaces/${v}`],
  },
  {
    name: 'tool goodmem/delete_space',
    field: /spaceId/,
    call: (id) => callTool(plugin({ allowDelete: true }), 'delete_space', { spaceId: id }),
    expect: (v) => [`DELETE /v1/spaces/${v}`],
  },
  {
    name: 'tool goodmem/list_memories',
    field: /spaceId/,
    call: (id) => callTool(plugin({ allowAdminTools: true }), 'list_memories', { spaceId: id }),
    expect: (v) => [`GET /v1/spaces/${v}/memories`],
  },
  {
    name: 'tool goodmem/create_space (embedderId, request body)',
    field: /embedderId/,
    call: (id) =>
      callTool(plugin({ allowAdminTools: true }), 'create_space', { name: 'new-space', embedderId: id }),
    expect: () => ['GET /v1/spaces', 'POST /v1/spaces'],
    inBody: true,
  },
];

/** Developer-facing methods on the exported connection. */
const DEVELOPER_METHODS: Entry[] = [
  {
    name: 'GoodMemConnection.getMemory',
    field: /memoryId/,
    call: (id) => connection().getMemory(id, true),
    expect: (v) => [`GET /v1/memories/${v}`, `GET /v1/memories/${v}/content`],
  },
  {
    name: 'GoodMemConnection.deleteMemory',
    field: /memoryId/,
    call: (id) => connection().deleteMemory(id),
    expect: (v) => [`DELETE /v1/memories/${v}`],
  },
  {
    name: 'GoodMemConnection.getSpace',
    field: /spaceId/,
    call: (id) => connection().getSpace(id),
    expect: (v) => [`GET /v1/spaces/${v}`],
  },
  {
    name: 'GoodMemConnection.updateSpace',
    field: /spaceId/,
    call: (id) => connection().updateSpace(id, { name: 'renamed' }),
    expect: (v) => [`PUT /v1/spaces/${v}`],
  },
  {
    name: 'GoodMemConnection.deleteSpace',
    field: /spaceId/,
    call: (id) => connection().deleteSpace(id),
    expect: (v) => [`DELETE /v1/spaces/${v}`],
  },
  {
    name: 'GoodMemConnection.listMemories(spaceId)',
    field: /spaceId/,
    call: (id) => connection().listMemories(id),
    expect: (v) => [`GET /v1/spaces/${v}/memories`],
  },
  {
    name: 'GoodMemConnection.createSpace (embedderId, request body)',
    field: /embedderId/,
    call: (id) => connection().createSpace('new-space', id),
    expect: () => ['GET /v1/spaces', 'POST /v1/spaces'],
    inBody: true,
  },
];

/** Ids that come from configuration rather than from a call. */
const CONFIGURED: Entry[] = [
  {
    name: 'config spaceIds[0] -> GoodMemConnection.listMemories()',
    field: /spaceIds\[0\]/,
    call: (id) => connection({ spaceIds: [id] }).listMemories(),
    expect: (v) => [`GET /v1/spaces/${v}/memories`],
  },
  {
    name: 'config spaceIds -> GoodMemConnection.retrieve (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => connection({ spaceIds: [id] }).retrieve('q', 1),
    expect: () => ['POST /v1/memories:retrieve'],
    inBody: true,
  },
  {
    name: 'config spaceIds[0] -> GoodMemConnection.createFromText (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => connection({ spaceIds: [id] }).createFromText('canary'),
    expect: () => ['POST /v1/memories'],
    inBody: true,
  },
  {
    name: 'config spaceIds[0] -> GoodMemConnection.createFromFile (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => connection({ spaceIds: [id], uploadDir }).createFromFile('note.txt'),
    expect: () => ['POST /v1/memories'],
    inBody: true,
  },
  {
    name: 'config rerankerId -> GoodMemConnection.retrieve (request body)',
    field: /rerankerId/,
    call: (id) => connection({ rerankerId: id }).retrieve('q', 1),
    expect: () => ['POST /v1/memories:retrieve'],
    inBody: true,
  },
  {
    name: 'plugin spaceIds -> tool goodmem/search (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => callTool(plugin({ spaceIds: [id] }), 'search', { query: 'q', topK: 1 }),
    expect: () => ['POST /v1/memories:retrieve'],
    inBody: true,
  },
  {
    name: 'plugin spaceIds -> retriever goodmem/memories (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => plugin({ spaceIds: [id] }).retrieve({ retriever: 'goodmem/memories', query: 'q' }),
    expect: () => ['POST /v1/memories:retrieve'],
    inBody: true,
  },
  {
    name: 'plugin spaceIds -> indexer goodmem/memories (request body)',
    field: /spaceIds\[0\]/,
    call: (id) =>
      plugin({ spaceIds: [id] }).index({ indexer: 'goodmem/memories', documents: [Document.fromText('x')] }),
    expect: () => ['POST /v1/memories'],
    inBody: true,
  },
  {
    name: 'plugin spaceIds -> tool goodmem/remember (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => callTool(plugin({ spaceIds: [id] }), 'remember', { text: 'canary' }),
    expect: () => ['POST /v1/memories'],
    inBody: true,
  },
  {
    name: 'plugin spaceIds -> tool goodmem/upload_file (request body)',
    field: /spaceIds\[0\]/,
    call: (id) => callTool(plugin({ spaceIds: [id], uploadDir }), 'upload_file', { fileName: 'note.txt' }),
    expect: () => ['POST /v1/memories'],
    inBody: true,
  },
  {
    name: 'plugin rerankerId -> tool goodmem/search (request body)',
    field: /rerankerId/,
    call: (id) => callTool(plugin({ rerankerId: id }), 'search', { query: 'q', topK: 1 }),
    expect: () => ['POST /v1/memories:retrieve'],
    inBody: true,
  },
];

// ---- the checks -----------------------------------------------------------

async function attempt(entry: Entry, id: string) {
  requests = [];
  let error: any;
  let result: unknown;
  try {
    result = await entry.call(id);
  } catch (e) {
    error = e;
  }
  return { error, result, sent: sent() };
}

function checkRefusals(group: string, entries: Entry[]) {
  describe(`${group}: a non-UUID id is refused before any request`, () => {
    for (const entry of entries) {
      it(entry.name, async () => {
        const leaked: Array<Record<string, unknown>> = [];
        const unnamed: Array<Record<string, unknown>> = [];
        for (const payload of PAYLOADS) {
          const { error, result, sent: wire } = await attempt(entry, payload);
          if (wire.length > 0 || !error) {
            leaked.push({
              payload,
              reported: error ? `error: ${String(error.message).split('\n')[0]}` : result,
              serverReceived: wire,
            });
          } else if (!entry.field.test(error.message) || !/uuid/i.test(error.message)) {
            unnamed.push({ payload, message: String(error.message).split('\n')[0] });
          }
        }
        if (leaked.length > 0) {
          assert.fail(`${entry.name} sent or accepted a non-UUID id:\n${JSON.stringify(leaked, null, 2)}`);
        }
        assert.deepEqual(unnamed, [], `${entry.name} refused without naming the field and UUID`);
      });
    }
  });
}

function checkValid(group: string, entries: Entry[]) {
  describe(`${group}: a valid UUID still reaches exactly the intended path`, () => {
    for (const entry of entries) {
      it(entry.name, async () => {
        const { error, sent: wire } = await attempt(entry, U);
        assert.equal(error, undefined, `valid id refused: ${error?.message}`);
        assert.deepEqual(wire, entry.expect(U));
        if (entry.inBody) assert.ok(requests.some((r) => r.body.includes(U)), 'id missing from the body');

        // An upper-case UUID is the same id; it is sent in canonical form.
        const upper = await attempt(entry, U.toUpperCase());
        assert.equal(upper.error, undefined, `upper-case id refused: ${upper.error?.message}`);
        assert.deepEqual(upper.sent, entry.expect(U));
      });
    }
  });
}

checkRefusals('model-facing tools', MODEL_TOOLS);
checkRefusals('developer-facing methods', DEVELOPER_METHODS);
checkRefusals('configured ids', CONFIGURED);
checkValid('model-facing tools', MODEL_TOOLS);
checkValid('developer-facing methods', DEVELOPER_METHODS);
checkValid('configured ids', CONFIGURED);

describe('a model driving the tools through ai.generate', () => {
  /**
   * A model that makes one tool call, then answers. It calls the tool by the
   * name Genkit advertised to it, and hands back what it was shown.
   */
  async function attacker(tool: string, input: Record<string, unknown>) {
    const ai = plugin({ allowAdminTools: true, allowDelete: true });
    const shown: any[] = [];
    ai.defineModel({ name: 'test/attacker' }, async (request) => {
      shown.push(...(request.tools ?? []));
      if (request.messages.some((m) => m.role === 'tool')) {
        return { message: { role: 'model', content: [{ text: 'done' }] }, finishReason: 'stop' };
      }
      const name = request.tools![0].name;
      return {
        message: { role: 'model', content: [{ toolRequest: { name, ref: '1', input } }] },
        finishReason: 'stop',
      };
    });
    let error: any;
    let response: any;
    try {
      response = await ai.generate({ model: 'test/attacker', prompt: 'tidy up', tools: [tool] });
    } catch (e) {
      error = e;
    }
    return { error, response, shown };
  }

  for (const [tool, field] of [
    ['goodmem/delete_memory', 'memoryId'],
    ['goodmem/delete_space', 'spaceId'],
  ] as const) {
    it(`${tool} with ${field}="../spaces/<U>" deletes nothing`, async () => {
      requests = [];
      const { error, response } = await attacker(tool, { [field]: `../spaces/${U}` });
      const toolOutput = response?.messages
        ?.flatMap((m: any) => m.content)
        .find((p: any) => p.toolResponse)?.toolResponse.output;
      if (sent().length > 0 || !error) {
        assert.fail(
          `${tool} was not refused: the server received ${JSON.stringify(sent())} ` +
            `and the tool answered ${JSON.stringify(toolOutput)}`
        );
      }
      assert.match(error.message, new RegExp(field));
    });
  }

  it('every id argument the model is shown is declared as a uuid', async () => {
    const idFields: Array<[string, string]> = [
      ['goodmem/get_memory', 'memoryId'],
      ['goodmem/delete_memory', 'memoryId'],
      ['goodmem/get_space', 'spaceId'],
      ['goodmem/update_space', 'spaceId'],
      ['goodmem/delete_space', 'spaceId'],
      ['goodmem/list_memories', 'spaceId'],
      ['goodmem/create_space', 'embedderId'],
    ];
    const undeclared: Array<Record<string, unknown>> = [];
    for (const [tool, field] of idFields) {
      requests = [];
      const { shown } = await attacker(tool, {});
      const property = shown[0]?.inputSchema?.properties?.[field];
      if (property?.format !== 'uuid') undeclared.push({ tool, field, shown: property });
    }
    assert.deepEqual(undeclared, []);
  });
});

describe('the plugin refuses a malformed configuration at startup', () => {
  it('a non-UUID spaceId, before any request', () => {
    for (const payload of SHARED_PAYLOADS) {
      assert.throws(
        () => goodmem({ baseUrl, apiKey: KEY, spaceIds: [SPACE_ID, payload] }),
        /spaceIds\[1\].*UUID/,
        JSON.stringify(payload)
      );
    }
    assert.deepEqual(sent(), []);
  });

  it('a non-UUID rerankerId, before any request', () => {
    assert.throws(
      () => goodmem({ baseUrl, apiKey: KEY, spaceIds: [SPACE_ID], rerankerId: `../rerankers/${U}` }),
      /rerankerId.*UUID/
    );
    assert.deepEqual(sent(), []);
  });

  it('accepts UUIDs', () => {
    assert.doesNotThrow(() =>
      goodmem({ baseUrl, apiKey: KEY, spaceIds: [SPACE_ID, U.toUpperCase()], rerankerId: EMBEDDER_ID })
    );
  });
});
