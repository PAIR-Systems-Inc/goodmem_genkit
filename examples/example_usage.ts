/**
 * GoodMem + Genkit example.
 *
 * Three scenarios:
 *   1. Persistent project context -- facts are indexed through the native
 *      indexer and recalled through the native retriever to ground a model.
 *   2. A two-role team knowledge pipeline -- a Scribe stores notes with the
 *      `goodmem/remember` tool inside a span, an Analyst summarises them
 *      from `goodmem/search`.
 *   3. A structured activity log -- entries carry a `category` in their
 *      metadata, and a second plugin instance scoped server-side to
 *      `category == "feat"` retrieves only the features. The developer sets
 *      the scope; the model never composes a filter.
 *
 * Every plugin instance is scoped to its space at construction time. The
 * model never chooses a space.
 *
 * Set GOODMEM_API_KEY, GOODMEM_BASE_URL and OPENAI_API_KEY. GOODMEM_EMBEDDER_ID
 * pins the embedder; otherwise the first one the server lists is used.
 * GOODMEM_VERIFY_SSL=false is for a local server with a self-signed
 * certificate only.
 *
 *   npx tsx examples/example_usage.ts
 */

import { openAI } from '@genkit-ai/compat-oai/openai';
import { Document, genkit } from 'genkit';
import { runInNewSpan } from 'genkit/tracing';
import { GoodMemConnection, goodmem } from '../src/index';

const BASE_URL = process.env.GOODMEM_BASE_URL ?? 'https://localhost:8080';
const API_KEY = process.env.GOODMEM_API_KEY;
if (!API_KEY) {
  console.error('Set GOODMEM_API_KEY before running this demo.');
  process.exit(1);
}
if (!process.env.OPENAI_API_KEY) {
  console.error('Set OPENAI_API_KEY before running this demo.');
  process.exit(1);
}
if (process.env.GOODMEM_VERIFY_SSL === 'false') {
  // Self-signed local certificate only. Never in production.
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
}

const MODEL = openAI.model('gpt-4o-mini');
const PROJECT_FACTS = [
  "I'm building a customer support assistant for our SaaS product.",
  'The team uses Python 3.12 with FastAPI and Postgres.',
  'For tests we use pytest with at least 80% coverage required.',
];
const TEAM_NOTES = [
  'Q2 goal: reduce customer support response time to under 2 hours.',
  'Our main services are auth-service, billing-service, and notifications-service.',
  'Known issue: notifications-service drops messages during high load.',
  'Team retro: the CI pipeline is too slow; we should parallelize tests.',
];
const RELEASE_LOG: Array<{ content: string; category: 'feat' | 'fix' | 'chore' | 'docs' }> = [
  { content: 'Added user profile editing to the dashboard.', category: 'feat' },
  { content: 'Built the CSV export feature.', category: 'feat' },
  { content: 'Resolved slow login on the mobile app.', category: 'fix' },
  { content: 'Fixed crash when opening large attachments.', category: 'fix' },
  { content: 'Upgraded Python version across services.', category: 'chore' },
  { content: 'Updated the API reference for billing endpoints.', category: 'docs' },
];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function section(title: string): void {
  console.log('\n' + '='.repeat(70) + `\n  ${title}\n` + '='.repeat(70));
}

/** One connection with no space scope, used only to set up and tear down. */
const admin = new GoodMemConnection({ baseUrl: BASE_URL, apiKey: API_KEY, spaceIds: ['setup'] });

function aiFor(spaceId: string, extra: Record<string, unknown> = {}) {
  return genkit({
    plugins: [openAI(), goodmem({ baseUrl: BASE_URL, apiKey: API_KEY!, spaceIds: [spaceId], ...extra })],
  });
}

async function tool(ai: ReturnType<typeof genkit>, name: string, input: Record<string, unknown>) {
  const action = await ai.registry.lookupAction(`/tool/goodmem/${name}`);
  if (!action) throw new Error(`Tool not found: goodmem/${name}`);
  const result: any = await (action as any)(input);
  return result.result ?? result;
}

/** Poll on the write path until a just-written memory is retrievable. */
async function waitUntilSearchable(ai: ReturnType<typeof genkit>, probe: string): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const out = await tool(ai, 'search', { query: probe, topK: 3 });
    if (out.totalResults > 0) return;
    await sleep(2000);
  }
}

async function scenario1PersistentMemory(spaceId: string): Promise<void> {
  section('Scenario 1: persistent project context across sessions');
  const ai = aiFor(spaceId);
  await ai.index({ indexer: 'goodmem/memories', documents: PROJECT_FACTS.map((f) => Document.fromText(f)) });
  console.log(`  Indexed ${PROJECT_FACTS.length} facts through the native indexer.`);
  await waitUntilSearchable(ai, 'coverage requirement');

  const question = 'What is our test coverage requirement?';
  const docs = await ai.retrieve({ retriever: 'goodmem/memories', query: question, options: { k: 3 } });
  console.log(`  Retrieved ${docs.length} document(s); partial=${docs[0]?.metadata?.goodmem_partial}`);
  const { text } = await ai.generate({
    model: MODEL,
    prompt: `Answer from the context only.\n\nContext:\n${docs.map((d) => d.text).join('\n')}\n\nQuestion: ${question}`,
  });
  console.log(`\n  User:  ${question}\n  Model: ${text}`);
}

async function scenario2ScribeAnalyst(spaceId: string): Promise<void> {
  section('Scenario 2: two-role team knowledge pipeline');
  const ai = aiFor(spaceId);
  await runInNewSpan(ai.registry, { metadata: { name: 'scribe.store-notes' } }, async () => {
    for (const note of TEAM_NOTES) await tool(ai, 'remember', { text: note });
    console.log(`  Scribe stored ${TEAM_NOTES.length} notes.`);
  });
  await waitUntilSearchable(ai, 'CI pipeline');

  const summary = await runInNewSpan<string>(ai.registry, { metadata: { name: 'analyst.summarize' } }, async () => {
    const found = await tool(ai, 'search', { query: 'services and current priorities', topK: 5 });
    if (found.partial) console.warn(`  Analyst: retrieval was degraded -- ${found.warning}`);
    const { text } = await ai.generate({
      model: MODEL,
      prompt: `Summarise what the team knows about its services and priorities, from these notes only:\n${found.results
        .map((r: any) => `- ${r.text}`)
        .join('\n')}`,
    });
    return text;
  });
  console.log(`\n  Analyst: ${summary}`);
}

async function scenario3MetadataScope(spaceId: string): Promise<void> {
  section('Scenario 3: structured team activity log');
  const scoped = new GoodMemConnection({ baseUrl: BASE_URL, apiKey: API_KEY!, spaceIds: [spaceId] });
  for (const entry of RELEASE_LOG) await scoped.createFromText(entry.content, { category: entry.category });
  console.log(`  Wrote ${RELEASE_LOG.length} entries with a category in their metadata.`);
  await waitUntilSearchable(aiFor(spaceId), 'CSV export');

  // A second plugin instance whose retrieval is scoped server-side.
  const featuresOnly = aiFor(spaceId, { metadataFilter: { category: 'feat' } });
  const docs = await featuresOnly.retrieve({ retriever: 'goodmem/memories', query: 'what did we ship', options: { k: 10 } });
  console.log(`  Retrieved ${docs.length} feature entr${docs.length === 1 ? 'y' : 'ies'}:`);
  for (const d of docs) console.log(`    [${d.metadata?.category}] ${d.text}`);
}

async function main(): Promise<void> {
  const embedderId = process.env.GOODMEM_EMBEDDER_ID ?? (await admin.listEmbedders())[0]?.embedderId;
  if (!embedderId) throw new Error('No embedders on the GoodMem server. Register one and retry.');

  const created: string[] = [];
  for (const name of ['genkit-goodmem-example', 'genkit-goodmem-example-team', 'genkit-goodmem-example-log']) {
    created.push((await admin.createSpace(name, embedderId)).spaceId);
  }
  try {
    await scenario1PersistentMemory(created[0]);
    await scenario2ScribeAnalyst(created[1]);
    await scenario3MetadataScope(created[2]);
  } finally {
    section('Cleanup');
    for (const id of created) await admin.deleteSpace(id);
    const remaining = (await admin.listSpaces()).filter((s) => created.includes(s.spaceId));
    console.log(remaining.length === 0 ? '  all example spaces deleted' : `  WARNING: not deleted: ${remaining.map((s) => s.spaceId)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
