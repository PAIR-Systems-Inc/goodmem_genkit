/**
 * GoodMem plugin for Genkit.
 *
 * Registers a native retriever and indexer -- so GoodMem works with Genkit's
 * RAG paths, not only through tool calls -- plus a deliberately narrow set of
 * tools for an agent to use.
 */

import { Document, Genkit, z } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { Goodmem } from '@pairsystems/goodmem';

import * as filters from './filters.js';
import { fromMapping } from './filters.js';
import {
  MALFORMED_STREAM_CODE,
  UNKNOWN_CODE,
  outcomeFromEvents,
  warningText,
  type RetrievalHit,
  type RetrievalOutcome,
  type RetrievalStatus,
} from './results.js';
import { GoodMemUploadError, resolveUploadPath } from './uploads.js';

export { filters, GoodMemUploadError, MALFORMED_STREAM_CODE, UNKNOWN_CODE };
export type { RetrievalHit, RetrievalOutcome, RetrievalStatus };

/** Upper bound on items a single listing will pull. */
const DEFAULT_MAX_LIST_ITEMS = 200;
const DEFAULT_TIMEOUT_MS = 30_000;

/** Configuration for the GoodMem plugin. */
export interface GoodMemPluginParams {
  /** The GoodMem server URL. */
  baseUrl: string;
  /** The GoodMem API key. */
  apiKey: string;
  /**
   * The spaces this plugin reads from and writes to. A model never chooses
   * a space; retrieval and writes are scoped here.
   */
  spaceIds: string[];
  /** Per-request timeout in milliseconds. Defaults to 30s. */
  timeoutMs?: number;
  /**
   * A directory that file uploads are confined to. Without one, no upload
   * tool is registered and no path is ever read from disk.
   */
  uploadDir?: string;
  /** A reranker applied to retrieval. */
  rerankerId?: string;
  /**
   * Drop hits scoring below this value. Applies only with `rerankerId` set,
   * because reranker scales are provider-dependent. Off by default.
   */
  minScore?: number;
  /** Metadata every retrieved memory must match, applied server-side. */
  metadataFilter?: Record<string, unknown>;
  /** Whether the model may store new memories. Defaults to true. */
  allowWrite?: boolean;
  /** Whether space and embedder management is exposed. Defaults to false. */
  allowAdminTools?: boolean;
  /** Whether the model may delete memories and spaces. Defaults to false. */
  allowDelete?: boolean;
  /** Upper bound on items returned by a listing. */
  maxListItems?: number;
}

/** Raised when a GoodMem operation fails, carrying the server's own message. */
export class GoodMemError extends Error {
  readonly statusCode?: number;
  readonly body?: string;
  constructor(message: string, statusCode?: number, body?: string) {
    super(message);
    this.name = 'GoodMemError';
    this.statusCode = statusCode;
    this.body = body;
  }
}

function wrapError(error: any, what: string): GoodMemError {
  const statusCode = error?.statusCode ?? error?.status;
  const body = typeof error?.body === 'string' ? error.body : undefined;
  let detail = error?.message ?? String(error);
  if (body && !detail.includes(body)) detail = `${detail} -- ${body}`;
  return new GoodMemError(`${what} failed: ${detail}`, statusCode, body);
}

function spaceEmbedderIds(space: any): string[] {
  return (space?.spaceEmbedders ?? [])
    .map((e: any) => e?.embedderId)
    .filter(Boolean)
    .map(String);
}

/** Decode memory content by its content type: text as text, else base64. */
export function decodeContent(
  raw: Uint8Array,
  contentType: string
): { content: string; encoding: 'text' | 'base64' } {
  const primary = (contentType || '').split(';')[0].trim().toLowerCase();
  let charset = 'utf-8';
  for (const part of (contentType || '').split(';').slice(1)) {
    if (part.includes('charset=')) charset = part.split('charset=')[1].trim() || 'utf-8';
  }
  const textual =
    primary.startsWith('text/') ||
    ['application/json', 'application/xml', 'application/javascript'].includes(primary);
  const buffer = Buffer.from(raw);
  if (textual) {
    try {
      return { content: new TextDecoder(charset, { fatal: true }).decode(raw), encoding: 'text' };
    } catch {
      return { content: buffer.toString('base64'), encoding: 'base64' };
    }
  }
  return { content: buffer.toString('base64'), encoding: 'base64' };
}

/** A connection to GoodMem, shared by the plugin's retriever, indexer and tools. */
export class GoodMemConnection {
  readonly client: Goodmem;
  readonly spaceIds: string[];
  readonly uploadDir?: string;
  readonly rerankerId?: string;
  readonly minScore?: number;
  readonly metadataFilter: Record<string, unknown>;
  readonly maxListItems: number;

  constructor(params: GoodMemPluginParams) {
    this.client = new Goodmem({
      baseUrl: params.baseUrl,
      apiKey: params.apiKey,
      timeoutMs: params.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    } as any);
    this.spaceIds = params.spaceIds;
    this.uploadDir = params.uploadDir;
    this.rerankerId = params.rerankerId;
    this.minScore = params.minScore;
    this.metadataFilter = params.metadataFilter ?? {};
    this.maxListItems = params.maxListItems ?? DEFAULT_MAX_LIST_ITEMS;
  }

  private spaceKeys(): Array<Record<string, unknown>> {
    const expression = fromMapping(this.metadataFilter);
    return this.spaceIds.map((spaceId) =>
      expression ? { spaceId, filter: expression } : { spaceId }
    );
  }

  /** Run one retrieval and fold the stream into an outcome. */
  async retrieve(query: string, topK: number): Promise<RetrievalOutcome> {
    const request: Record<string, unknown> = {
      message: query,
      spaceKeys: this.spaceKeys(),
      requestedSize: topK,
      fetchMemory: true,
    };
    if (this.rerankerId) request.rerankerId = this.rerankerId;

    let outcome: RetrievalOutcome;
    try {
      const events = this.client.memories.retrieve(request as any);
      outcome = await outcomeFromEvents(events, Boolean(this.rerankerId));
    } catch (error: any) {
      throw wrapError(error, 'Retrieval');
    }

    if (this.minScore !== undefined && this.rerankerId) {
      const kept = outcome.hits.filter((h) => h.score !== null && h.score >= this.minScore!);
      if (outcome.hits.length > 0 && kept.length === 0) {
        const scores = outcome.hits.map((h) => h.score).filter((s): s is number => s !== null);
        console.warn(
          `[goodmem] minScore=${this.minScore} removed all ${outcome.hits.length} reranked ` +
            `result(s); observed scores ranged ${Math.min(...scores).toFixed(4)}..` +
            `${Math.max(...scores).toFixed(4)}. Reranker scales are provider-dependent, not 0-1.`
        );
      }
      outcome.hits = kept;
    }
    if (outcome.partial) console.warn(`[goodmem] ${warningText(outcome.statuses)}`);
    return outcome;
  }

  async createFromText(text: string, metadata?: Record<string, unknown>) {
    try {
      return await this.client.memories.create({
        spaceId: this.spaceIds[0],
        originalContent: text,
        contentType: 'text/plain',
        ...(metadata ? { metadata } : {}),
      } as any);
    } catch (error: any) {
      throw wrapError(error, 'Creating a memory');
    }
  }

  async createFromFile(fileName: string, metadata?: Record<string, unknown>) {
    const resolved = resolveUploadPath(fileName, this.uploadDir);
    try {
      return await (this.client.memories as any).createFromPath({
        path: resolved,
        spaceId: this.spaceIds[0],
        ...(metadata ? { metadata } : {}),
      });
    } catch (error: any) {
      if (error instanceof GoodMemUploadError) throw error;
      throw wrapError(error, 'Uploading a file');
    }
  }

  async getMemory(memoryId: string, includeContent: boolean) {
    let memory: any;
    try {
      memory = await this.client.memories.get(memoryId);
    } catch (error: any) {
      throw wrapError(error, `Fetching memory ${memoryId}`);
    }
    const result: Record<string, unknown> = { success: true, memory };
    if (includeContent) {
      let raw: Uint8Array;
      try {
        raw = await this.client.memories.content(memoryId);
      } catch (error: any) {
        throw wrapError(error, `Fetching content of memory ${memoryId}`);
      }
      const decoded = decodeContent(raw, String(memory?.contentType ?? ''));
      result.content = decoded.content;
      result.contentEncoding = decoded.encoding;
    }
    return result;
  }

  async listSpaces() {
    const out: any[] = [];
    try {
      for await (const space of await this.client.spaces.list({} as any)) {
        out.push(space);
        if (out.length >= this.maxListItems) break;
      }
    } catch (error: any) {
      throw wrapError(error, 'Listing spaces');
    }
    return out.map((s) => ({
      spaceId: String(s.spaceId ?? ''),
      name: String(s.name ?? ''),
      embedderIds: spaceEmbedderIds(s),
    }));
  }

  async listMemories(spaceId?: string) {
    const target = spaceId ?? this.spaceIds[0];
    const out: any[] = [];
    try {
      for await (const memory of await this.client.memories.list(target, {} as any)) {
        out.push(memory);
        if (out.length >= this.maxListItems) break;
      }
    } catch (error: any) {
      throw wrapError(error, 'Listing memories');
    }
    return out.map((m) => ({
      memoryId: String(m.memoryId ?? ''),
      spaceId: String(m.spaceId ?? ''),
      contentType: String(m.contentType ?? ''),
      processingStatus: String(m.processingStatus ?? ''),
      metadata: m.metadata ?? {},
    }));
  }

  async listEmbedders() {
    const out: any[] = [];
    try {
      for await (const embedder of await this.client.embedders.list({} as any)) {
        out.push(embedder);
      }
    } catch (error: any) {
      throw wrapError(error, 'Listing embedders');
    }
    return out.map((e) => ({
      embedderId: String(e.embedderId ?? ''),
      displayName: String(e.displayName ?? ''),
      modelIdentifier: String(e.modelIdentifier ?? ''),
    }));
  }

  /**
   * Create a space, or reuse one whose embedder already matches.
   *
   * A space cannot change embedder after creation, so reusing by name alone
   * silently writes vectors from a different model than the caller asked for.
   */
  async createSpace(name: string, embedderId: string) {
    const existing = (await this.listSpaces()).filter((s) => s.name === name);
    if (existing.length > 1) {
      throw new GoodMemError(
        `${existing.length} spaces are named ${JSON.stringify(name)}; refusing to guess ` +
          'which one was meant. Pass a space id instead.'
      );
    }
    if (existing.length === 1) {
      if (!existing[0].embedderIds.includes(embedderId)) {
        throw new GoodMemError(
          `Space ${JSON.stringify(name)} already exists and is indexed by embedder(s) ` +
            `${JSON.stringify(existing[0].embedderIds)}, not ${JSON.stringify(embedderId)}. ` +
            'An embedder cannot be changed after creation.'
        );
      }
      return { success: true, spaceId: existing[0].spaceId, name, embedderId, reused: true };
    }
    try {
      const space: any = await this.client.spaces.create({
        name,
        spaceEmbedders: [{ embedderId, defaultRetrievalWeight: 1.0 }],
      } as any);
      return {
        success: true,
        spaceId: String(space.spaceId ?? ''),
        name: String(space.name ?? name),
        embedderId,
        reused: false,
      };
    } catch (error: any) {
      throw wrapError(error, `Creating space ${JSON.stringify(name)}`);
    }
  }

  /**
   * Rename a space or edit its labels.
   *
   * `publicRead` is deliberately not offered: the server removed the field
   * and answers `400 Unrecognized field "publicRead"`.
   */
  async updateSpace(
    spaceId: string,
    opts: { name?: string; labels?: Record<string, string>; replaceLabels?: boolean }
  ) {
    const request: Record<string, unknown> = {};
    if (opts.name !== undefined) request.name = opts.name;
    if (opts.labels !== undefined) {
      request[opts.replaceLabels ? 'replaceLabels' : 'mergeLabels'] = opts.labels;
    }
    if (Object.keys(request).length === 0) {
      throw new GoodMemError('updateSpace() needs a name or labels to change.');
    }
    try {
      const space: any = await this.client.spaces.update(spaceId, request as any);
      return { success: true, spaceId: String(space.spaceId ?? spaceId), name: String(space.name ?? '') };
    } catch (error: any) {
      throw wrapError(error, `Updating space ${spaceId}`);
    }
  }

  async getSpace(spaceId: string) {
    try {
      const space: any = await this.client.spaces.get(spaceId);
      return {
        success: true,
        spaceId: String(space.spaceId ?? ''),
        name: String(space.name ?? ''),
        embedderIds: spaceEmbedderIds(space),
        labels: space.labels ?? {},
      };
    } catch (error: any) {
      throw wrapError(error, `Fetching space ${spaceId}`);
    }
  }

  async deleteSpace(spaceId: string) {
    try {
      await this.client.spaces.delete(spaceId);
      return { success: true, spaceId };
    } catch (error: any) {
      throw wrapError(error, `Deleting space ${spaceId}`);
    }
  }

  async deleteMemory(memoryId: string) {
    try {
      await this.client.memories.delete(memoryId);
      return { success: true, memoryId };
    } catch (error: any) {
      throw wrapError(error, `Deleting memory ${memoryId}`);
    }
  }
}

/** Turn a hit into a Genkit Document, carrying its ids, score and metadata. */
export function hitToDocument(hit: RetrievalHit, outcome: RetrievalOutcome): Document {
  return Document.fromText(hit.text, {
    goodmem_chunk_id: hit.chunkId,
    goodmem_memory_id: hit.memoryId,
    goodmem_space_id: hit.spaceId,
    goodmem_score: hit.score,
    goodmem_raw_score: hit.rawScore,
    goodmem_score_kind: hit.scoreKind,
    goodmem_partial: outcome.partial,
    ...(outcome.partial ? { goodmem_statuses: outcome.statuses } : {}),
    ...hit.metadata,
  });
}

const RetrieverConfigSchema = z.object({
  k: z.number().int().positive().optional().describe('How many chunks to return.'),
});

const SearchInputSchema = z.object({
  query: z.string().describe('A natural-language description of what to find.'),
  topK: z.number().int().positive().max(100).default(5).describe('How many results to return.'),
});

const RememberInputSchema = z.object({
  text: z.string().describe('The text to remember.'),
});

const UploadInputSchema = z.object({
  fileName: z
    .string()
    .describe('The name of a file inside the configured upload directory. Any other path is refused.'),
});

/**
 * The GoodMem plugin.
 *
 * Registers `goodmem` as a retriever and an indexer, plus `goodmem/search`
 * and (by default) `goodmem/remember` as tools. Space management, deletion
 * and uploads are each opt-in: a model does not need to administer a memory
 * server in order to use one.
 */
export function goodmem(params: GoodMemPluginParams): GenkitPlugin {
  if (!params.baseUrl) {
    throw new Error('GoodMem plugin requires a baseUrl: the URL of your GoodMem server.');
  }
  if (!params.apiKey) {
    throw new Error(
      'GoodMem plugin requires an apiKey. Pass it in or set GOODMEM_API_KEY.'
    );
  }
  if (!params.spaceIds || params.spaceIds.length === 0) {
    throw new Error(
      'GoodMem plugin requires spaceIds: the space or spaces it may read and write. ' +
        'The model never chooses a space.'
    );
  }

  return genkitPlugin('goodmem', async (ai: Genkit) => {
    const conn = new GoodMemConnection(params);

    // ---- native retriever: this is what Genkit's RAG paths consume --------
    ai.defineRetriever(
      { name: 'goodmem/memories', configSchema: RetrieverConfigSchema },
      async (query, options) => {
        const outcome = await conn.retrieve(query.text, options?.k ?? 5);
        if (outcome.partial && outcome.hits.length === 0) {
          console.warn(
            `[goodmem] retrieval returned no documents and the server reported a ` +
              `problem -- this is not an empty index: ${warningText(outcome.statuses)}`
          );
        }
        return { documents: outcome.hits.map((h) => hitToDocument(h, outcome)) };
      }
    );

    // ---- native indexer --------------------------------------------------
    ai.defineIndexer({ name: 'goodmem/memories' }, async (docs) => {
      for (const doc of docs) {
        await conn.createFromText(doc.text, doc.metadata);
      }
    });

    // ---- tools -----------------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/search',
        description:
          'Search stored memories for information relevant to a question. Use this ' +
          'to recall facts, documents or past context saved earlier.',
        inputSchema: SearchInputSchema,
      },
      async ({ query, topK }) => {
        const outcome = await conn.retrieve(query, topK);
        return {
          success: true,
          query,
          results: outcome.hits,
          totalResults: outcome.hits.length,
          partial: outcome.partial,
          statuses: outcome.statuses,
          ...(outcome.partial ? { warning: warningText(outcome.statuses) } : {}),
        };
      }
    );

    if (params.allowWrite !== false) {
      ai.defineTool(
        {
          name: 'goodmem/remember',
          description: 'Store a piece of text as a memory for later recall.',
          inputSchema: RememberInputSchema,
        },
        async ({ text }) => {
          const memory: any = await conn.createFromText(text);
          return {
            success: true,
            memoryId: String(memory?.memoryId ?? ''),
            spaceId: String(memory?.spaceId ?? conn.spaceIds[0]),
          };
        }
      );
    }

    if (params.uploadDir) {
      ai.defineTool(
        {
          name: 'goodmem/upload_file',
          description:
            'Store a file from the configured upload directory as a memory. Only ' +
            'files inside that directory can be uploaded.',
          inputSchema: UploadInputSchema,
        },
        async ({ fileName }) => {
          const memory: any = await conn.createFromFile(fileName);
          return {
            success: true,
            memoryId: String(memory?.memoryId ?? ''),
            spaceId: String(memory?.spaceId ?? conn.spaceIds[0]),
          };
        }
      );
    }

    if (params.allowAdminTools) {
      ai.defineTool(
        { name: 'goodmem/list_spaces', description: 'List GoodMem spaces.', inputSchema: z.object({}) },
        async () => ({ success: true, spaces: await conn.listSpaces() })
      );
      ai.defineTool(
        {
          name: 'goodmem/list_embedders',
          description: 'List the embedder models available in GoodMem.',
          inputSchema: z.object({}),
        },
        async () => ({ success: true, embedders: await conn.listEmbedders() })
      );
      ai.defineTool(
        {
          name: 'goodmem/get_space',
          description: 'Fetch one GoodMem space by id.',
          inputSchema: z.object({ spaceId: z.string() }),
        },
        async ({ spaceId }) => conn.getSpace(spaceId)
      );
      ai.defineTool(
        {
          name: 'goodmem/create_space',
          description: 'Create a GoodMem space, or reuse one whose embedder matches.',
          inputSchema: z.object({ name: z.string(), embedderId: z.string() }),
        },
        async ({ name, embedderId }) => conn.createSpace(name, embedderId)
      );
      ai.defineTool(
        {
          name: 'goodmem/update_space',
          description: 'Rename a GoodMem space or edit its labels.',
          inputSchema: z.object({
            spaceId: z.string(),
            name: z.string().optional(),
            labels: z.record(z.string()).optional(),
            replaceLabels: z.boolean().optional(),
          }),
        },
        async ({ spaceId, name, labels, replaceLabels }) =>
          conn.updateSpace(spaceId, { name, labels, replaceLabels })
      );
      ai.defineTool(
        {
          name: 'goodmem/list_memories',
          description: 'List memories in a GoodMem space.',
          inputSchema: z.object({ spaceId: z.string().optional() }),
        },
        async ({ spaceId }) => ({ success: true, memories: await conn.listMemories(spaceId) })
      );
      ai.defineTool(
        {
          name: 'goodmem/get_memory',
          description: 'Fetch a GoodMem memory by id, optionally with its content.',
          inputSchema: z.object({
            memoryId: z.string(),
            includeContent: z.boolean().default(false),
          }),
        },
        async ({ memoryId, includeContent }) => conn.getMemory(memoryId, includeContent)
      );
    }

    if (params.allowDelete) {
      ai.defineTool(
        {
          name: 'goodmem/delete_memory',
          description: 'Permanently delete a GoodMem memory.',
          inputSchema: z.object({ memoryId: z.string() }),
        },
        async ({ memoryId }) => conn.deleteMemory(memoryId)
      );
      ai.defineTool(
        {
          name: 'goodmem/delete_space',
          description: 'Permanently delete a GoodMem space and every memory in it.',
          inputSchema: z.object({ spaceId: z.string() }),
        },
        async ({ spaceId }) => conn.deleteSpace(spaceId)
      );
    }
  });
}

export default goodmem;
