/**
 * Shared handling of a GoodMem retrieval stream.
 *
 * Every retrieval in this plugin is folded through this module, so the tool
 * and the retriever cannot drift apart in how they classify a status, join a
 * chunk to its memory, or orient a score.
 */

/**
 * Status codes that report an *optional* feature the caller never configured.
 * The server files both under "Informational status messages (non-error)":
 * nothing the caller asked for is missing, so they are noise unconditionally
 * and their `details` are never inspected to decide.
 */
export const INFORMATIONAL_CODES = new Set([
  'LLM_CAPABILITY_INFERRED',
  'FEATURE_DISABLED',
]);

/** Surfaced in place of a code this build does not recognise. */
export const UNKNOWN_CODE = 'UNKNOWN';

/** Reported when the stream ended mid-line or carried an undecodable line. */
export const MALFORMED_STREAM_CODE = 'MALFORMED_STREAM';

/** Codes this build knows about. Anything else becomes `UNKNOWN`. */
export const KNOWN_CODES = new Set([
  'GOODMEM_STATUS_CODE_UNSPECIFIED',
  'INVALID_ARGUMENT',
  'NOT_FOUND',
  'PERMISSION_DENIED',
  'FAILED_PRECONDITION',
  'EMBEDDER_FAILED',
  'EMBEDDER_UNAVAILABLE',
  'EMBEDDER_TIMEOUT',
  'VECTOR_SEARCH_FAILED',
  'VECTOR_SEARCH_PARTIAL',
  'VECTOR_SEARCH_TIMEOUT',
  'SPACE_INACCESSIBLE',
  'SPACE_NOT_FOUND',
  'SPACE_NO_EMBEDDERS',
  'CHUNK_NOT_FOUND',
  'MEMORY_LOAD_FAILED',
  'MEMORY_CONTENT_UNAVAILABLE',
  'RERANKING_FAILED',
  'SUMMARIZATION_FAILED',
  'SUMMARIZATION_TIMEOUT',
  'RATE_LIMITED',
  'RESOURCE_EXHAUSTED',
  'CONFIGURATION_ERROR',
  'LLM_CAPABILITY_INFERRED',
  'FEATURE_DISABLED',
]);

/** One status event reported by the server during a retrieval. */
export interface RetrievalStatus {
  /** The server's code, or `UNKNOWN` when this build does not recognise it. */
  code: string;
  /** The server's own human-readable message. */
  message: string;
  /** Any structured detail the server attached. */
  details?: Record<string, unknown>;
}

/** One chunk, joined to the memory it came from. */
export interface RetrievalHit {
  chunkId: string;
  text: string;
  memoryId: string;
  spaceId: string;
  /** Relevance oriented so that higher is better. */
  score: number | null;
  /** The score exactly as the server sent it. */
  rawScore: number | null;
  /** `vector` or `reranker` -- the two are not on a common scale. */
  scoreKind: 'vector' | 'reranker';
  contentType: string;
  metadata: Record<string, unknown>;
}

/** Everything one retrieval produced. */
export interface RetrievalOutcome {
  hits: RetrievalHit[];
  statuses: RetrievalStatus[];
  /**
   * True when the server reported a real problem during this retrieval.
   * Independent of whether hits came back.
   */
  partial: boolean;
  /**
   * True when the hits carry reranker scores: a reranker was requested *and*
   * the server did not report that it failed. Read from the response, never
   * from configuration alone -- a failed reranker still returns vector hits.
   */
  reranked: boolean;
  resultSetId: string;
  abstractReply?: string;
}

/**
 * Classify one status event.
 *
 * Implements Q1 and Q3 of the retrieval status contract: the two
 * informational codes are noise unconditionally, and a code this build does
 * not recognise is surfaced as `UNKNOWN` rather than dropped or thrown on.
 */
export function classifyStatus(
  rawCode: string | null | undefined,
  message: string
): { status: RetrievalStatus; informational: boolean } {
  if (!rawCode || !KNOWN_CODES.has(rawCode)) {
    // Q3: a server upgrade must not silently change behaviour.
    return {
      status: { code: UNKNOWN_CODE, message, details: rawCode ? { serverCode: rawCode } : undefined },
      informational: false,
    };
  }
  return {
    status: { code: rawCode, message },
    informational: INFORMATIONAL_CODES.has(rawCode),
  };
}

/**
 * Return a score oriented so that a higher number is a better match.
 *
 * GoodMem vector scores are negative distances -- `-0.51` is closer than
 * `-0.88` -- while reranker scores are already higher-is-better on a
 * provider-dependent scale. Negating a reranker score would invert the
 * ranking, so only vector scores are flipped.
 */
export function orientScore(raw: number | null | undefined, reranked: boolean): number | null {
  if (raw === null || raw === undefined) return null;
  return reranked ? raw : -raw;
}

/**
 * Whether the server reported that the requested reranker was not applied.
 *
 * On `RERANKING_FAILED`, or a `NOT_FOUND` naming the reranker, GoodMem still
 * returns the vector-scored hits it had before reranking. Those scores are
 * negative distances, so they must be oriented as vector scores and must not
 * meet a reranker threshold -- which would discard every one of them.
 */
export function rerankerFailed(statuses: RetrievalStatus[]): boolean {
  return statuses.some(
    (s) =>
      s.code === 'RERANKING_FAILED' ||
      (s.code === 'NOT_FOUND' &&
        (s.details?.reranker_id !== undefined || /reranker/i.test(s.message)))
  );
}

/** A one-line summary of why a retrieval was degraded. */
export function warningText(statuses: RetrievalStatus[]): string {
  if (statuses.length === 0) return '';
  const parts = statuses.map((s) => (s.message ? `${s.code}: ${s.message}` : s.code));
  return `GoodMem reported a problem during retrieval -- ${parts.join('; ')}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
}

/**
 * Fold a GoodMem retrieval stream into an outcome.
 *
 * Chunks are joined to their memory by memory UUID rather than by the
 * positional `memoryIndex` the stream also carries, so a reordered or partial
 * stream cannot attach a chunk to the wrong memory. Chunks are de-duplicated
 * by chunk id -- never by memory id, which would collapse distinct chunks of
 * one document.
 *
 * A stream that ends badly is reported as `MALFORMED_STREAM` with whatever
 * arrived kept, rather than thrown away or presented as complete.
 */
export async function outcomeFromEvents(
  events: AsyncIterable<any>,
  rerankRequested = false
): Promise<RetrievalOutcome> {
  const outcome: RetrievalOutcome = {
    hits: [],
    statuses: [],
    partial: false,
    reranked: false,
    resultSetId: '',
  };
  const memories = new Map<string, Record<string, unknown>>();
  const pending: Array<{ hit: RetrievalHit; memoryId: string }> = [];
  const seen = new Set<string>();

  let received = 0;
  try {
    for await (const event of events) {
      received += 1;
      if (event?.resultSetBoundary) {
        const id = event.resultSetBoundary.resultSetId;
        if (id) outcome.resultSetId = String(id);
        continue;
      }
      if (event?.status) {
        const { status, informational } = classifyStatus(
          event.status.code,
          String(event.status.message ?? '')
        );
        if (event.status.details) status.details = { ...(status.details ?? {}), ...asRecord(event.status.details) };
        if (!informational) {
          outcome.statuses.push(status);
          outcome.partial = true;
        }
        continue;
      }
      if (event?.memoryDefinition) {
        const mem = asRecord(event.memoryDefinition);
        const id = String(mem.memoryId ?? '');
        if (id) memories.set(id, mem);
        continue;
      }
      if (event?.abstractReply) {
        const reply = event.abstractReply;
        outcome.abstractReply = String(reply.reply ?? reply.text ?? reply.content ?? '');
        continue;
      }
      const inner = event?.retrievedItem?.chunk?.chunk;
      if (!inner) continue;
      const chunkId = String(inner.chunkId ?? '');
      if (!chunkId || seen.has(chunkId)) continue;
      seen.add(chunkId);
      const rawScore = event.retrievedItem.chunk.relevanceScore ?? null;
      const memoryId = String(inner.memoryId ?? '');
      pending.push({
        hit: {
          chunkId,
          text: String(inner.chunkText ?? ''),
          memoryId,
          spaceId: '',
          rawScore: rawScore === null ? null : Number(rawScore),
          // Oriented below, once every status has arrived.
          score: null,
          scoreKind: 'vector',
          contentType: '',
          metadata: {},
        },
        memoryId,
      });
    }
  } catch (error: any) {
    if (received === 0) {
      // Nothing ever arrived: the request itself failed. That is an error,
      // not a partial result -- reporting it as "degraded, no hits" would
      // present a dead connection as a search that merely found nothing.
      throw error;
    }
    outcome.statuses.push({
      code: MALFORMED_STREAM_CODE,
      message: `The retrieval stream ended badly after ${received} event(s); results may be incomplete: ${error?.message ?? error}`,
    });
    outcome.partial = true;
  }

  // A status can arrive after the hits it concerns, so what kind of score the
  // hits carry is decided only once the stream is done.
  outcome.reranked = rerankRequested && !rerankerFailed(outcome.statuses);
  for (const { hit, memoryId } of pending) {
    hit.score = orientScore(hit.rawScore, outcome.reranked);
    hit.scoreKind = outcome.reranked ? 'reranker' : 'vector';
    const mem = memories.get(memoryId);
    if (mem) {
      hit.spaceId = String(mem.spaceId ?? '');
      hit.contentType = String(mem.contentType ?? '');
      hit.metadata = asRecord(mem.metadata);
    }
    outcome.hits.push(hit);
  }
  return outcome;
}
