/**
 * GoodMem ids are UUIDs; anything else is refused before a request is made.
 *
 * Memory, space, embedder and reranker ids are interpolated into URL paths --
 * `/v1/memories/{id}`, `/v1/spaces/{id}`, `/v1/spaces/{id}/memories`. The
 * SDK percent-encodes them, which turns `/` into `%2F` but leaves a bare `..`
 * or `.` alone, and the URL parser resolves those: a memory id of `..` sent
 * `DELETE /v1/`. How the server treats `%2F` is unknown, and it was seen to
 * normalise `%2e%2e/spaces/<id>` into a traversal through another client. So
 * neither the encoding nor the server is relied on: every id -- whether a
 * model, the developer or the plugin configuration supplied it -- passes
 * through {@link requireUuid} at the call that sends it.
 */

import { GoodMemError } from './errors.js';

/** A canonical UUID: 8-4-4-4-12 hex digits, nothing before or after. */
export const UUID_PATTERN =
  /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

const SHOWN_CHARS = 80;

/**
 * Return `value` as a lower-case UUID, or refuse it.
 *
 * @param field the argument or setting the id came from, named in the refusal.
 * @throws {GoodMemError} if `value` is not a string holding exactly one
 * canonical UUID -- no whitespace, no path, no query, no encoding.
 */
export function requireUuid(value: unknown, field: string): string {
  if (typeof value === 'string' && UUID_PATTERN.test(value)) return value.toLowerCase();
  let shown = typeof value === 'string' ? JSON.stringify(value) : String(value);
  if (shown.length > SHOWN_CHARS) shown = `${shown.slice(0, SHOWN_CHARS)}...`;
  throw new GoodMemError(
    `${field} must be a UUID (8-4-4-4-12 hex digits), got ${shown}. GoodMem ids are UUIDs; ` +
      'anything else is refused before a request is made, because an id is part of the URL path.'
  );
}
