/**
 * Builds GoodMem metadata filter expressions safely.
 *
 * GoodMem filters are expressions evaluated server-side, not SQL.
 * Interpolating a caller's value into one is both a filter-injection hole and
 * a correctness bug: an ordinary apostrophe produces a malformed expression.
 *
 * The escaping and casting rules below were verified live against GoodMem
 * server v1.0.320:
 *
 * - a literal is single-quoted; `'` escapes as `\'` and a backslash as `\\`.
 *   SQL-style `''` doubling and double-quoted strings are both rejected with
 *   HTTP 400.
 * - a raw newline inside a literal is rejected, so control characters are
 *   refused here rather than sent.
 * - `val()` yields JSON, so a comparison must cast to the stored type:
 *   TEXT for strings, NUMERIC for numbers, BOOLEAN for booleans. **The cast
 *   has to match:** comparing a boolean as TEXT is accepted with HTTP 200 and
 *   matches nothing.
 */

const SAFE_FIELD = /^[A-Za-z_][A-Za-z0-9_.-]*$/;
// eslint-disable-next-line no-control-regex
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

/** Thrown when a filter cannot be expressed safely. */
export class GoodMemFilterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GoodMemFilterError';
  }
}

/** Quote a string as a GoodMem filter literal. */
export function escapeLiteral(value: string): string {
  if (CONTROL_CHARS.test(value)) {
    throw new GoodMemFilterError(
      'Filter values cannot contain control characters (including newlines ' +
        'and tabs); the server rejects them inside a literal.'
    );
  }
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

function checkField(field: string): string {
  if (!SAFE_FIELD.test(field ?? '')) {
    throw new GoodMemFilterError(
      `Unsupported metadata field name ${JSON.stringify(field)}. Field names ` +
        'may contain letters, digits, underscore, dot and hyphen, and must ' +
        'start with a letter or underscore.'
    );
  }
  return field;
}

function accessor(field: string, cast: string): string {
  return `CAST(val('$.${checkField(field)}') AS ${cast})`;
}

function render(value: unknown): [string, string] {
  // Booleans first: stringifying one produces a filter the server accepts
  // and that matches nothing.
  if (typeof value === 'boolean') return ['BOOLEAN', value ? 'true' : 'false'];
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new GoodMemFilterError('Filter numbers must be finite.');
    }
    return ['NUMERIC', String(value)];
  }
  if (typeof value === 'string') return ['TEXT', escapeLiteral(value)];
  throw new GoodMemFilterError(
    `Unsupported filter value type ${typeof value}; use string, number or boolean.`
  );
}

/** An equality filter for one metadata field. */
export function equals(field: string, value: unknown): string {
  const [cast, literal] = render(value);
  return `${accessor(field, cast)} = ${literal}`;
}

/** An inequality filter for one metadata field. */
export function notEquals(field: string, value: unknown): string {
  const [cast, literal] = render(value);
  return `${accessor(field, cast)} != ${literal}`;
}

/** An ordering comparison against a numeric metadata field. */
export function compare(field: string, operator: '>' | '>=' | '<' | '<=', value: number): string {
  if (!['>', '>=', '<', '<='].includes(operator)) {
    throw new GoodMemFilterError(`Unsupported comparison operator ${operator}.`);
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new GoodMemFilterError('Ordering comparisons apply to finite numbers only.');
  }
  return `${accessor(field, 'NUMERIC')} ${operator} ${value}`;
}

/** An `IN` filter for one metadata field. */
export function oneOf(field: string, values: readonly unknown[]): string {
  if (!values || values.length === 0) {
    throw new GoodMemFilterError('oneOf() needs at least one value.');
  }
  const casts = new Set<string>();
  const rendered = values.map((v) => {
    const [cast, literal] = render(v);
    casts.add(cast);
    return literal;
  });
  if (casts.size > 1) {
    throw new GoodMemFilterError('oneOf() values must all be of the same type.');
  }
  return `${accessor(field, [...casts][0])} IN (${rendered.join(', ')})`;
}

/** Combine filter expressions with `AND`. */
export function allOf(...expressions: Array<string | null | undefined>): string {
  const parts = expressions.filter((e): e is string => Boolean(e));
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return parts.map((p) => `(${p})`).join(' AND ');
}

/** Combine filter expressions with `OR`. */
export function anyOf(...expressions: Array<string | null | undefined>): string {
  const parts = expressions.filter((e): e is string => Boolean(e));
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return parts.map((p) => `(${p})`).join(' OR ');
}

/** An `AND` of equality filters built from a plain object. */
export function fromMapping(metadata: Record<string, unknown> | undefined): string {
  if (!metadata) return '';
  const keys = Object.keys(metadata).sort();
  if (keys.length === 0) return '';
  return allOf(...keys.map((k) => equals(k, metadata[k])));
}
