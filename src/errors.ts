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
