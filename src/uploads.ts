/**
 * Confines model-supplied upload paths to one directory.
 *
 * A path that reaches an upload call is chosen by a language model. Without a
 * boundary, `/etc/hostname` is a valid argument and the file is read and
 * uploaded -- which 0.1.2 did, byte for byte. Uploads therefore exist only
 * when the developer configures `uploadDir`, and every path is resolved --
 * symlinks included -- before it is compared against that directory.
 */

import { accessSync, constants, lstatSync, realpathSync, statSync } from 'node:fs';
import { isAbsolute, resolve, sep } from 'node:path';

/** Thrown when a requested upload path is not allowed. */
export class GoodMemUploadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GoodMemUploadError';
  }
}

/**
 * Resolve a model-supplied path inside the configured upload directory.
 *
 * @throws {GoodMemUploadError} if uploads are not configured, if the path
 * escapes the upload directory (directly, via `..` or through a symlink), or
 * if it is not a readable regular file.
 */
export function resolveUploadPath(path: string, uploadDir: string | undefined): string {
  if (!uploadDir) {
    throw new GoodMemUploadError(
      'File uploads are disabled. Configure the plugin with uploadDir=<directory> ' +
        'to enable them; only files inside that directory can be uploaded.'
    );
  }
  let root: string;
  try {
    root = realpathSync(resolve(uploadDir));
  } catch {
    root = resolve(uploadDir);
  }

  const candidate = isAbsolute(path) ? path : resolve(root, path);
  let resolved: string;
  try {
    // realpath follows symlinks, so a link inside the directory that points
    // outside it is caught by the comparison below rather than followed.
    resolved = realpathSync(candidate);
  } catch {
    throw new GoodMemUploadError(`${JSON.stringify(path)} does not exist.`);
  }

  if (resolved !== root && !resolved.startsWith(root + sep)) {
    throw new GoodMemUploadError(
      `${JSON.stringify(path)} is outside the upload directory ${JSON.stringify(root)}.`
    );
  }
  if (!lstatSync(resolved).isFile() && !statSync(resolved).isFile()) {
    throw new GoodMemUploadError(`${JSON.stringify(path)} is not a regular file.`);
  }
  try {
    accessSync(resolved, constants.R_OK);
  } catch {
    throw new GoodMemUploadError(`${JSON.stringify(path)} is not readable.`);
  }
  return resolved;
}
