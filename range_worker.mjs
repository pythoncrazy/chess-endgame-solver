/* Read-only Cloudflare Worker cache proxy for immutable tablebase ranges.
 * Bind TB_BUCKET to an existing R2 bucket. No bucket creation/upload/deployment.
 * GET /range/<64-hex>.tbobj/<offset>/<length> returns a complete range-resource
 * with HTTP 200, which Cache API can store (a raw HTTP 206 cannot be put).
 * The Python/native/WASM prober remains responsible for chess rules/decoding.
 */
const inflight = new Map();
const MAX_RANGE = 1 << 20;
const TTL = 31536000;

function reply(message, status) {
  return new Response(message, {status, headers:{'Cache-Control':'no-store'}});
}

export default {
  async fetch(request, env, ctx) {
    if (request.method !== 'GET') return reply('GET required', 405);
    const url = new URL(request.url);
    const match = /^\/range\/([a-f0-9]{64}\.tbobj)\/(\d+)\/(\d+)$/.exec(url.pathname);
    if (!match || url.search) return reply('Invalid range resource', 400);
    const [,key,offsetText,lengthText] = match;
    const offset = Number(offsetText), length = Number(lengthText);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) ||
        offset < 0 || length <= 0 || length > MAX_RANGE || !Number.isSafeInteger(offset+length)) {
      return reply('Range out of bounds', 416);
    }
    // Normalize numeric spelling, giving one stable cache key for each range.
    const token = `${key}/${offset}/${length}`;
    const cacheKey = new Request(`${url.origin}/range/${token}`, {method:'GET'});
    const cache = caches.default;
    const hit = await cache.match(cacheKey);
    if (hit) return hit;
    let work = inflight.get(token);
    let owner = false;
    if (!work) {
      owner = true;
      work = (async () => {
        const object = await env.TB_BUCKET.get(key, {range:{offset,length}});
        if (!object) return {error:'Object missing', status:404};
        if (offset+length > object.size) {
          if (object.body?.cancel) await object.body.cancel();
          return {error:'Range out of bounds', status:416};
        }
        const bytes = await object.arrayBuffer();
        if (bytes.byteLength !== length) return {error:'Short origin read', status:502};
        return {bytes};
      })();
      inflight.set(token, work);
    }
    const release = () => {if (inflight.get(token) === work) inflight.delete(token);};
    let result;
    try {result = await work;} catch {if (owner) release();return reply('Origin read failed', 502);}
    if (result.error) {if (owner) release();return reply(result.error, result.status);}
    const response = new Response(result.bytes, {status:200, headers:{
      'Content-Type':'application/octet-stream',
      'Content-Length':String(length),
      'Cache-Control':`public, max-age=${TTL}, immutable`,
      'ETag':`"${token}"`
    }});
    if (owner) ctx.waitUntil(cache.put(cacheKey,response.clone()).finally(release));
    return response;
  }
};
