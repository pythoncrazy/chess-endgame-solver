import assert from 'node:assert/strict';
import fs from 'node:fs';
import worker from './range_worker.mjs';
const held = new Map();
globalThis.caches = {default:{
  async match(request) {return held.get(request.url)?.clone();},
  async put(request,response) {assert.equal(response.status,200);held.set(request.url,response.clone());}
}};
const original = Uint8Array.from({length:4096},(_,i)=>i%251);
let originReads=0, fail=false;
const key = 'a'.repeat(64)+'.tbobj';
const env={TB_BUCKET:{async get(name,{range}) {
  originReads++;await new Promise(resolve=>setTimeout(resolve,5));
  if (fail) throw new Error('Injected origin failure');
  if (name !== key) return null;
  const bytes=original.slice(range.offset,range.offset+range.length);
  return {size:original.length,async arrayBuffer(){return bytes.buffer;}};
}}};
let writes=[];
const ctx={waitUntil(promise){writes.push(promise);}};
const request=new Request(`https://example.test/range/${key}/127/1024`);
const first=await worker.fetch(request,env,ctx);
assert.equal(first.status,200);assert.deepEqual(new Uint8Array(await first.arrayBuffer()),original.slice(127,1151));
await Promise.all(writes);writes=[];
const second=await worker.fetch(request,env,ctx);
assert.equal(second.status,200);assert.equal(originReads,1);
held.clear();originReads=0;
const simultaneous=await Promise.all(Array.from({length:16},()=>worker.fetch(request,env,ctx)));
assert(simultaneous.every(r=>r.status===200));assert.equal(originReads,1);
await Promise.all(writes);writes=[];
assert.equal((await worker.fetch(new Request('https://example.test/range/no/0/2'),env,ctx)).status,400);
assert.equal((await worker.fetch(new Request(`https://example.test/range/${key}/0/1048577`),env,ctx)).status,416);
assert.equal((await worker.fetch(new Request(`https://example.test/range/${key}/4000/1000`),env,ctx)).status,416);
held.clear();fail=true;
assert.equal((await worker.fetch(request,env,ctx)).status,502);fail=false;
assert.equal((await worker.fetch(request,env,ctx)).status,200);
await Promise.all(writes);
const result={status:'pass',cached_repeat_origin_reads:0,concurrent_callers:16,
  simultaneous_identical_origin_reads:1,exact_range_bytes:true,failure_retry_passed:true,
  deployed:false,live_r2_tested:false,note:'Mock Cache API and mock R2 binding; not a Cloudflare latency measurement.'};
fs.writeFileSync('range_worker_validation.json',JSON.stringify(result,null,2)+'\n');
process.stdout.write(JSON.stringify(result,null,2)+'\n');
