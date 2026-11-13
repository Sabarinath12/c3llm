// Side-by-side comparison of c3llm and llama.cpp on the same prompt, both greedy.
// Run: node compare/server.js   then open http://127.0.0.1:8790
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');
const MODEL = path.join(ROOT, 'models', 'Llama-3.2-1B-Instruct-Q4_0.gguf');
const C3LLM = path.join(ROOT, 'build', 'c3llm.exe');
const LLAMA_SERVER = path.join(ROOT, 'tools', 'llama.cpp', 'llama-server.exe');
const PORT = 8790;
const LLAMA_PORT = 8791;
const LLAMA_URL = `http://127.0.0.1:${LLAMA_PORT}`;

// Both engines get the same raw text, so neither applies its own chat template.
const chatPrompt = (message) =>
  `<|start_header_id|>user<|end_header_id|>\n\n${message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;

const llama = spawn(LLAMA_SERVER, ['-m', MODEL, '--host', '127.0.0.1', '--port', String(LLAMA_PORT), '-c', '4096'], {
  stdio: 'ignore',
});
llama.on('exit', (code) => console.log(`llama-server exited (${code})`));
const stop = () => { llama.kill(); process.exit(); };
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
process.on('exit', () => llama.kill());

function runC3(prompt, maxTokens, send, signal) {
  return new Promise((resolve) => {
    const child = spawn(C3LLM, ['generate', MODEL, String(maxTokens), prompt]);
    signal.addEventListener('abort', () => child.kill());
    let stderr = '';
    // Windows text-mode stdout turns \n into \r\n; drop the \r so the two outputs compare equal
    child.stdout.on('data', (d) => send({ text: d.toString('utf8').replaceAll('\r', '') }));
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('error', (e) => resolve({ error: `cannot start c3llm: ${e.message}` }));
    child.on('close', (code) => {
      const m = stderr.match(/prompt: (\d+) tokens, ([\d.]+) tok\/s \| generation: (\d+) tokens, ([\d.]+) tok\/s/);
      if (code !== 0 || !m) return resolve({ error: `c3llm failed (exit ${code}) ${stderr.trim()}` });
      resolve({ promptTokens: +m[1], promptTps: +m[2], genTokens: +m[3], genTps: +m[4] });
    });
  });
}

async function runLlama(prompt, maxTokens, send, signal) {
  const res = await fetch(`${LLAMA_URL}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // samplers reduced to top_k 1 = greedy; cache off so the prompt is timed every run, like c3llm
    body: JSON.stringify({ prompt, n_predict: maxTokens, temperature: 0, top_k: 1, stream: true, cache_prompt: false }),
    signal,
  });
  if (!res.ok) return { error: `llama-server answered ${res.status}` };
  let buffer = '';
  let stats = { error: 'llama-server ended without timings' };
  for await (const chunk of res.body) {
    buffer += Buffer.from(chunk).toString('utf8');
    let nl;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));
      if (event.content) send({ text: event.content });
      if (event.stop && event.timings) {
        const t = event.timings;
        stats = { promptTokens: t.prompt_n, promptTps: t.prompt_per_second, genTokens: t.predicted_n, genTps: t.predicted_per_second };
      }
    }
  }
  return stats;
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (d) => {
      body += d;
      if (body.length > 20000) reject(new Error('request too large'));
    });
    req.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
  });
}

http.createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return fs.createReadStream(path.join(__dirname, 'index.html')).pipe(res);
  }
  if (req.method === 'GET' && req.url === '/api/ready') {
    const ok = await fetch(`${LLAMA_URL}/health`).then((r) => r.ok, () => false);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ready: ok }));
  }
  if (req.method === 'POST' && req.url === '/api/run') {
    let input;
    try { input = await readJson(req); } catch { res.writeHead(400); return res.end('bad request'); }
    const { engine, message } = input;
    const maxTokens = Number(input.maxTokens);
    if (!['c3', 'llama'].includes(engine) || typeof message !== 'string' || !message.trim() || message.length > 4000
        || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
      res.writeHead(400);
      return res.end('bad request');
    }
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' });
    const send = (obj) => res.write(JSON.stringify(obj) + '\n');
    const abort = new AbortController();
    res.on('close', () => abort.abort()); // browser went away: stop generating
    const run = engine === 'c3' ? runC3 : runLlama;
    const stats = await run(chatPrompt(message), maxTokens, send, abort.signal).catch((e) => ({ error: e.message }));
    send({ done: true, ...stats });
    return res.end();
  }
  res.writeHead(404);
  res.end('not found');
}).listen(PORT, '127.0.0.1', () => console.log(`compare page: http://127.0.0.1:${PORT}`));
