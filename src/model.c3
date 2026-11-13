module c3llm::model;
import std::math, std::thread, std::atomic, std::time::clock;
import c3llm::gguf, c3llm::quant;

excuse MISSING_TENSOR, BAD_SHAPE, UNSUPPORTED_TYPE, BAD_CONFIG, BAD_TOKEN, CONTEXT_FULL;

const sz MAX_SEQ_CAP = 4096; // the model declares 131072, which would need ~8.6 GB of KV cache

// A tensor checked to lie inside the file. rows == 1 for vectors.
struct Weight
{
	char* data;
	uint type;
	sz cols;
	sz rows;
	sz row_bytes;
}

struct Layer
{
	Weight attn_norm;
	Weight q;
	Weight k;
	Weight v;
	Weight o;
	Weight ffn_norm;
	Weight gate;
	Weight up;
	Weight down;
}

struct Model
{
	sz dim;
	sz hidden;
	sz n_heads;
	sz n_kv_heads;
	sz head_dim;
	sz kv_dim;
	sz vocab;
	sz max_seq;
	float eps;
	float[] rope; // rotation angle per dimension pair at position 1
	Weight embd;
	Weight out_norm;
	Layer[] layers;
	// run state, allocated once
	float[] x;
	float[] xb;
	float[] xb2;
	float[] q;
	float[] k;
	float[] v;
	float[] hb;
	float[] hb2;
	float[] att;
	float[] logits;
	float[] key_cache;   // [layer][pos][kv_dim]
	float[] value_cache;
}

fn Weight? weight(Gguf* g, String name, sz cols, sz rows)
{
	foreach (&t : g.tensors)
	{
		if (t.name != name) continue;
		if (t.dims[0] != (ulong)cols) return BAD_SHAPE~;
		if (rows == 1 ? t.n_dims != 1 : (t.n_dims != 2 || t.dims[1] != (ulong)rows)) return BAD_SHAPE~;
		sz rb = quant::row_bytes(t.type, cols);
		if (rb == 0) return UNSUPPORTED_TYPE~;
		if (rows == 1 && t.type != quant::F32) return UNSUPPORTED_TYPE~; // vectors are read as float*
		if (t.offset + (ulong)(rb * rows) > (ulong)g.data.len) return gguf::TRUNCATED~;
		return { .data = &g.data[(sz)t.offset], .type = t.type, .cols = cols, .rows = rows, .row_bytes = rb };
	}
	return MISSING_TENSOR~;
}

fn Weight? layer_weight(Gguf* g, sz layer, String name, sz cols, sz rows)
{
	return weight(g, string::tformat("blk.%d.%s.weight", layer, name), cols, rows);
}

fn Value? config(Gguf* g, String arch, String key)
{
	return g.get(string::tformat("%s.%s", arch, key));
}

fn Model? load(Gguf* g)
{
	Value arch_value = g.get("general.architecture")!;
	if (arch_value.type != gguf::T_STRING) return BAD_CONFIG~;
	String arch = (String)arch_value.raw;

	Model m;
	m.dim = (sz)config(g, arch, "embedding_length")!.as_uint();
	m.hidden = (sz)config(g, arch, "feed_forward_length")!.as_uint();
	m.n_heads = (sz)config(g, arch, "attention.head_count")!.as_uint();
	m.n_kv_heads = (sz)config(g, arch, "attention.head_count_kv")!.as_uint();
	sz n_layers = (sz)config(g, arch, "block_count")!.as_uint();
	sz ctx = (sz)config(g, arch, "context_length")!.as_uint();
	Value tokens = g.get("tokenizer.ggml.tokens")!;
	m.vocab = (sz)tokens.count;
	m.eps = 1e-5f;
	if (try e = config(g, arch, "attention.layer_norm_rms_epsilon")) m.eps = (float)e.as_float();
	double freq_base = 10000;
	if (try f = config(g, arch, "rope.freq_base")) freq_base = f.as_float();

	// The tensor shape checks below bound dim, hidden and vocab by the file size; these bound the rest.
	if (m.dim <= 0 || m.hidden <= 0 || m.vocab <= 0 || ctx <= 0) return BAD_CONFIG~;
	if (m.n_heads <= 0 || m.n_heads > m.dim || m.dim % m.n_heads != 0) return BAD_CONFIG~;
	if (m.n_kv_heads <= 0 || m.n_kv_heads > m.n_heads || m.n_heads % m.n_kv_heads != 0) return BAD_CONFIG~;
	if (n_layers <= 0 || n_layers > g.tensors.len) return BAD_CONFIG~;
	m.head_dim = m.dim / m.n_heads;
	if (m.head_dim % 2 != 0) return BAD_CONFIG~;
	m.kv_dim = m.head_dim * m.n_kv_heads;
	m.max_seq = ctx < MAX_SEQ_CAP ? ctx : MAX_SEQ_CAP;

	m.embd = weight(g, "token_embd.weight", m.dim, m.vocab)!;
	m.out_norm = weight(g, "output_norm.weight", m.dim, 1)!;
	m.layers = mem::new_array(Layer, n_layers);
	foreach (i, &l : m.layers)
	{
		l.attn_norm = layer_weight(g, i, "attn_norm", m.dim, 1)!;
		l.q = layer_weight(g, i, "attn_q", m.dim, m.dim)!;
		l.k = layer_weight(g, i, "attn_k", m.dim, m.kv_dim)!;
		l.v = layer_weight(g, i, "attn_v", m.dim, m.kv_dim)!;
		l.o = layer_weight(g, i, "attn_output", m.dim, m.dim)!;
		l.ffn_norm = layer_weight(g, i, "ffn_norm", m.dim, 1)!;
		l.gate = layer_weight(g, i, "ffn_gate", m.dim, m.hidden)!;
		l.up = layer_weight(g, i, "ffn_up", m.dim, m.hidden)!;
		l.down = layer_weight(g, i, "ffn_down", m.hidden, m.dim)!;
	}

	sz pairs = m.head_dim / 2;
	m.rope = mem::new_array(float, pairs);
	for (sz i = 0; i < pairs; i++) m.rope[i] = (float)math::pow(freq_base, -2.0 * i / m.head_dim);
	// Llama 3 stores per-frequency scaling factors; the angle is divided by them.
	if (try f = weight(g, "rope_freqs.weight", pairs, 1))
	{
		float* factors = (float*)f.data;
		for (sz i = 0; i < pairs; i++) m.rope[i] /= factors[i];
	}

	pool.acts = quant::new_acts(m.dim > m.hidden ? m.dim : m.hidden);
	m.x = mem::new_array(float, m.dim);
	m.xb = mem::new_array(float, m.dim);
	m.xb2 = mem::new_array(float, m.dim);
	m.q = mem::new_array(float, m.dim);
	m.k = mem::new_array(float, m.kv_dim);
	m.v = mem::new_array(float, m.kv_dim);
	m.hb = mem::new_array(float, m.hidden);
	m.hb2 = mem::new_array(float, m.hidden);
	m.att = mem::new_array(float, m.n_heads * m.max_seq);
	m.logits = mem::new_array(float, m.vocab);
	m.key_cache = mem::new_array(float, n_layers * m.max_seq * m.kv_dim);
	m.value_cache = mem::new_array(float, n_layers * m.max_seq * m.kv_dim);
	return m;
}

<*
 Worker threads that share one parallel loop at a time. Loops arrive microseconds apart during a forward
 pass, far closer than a sleeping thread can wake, so idle workers spin for a while and only go to sleep
 when nothing has come for about a millisecond.

 Every thread starts on its own contiguous share of the items, so each core reads the weights as one steady
 stream, and helps with the other shares once its own is done.
*>
alias Job = fn void(void* ctx, sz first, sz end);

// One cache line each, so threads do not fight over each other's counters.
struct Share @align(64)
{
	sz next; // first unclaimed item; atomic
	sz end;
}

const sz MAX_THREADS = 64;

struct Pool
{
	Mutex mu;          // only guards going to sleep and waking up
	ConditionVariable work;
	sz workers;
	ulong generation;  // bumped once per loop; atomic
	sz pending;        // workers that have not finished the current loop; atomic
	Job job;
	void* ctx;
	sz chunk;          // items claimed at a time
	Share[MAX_THREADS] shares;
	Acts acts;         // activations in integer form for the kernels, rebuilt for each matrix product
}

const sz ROWS_PER_CHUNK = 8;
const int SPINS_BEFORE_SLEEP = 50_000;
Pool pool;
long[16] matvec_ns; // wall time spent in matvec, by tensor type, for the bench command

// A waiting thread mostly pauses, but now and then offers its CPU to a thread that was pushed off one.
// Without that, a worker the OS has descheduled mid-chunk can be starved by the threads waiting for it.
macro void wait_a_moment(int spins)
{
	if (spins % 256 == 0)
	{
		thread::yield();
	}
	else
	{
		asm("pause");
	}
}

// Works through one share, then the others in turn.
fn void run_shares(sz mine)
{
	sz threads = pool.workers + 1;
	for (sz i = 0; i < threads; i++)
	{
		Share* s = &pool.shares[(mine + i) % threads];
		while (@atomic_load(s.next) < s.end)
		{
			sz first = atomic::fetch_add(&s.next, pool.chunk);
			if (first >= s.end) break;
			pool.job(pool.ctx, first, first + pool.chunk < s.end ? first + pool.chunk : s.end);
		}
	}
}

fn int worker(void* arg)
{
	sz index = (sz)(iptr)arg;
	ulong seen;
	while (true)
	{
		int spins;
		while (@atomic_load(pool.generation) == seen)
		{
			if (++spins < SPINS_BEFORE_SLEEP)
			{
				wait_a_moment(spins);
				continue;
			}
			pool.mu.lock();
			while (@atomic_load(pool.generation) == seen) pool.work.wait(&pool.mu);
			pool.mu.unlock();
		}
		// parallel() waits for every worker before posting again, so this is exactly one loop ahead
		seen++;
		run_shares(index);
		atomic::fetch_sub(&pool.pending, 1);
	}
}

// threads counts the calling thread, so 1 means no workers.
fn void? start_threads(sz threads)
{
	pool.mu.init()!;
	pool.work.init()!;
	for (sz i = 1; i < threads && i < MAX_THREADS; i++)
	{
		Thread t;
		t.create(&worker, (void*)(iptr)i)!;
		t.detach();
		pool.workers++;
	}
}

// Calls job over [0, count) in pieces of at most chunk items, on all threads, and returns when all are done.
fn void parallel(Job job, void* ctx, sz count, sz chunk)
{
	sz threads = pool.workers + 1;
	if (threads == 1 || count <= chunk)
	{
		job(ctx, 0, count);
		return;
	}
	pool.job = job;
	pool.ctx = ctx;
	pool.chunk = chunk;
	sz share = (count / threads / chunk + 1) * chunk;
	for (sz t = 0; t < threads; t++)
	{
		sz first = t * share < count ? t * share : count;
		pool.shares[t].end = first + share < count ? first + share : count;
		@atomic_store(pool.shares[t].next, first);
	}
	@atomic_store(pool.pending, pool.workers);
	// the bump happens under the mutex so a worker about to sleep cannot miss it
	pool.mu.lock();
	@atomic_store(pool.generation, pool.generation + 1);
	pool.work.broadcast();
	pool.mu.unlock();
	run_shares(0);
	for (int spins = 1; @atomic_load(pool.pending) > 0; spins++) wait_a_moment(spins);
}

struct MatJob
{
	Weight* w;
	float* x;
	float* out;
}

fn void rows_job(void* ctx, sz first, sz end)
{
	MatJob* j = ctx;
	Weight* w = j.w;
	for (sz r = first; r < end; r++) j.out[r] = quant::dot(w.type, w.data + r * w.row_bytes, j.x, &pool.acts, w.cols);
}

fn void matvec(Weight* w, float* x, float* out)
{
	Clock start = clock::now();
	defer matvec_ns[w.type] += (long)start.to_now();
	if (w.type != quant::F32) pool.acts.set(x, w.cols, w.type == quant::Q6_K ? 256 : 32);
	MatJob job = { w, x, out };
	parallel(&rows_job, &job, w.rows, ROWS_PER_CHUNK);
}

fn void rmsnorm(float* out, float* x, Weight* w, float eps)
{
	float* gain = (float*)w.data;
	float ss = quant::dot_f32(x, x, w.cols);
	float scale = 1.0f / math::sqrt(ss / w.cols + eps);
	for (sz i = 0; i < w.cols; i++) out[i] = x[i] * scale * gain[i];
}

fn void Model.rotate(&self, float* v, sz heads, sz pos)
{
	for (sz i = 0; i < self.head_dim / 2; i++)
	{
		float angle = pos * self.rope[i];
		float c = math::cos(angle);
		float s = math::sin(angle);
		for (sz h = 0; h < heads; h++)
		{
			float* p = v + h * self.head_dim + 2 * i;
			float a = p[0];
			float b = p[1];
			p[0] = a * c - b * s;
			p[1] = a * s + b * c;
		}
	}
}

struct AttJob
{
	Model* m;
	sz layer;
	sz pos;
}

// Attention for heads [first, end): each head scores every cached key, then mixes the cached values.
fn void heads_job(void* ctx, sz first, sz end)
{
	AttJob* j = ctx;
	Model* m = j.m;
	sz hd = m.head_dim;
	float att_scale = 1.0f / math::sqrt((float)hd);
	float* kc = &m.key_cache[j.layer * m.max_seq * m.kv_dim];
	float* vc = &m.value_cache[j.layer * m.max_seq * m.kv_dim];
	for (sz h = first; h < end; h++)
	{
		float* att = &m.att[h * m.max_seq]; // a scratch row per head, since heads run on different threads
		float* qh = &m.q[h * hd];
		sz kv_off = h / (m.n_heads / m.n_kv_heads) * hd;
		float max = -float::inf;
		for (sz t = 0; t <= j.pos; t++)
		{
			att[t] = quant::dot_f32(qh, kc + t * m.kv_dim + kv_off, hd) * att_scale;
			if (att[t] > max) max = att[t];
		}
		float total = 0;
		for (sz t = 0; t <= j.pos; t++)
		{
			att[t] = math::exp(att[t] - max);
			total += att[t];
		}
		float* out = &m.xb2[h * hd];
		for (sz i = 0; i < hd; i++) out[i] = 0;
		for (sz t = 0; t <= j.pos; t++)
		{
			float a = att[t] / total;
			float* vt = vc + t * m.kv_dim + kv_off;
			for (sz i = 0; i < hd; i++) out[i] += a * vt[i];
		}
	}
}

// hb = silu(hb) * hb2 over [first, end)
fn void silu_job(void* ctx, sz first, sz end)
{
	Model* m = ctx;
	for (sz i = first; i < end; i++)
	{
		float g = m.hb[i];
		m.hb[i] = g / (1.0f + math::exp(-g)) * m.hb2[i];
	}
}

// Runs one token through the network. Logits land in self.logits when want_logits is set.
fn void? Model.forward(&self, uint token, sz pos, bool want_logits)
{
	if (token >= self.vocab) return BAD_TOKEN~;
	if (pos < 0 || pos >= self.max_seq) return CONTEXT_FULL~;
	sz dim = self.dim;
	float* x = self.x.ptr;
	float* xb = self.xb.ptr;

	quant::dequant(self.embd.type, self.embd.data + token * self.embd.row_bytes, x, dim);

	foreach (li, &l : self.layers)
	{
		// attention
		rmsnorm(xb, x, &l.attn_norm, self.eps);
		matvec(&l.q, xb, self.q.ptr);
		matvec(&l.k, xb, self.k.ptr);
		matvec(&l.v, xb, self.v.ptr);
		self.rotate(self.q.ptr, self.n_heads, pos);
		self.rotate(self.k.ptr, self.n_kv_heads, pos);
		sz cache = (li * self.max_seq + pos) * self.kv_dim;
		mem::copy(&self.key_cache[cache], self.k.ptr, self.kv_dim * 4);
		mem::copy(&self.value_cache[cache], self.v.ptr, self.kv_dim * 4);
		AttJob heads = { self, li, pos };
		parallel(&heads_job, &heads, self.n_heads, 1);
		matvec(&l.o, self.xb2.ptr, xb);
		for (sz i = 0; i < dim; i++) x[i] += xb[i];

		// feed-forward: down(silu(gate(x)) * up(x))
		rmsnorm(xb, x, &l.ffn_norm, self.eps);
		matvec(&l.gate, xb, self.hb.ptr);
		matvec(&l.up, xb, self.hb2.ptr);
		parallel(&silu_job, self, self.hidden, 256);
		matvec(&l.down, self.hb.ptr, xb);
		for (sz i = 0; i < dim; i++) x[i] += xb[i];
	}

	if (!want_logits) return;
	rmsnorm(xb, x, &self.out_norm, self.eps);
	matvec(&self.embd, xb, self.logits.ptr); // output weights are tied to the embedding table
}