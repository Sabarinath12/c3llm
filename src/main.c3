module c3llm;
import std::io, std::math, std::collections::list, std::time::clock, std::os;
import c3llm::gguf, c3llm::quant, c3llm::model, c3llm::tokenizer, c3llm::sampler;

fn void print_value(Value* v)
{
	switch (v.type)
	{
		case gguf::T_STRING:
			String s = (String)v.raw;
			if (s.len > 60) { io::printfn("%s... (%d bytes)", s[:60], s.len); return; }
			io::printfn("%s", s);
		case gguf::T_ARRAY:
			io::printfn("[array of %d, element type %d]", v.count, v.elem_type);
		case gguf::T_FLOAT32: case gguf::T_FLOAT64:
			io::printfn("%s", v.as_float());
		case gguf::T_INT8: case gguf::T_INT16: case gguf::T_INT32: case gguf::T_INT64:
			io::printfn("%d", (long)v.as_uint());
		default:
			io::printfn("%d", v.as_uint());
	}
}

fn void inspect(Gguf* g)
{
	io::printfn("GGUF v%d, %d metadata keys, %d tensors, alignment %d\n", g.version, g.meta.len, g.tensors.len, g.alignment);
	foreach (&m : g.meta)
	{
		io::printf("%s = ", m.key);
		print_value(&m.value);
	}
	io::printn();
	foreach (&t : g.tensors)
	{
		io::printf("%-40s %-6s @%-12d [", t.name, gguf::type_name(t.type), t.offset);
		for (uint d = 0; d < t.n_dims; d++) io::printf(d ? ", %d" : "%d", t.dims[d]);
		io::printn("]");
	}
}

char[256] buf;
sz n;

fn void put_bytes(void* p, sz len) { mem::copy(&buf[n], p, len); n += len; }
fn void put32(uint v) { put_bytes(&v, 4); }
fn void put64(ulong v) { put_bytes(&v, 8); }
fn void put_str(String s) { put64(s.len); put_bytes(s.ptr, s.len); }

// Builds a tiny GGUF in memory and checks the parser against it.
fn void selftest()
{
	put_bytes("GGUF".ptr, 4); put32(3); put64(1); put64(3);
	put_str("general.alignment"); put32(gguf::T_UINT32); put32(64);
	put_str("general.architecture"); put32(gguf::T_STRING); put_str("llama");
	put_str("list"); put32(gguf::T_ARRAY); put32(gguf::T_STRING); put64(2); put_str("a"); put_str("bc");
	put_str("w"); put32(2); put64(3); put64(5); put32(2); put64(32);
	sz header_end = n;

	Gguf g = gguf::parse(buf[..])!!;
	assert(g.version == 3 && g.alignment == 64 && g.meta.len == 3 && g.tensors.len == 1);
	assert((String)g.get("general.architecture")!!.raw == "llama");
	assert(g.get("list")!!.count == 2);
	assert(@catch(g.get("missing")) == gguf::NOT_FOUND);
	Tensor t = g.tensors[0];
	assert(t.name == "w" && t.n_dims == 2 && t.dims[1] == 5 && t.type == 2);
	assert(header_end == 190 && t.offset == 192 + 32); // base rounds up to 192, plus the relative offset

	assert(@catch(gguf::parse(buf[:header_end - 4])) == gguf::TRUNCATED);
	buf[0] = 'X';
	assert(@catch(gguf::parse(buf[..])) == gguf::BAD_MAGIC);

	// half floats: 1.0, -2.0, smallest subnormal
	char[2] one = { 0x00, 0x3C };
	char[2] neg2 = { 0x00, 0xC0 };
	char[2] tiny = { 0x01, 0x00 };
	assert(quant::f16(&one) == 1.0f && quant::f16(&neg2) == -2.0f && quant::f16(&tiny) == 5.9604645e-8f);

	// Q4_0 block with scale 1: byte 0x3A holds low nibble 10 (element 0) and high nibble 3 (element 16)
	char[18] q40;
	q40[1] = 0x3C; q40[2] = 0x3A;
	float[256] w;
	quant::dequant(quant::Q4_0, &q40, &w, 32);
	assert(w[0] == 2.0f && w[16] == -5.0f && w[1] == -8.0f);

	// dot() must agree with dequantize-then-multiply for every quantized type
	char[210] blk;
	float[256] x;
	foreach (i, &b : blk) *b = (char)(i * 37 + 11);
	blk[208] = 0x00; blk[209] = 0x3C; // Q6_K scale = 1.0
	// whole numbers from -127 to 127 survive the int8 rounding exactly, so the kernels must match closely
	foreach (i, &v : x) *v = (float)((int)i * 5 % 255 - 127);
	Acts acts = quant::new_acts(256);
	uint[3] types = { quant::Q4_0, quant::Q4_1, quant::Q6_K }; // arbitrary bytes are not finite as F32
	foreach (type : types)
	{
		sz len = type == quant::Q6_K ? 256 : 32;
		quant::dequant(type, &blk, &w, len);
		acts.set(&x, len, len);
		float want = 0;
		for (sz i = 0; i < len; i++) want += w[i] * x[i];
		float got = quant::dot(type, &blk, &x, &acts, len);
		assert(math::abs(got - want) <= 1e-3f * math::abs(want) + 1e-6f, "dot mismatch for type %d: %s vs %s", type, got, want);
	}
	// Q6_K element 0: low nibble of ql[0], low 2 bits of qh[0], scale[0]
	quant::dequant(quant::Q6_K, &blk, &w, 256);
	int q = (int)((blk[0] & 0xF) | ((blk[128] & 3) << 4)) - 32;
	assert(w[0] == (float)((ichar)blk[192] * q));

	// byte <-> character mapping: space is U+0120, newline is U+010A, and every byte round-trips
	assert(tokenizer::byte_cp(' ') == 0x120 && tokenizer::byte_cp('\n') == 0x10A && tokenizer::byte_cp('A') == 'A');
	for (uint b = 0; b < 256; b++) assert(tokenizer::cp_byte(tokenizer::byte_cp((char)b)) == (int)b);

	// pre-tokenizer split
	String text = "Hello, world!  I'm 12345\n\nok";
	String[*] want_words = { "Hello", ",", " world", "!", " ", " I", "'m", " ", "123", "45", "\n\n", "ok" };
	sz at;
	foreach (word : want_words)
	{
		sz end = tokenizer::word_end(text, at);
		assert(text[at:end - at] == word, "split gave '%s', wanted '%s'", text[at:end - at], word);
		at = end;
	}
	assert(at == text.len);

	// BPE merges in rank order, special tokens bypass BPE, decode restores the bytes
	String[*] vocab = { "h", "e", "l", "o", "\u0120", "he", "ll", "hell", "hello", "\u0120hello", "w", "<|x|>" };
	String[*] merges = { "h e", "l l", "he ll", "hell o", "\u0120 hello" };
	Tokenizer tok = tokenizer::create(&vocab, &merges);
	List{uint} ids;
	ids.init(mem);
	tok.encode("hello hello<|x|>wo", &ids);
	uint[*] want_ids = { 8, 9, 11, 10, 3 };
	assert(ids.len() == want_ids.len);
	foreach (i, id : want_ids) assert(ids[i] == id, "token %d is %d, wanted %d", i, ids[i], id);
	char[16] piece;
	assert(tok.decode(9, &piece) == " hello" && tok.decode(11, &piece) == "<|x|>" && tok.decode(99, &piece) == "");

	// sampler: temperature 0, top-k 1 and a tiny top-p all reduce to the most likely token
	float[*] logits = { 1.0f, 5.0f, 5.0f, -2.0f, 4.9f };
	Sampler greedy = { .state = 1 };
	Sampler k1 = { .temperature = 1, .top_k = 1, .top_p = 1, .state = 1 };
	Sampler p0 = { .temperature = 1, .top_k = 40, .top_p = 0.01f, .state = 1 };
	assert(greedy.pick(&logits) == 1 && k1.pick(&logits) == 1 && p0.pick(&logits) == 1);
	// top-k 3 only ever returns the three best, and reaches each of them
	Sampler k3 = { .temperature = 1, .top_k = 3, .top_p = 1, .state = 12345 };
	int[5] seen;
	for (int i = 0; i < 300; i++) seen[k3.pick(&logits)]++;
	assert(seen[0] == 0 && seen[3] == 0 && seen[1] > 50 && seen[2] > 50 && seen[4] > 50);
	assert(is_special("<|eot_id|>") && !is_special("<|") && !is_special("hello"));
	io::printn("selftest ok");
}

fn void show_tokens(Tokenizer* tok, String text)
{
	List{uint} ids;
	ids.init(mem);
	tok.encode(text, &ids);
	char[512] buf;
	foreach (id : ids.array_view()) io::printfn("%-8d %s", id, tok.vocab[id]);
	foreach (id : ids.array_view()) io::print(tok.decode(id, &buf));
	io::printn();
}

fn bool is_special(String piece) => piece.starts_with("<|") && piece.ends_with("|>");

fn void? feed(Model* m, uint[] ids, sz* pos, bool want_logits)
{
	foreach (i, id : ids)
	{
		m.forward(id, *pos, want_logits && i == ids.len - 1)!;
		(*pos)++;
	}
}

// Streams tokens until a special token, the limit, or a full context. Returns how many were printed.
fn sz? respond(Model* m, Tokenizer* tok, Sampler* s, sz* pos, sz limit)
{
	char[512] buf;
	sz made;
	while (made < limit)
	{
		uint id = s.pick(m.logits);
		if (is_special(tok.vocab[id])) break;
		io::print(tok.decode(id, &buf));
		(void)io::stdout().flush();
		made++;
		m.forward(id, *pos, true)!;
		(*pos)++;
	}
	return made;
}

fn void print_speed(sz prompt, double prompt_sec, sz made, double gen_sec)
{
	io::printn("\n");
	// stderr, so programs reading the reply from stdout get only the reply
	io::eprintfn("[prompt: %d tokens, %.1f tok/s | generation: %d tokens, %.1f tok/s]",
		prompt, prompt / prompt_sec, made, made ? made / gen_sec : 0.0);
}

// Greedy generation. The prompt may contain special tokens such as <|eot_id|>; BOS is added if missing.
fn int generate(Gguf* g, Tokenizer* tok, Model* m, String count, String prompt)
{
	int? n_new = count.to_int();
	if (catch err = n_new) { io::printfn("bad token count: %s", count); return 1; }

	List{uint} ids;
	ids.init(mem);
	if (try bos = g.get("tokenizer.ggml.bos_token_id")) ids.push((uint)bos.as_uint());
	tok.encode(prompt, &ids);
	if (ids.len() > 1 && ids[0] == ids[1]) ids.remove_first(); // prompt already started with BOS

	sz pos;
	Sampler greedy;
	Clock timer = clock::now();
	if (catch err = feed(m, ids.array_view(), &pos, true)) { io::printfn("forward failed: %s", err); return 1; }
	double prompt_sec = timer.mark().to_sec();
	sz? made = respond(m, tok, &greedy, &pos, n_new);
	if (catch err = made) { io::printfn("\nforward failed: %s", err); return 1; }
	print_speed(ids.len(), prompt_sec, made, timer.mark().to_sec());
	return 0;
}

// Feeds one prompt and streams the reply.
fn sz? turn(Model* m, Tokenizer* tok, Sampler* s, uint[] ids, sz* pos, Clock* timer, double* prompt_sec)
{
	feed(m, ids, pos, true)!;
	*prompt_sec = timer.mark().to_sec();
	return respond(m, tok, s, pos, m.max_seq);
}

// Times forward passes on one repeated token and shows where the time goes.
fn int bench(Model* m, sz threads)
{
	const sz WARMUP = 4; // the first passes also page the model file into memory
	const sz TOKENS = 32;
	Clock timer;
	for (sz i = 0; i < WARMUP + TOKENS; i++)
	{
		if (i == WARMUP)
		{
			model::matvec_ns = {};
			timer = clock::now();
		}
		if (catch err = m.forward(1, i, true)) { io::printfn("forward failed: %s", err); return 1; }
	}
	double ms = timer.mark().to_sec() * 1000 / TOKENS;
	double in_matvec = 0;
	io::printfn("%d threads: %.1f ms per token, %.1f tok/s", threads, ms, 1000 / ms);
	foreach (type, ns : model::matvec_ns)
	{
		if (!ns) continue;
		double part = ns / 1e6 / TOKENS;

		in_matvec += part;
		io::printfn("  %-5s matrix products  %6.1f ms  %4.1f%%", gguf::type_name((uint)type), part, 100 * part / ms);
	}
	io::printfn("  everything else        %6.1f ms  %4.1f%%", ms - in_matvec, 100 * (ms - in_matvec) / ms);

	// The same kernels on a slice small enough to stay in cache: compute speed without memory traffic.
	// Compare with the weights/s implied above to see how much the full pass is held back by memory.
	Weight[2] slices = { m.layers[0].q, m.embd };
	float[] out = mem::new_array(float, 768);
	foreach (&w : slices)
	{
		w.rows = 768; // about 1 MB: stays in the shared cache, and is enough rows to give every thread work
		const sz REPEATS = 2_000;
		timer = clock::now();
		for (sz i = 0; i < REPEATS; i++) model::matvec(w, m.xb.ptr, out.ptr);
		double sec = timer.mark().to_sec();
		double weights = (double)REPEATS * w.rows * w.cols;
		io::printfn("  %-5s kernel in cache: %.1f billion weights/s, %.1f GB/s", gguf::type_name(w.type), weights / sec / 1e9,
			(double)REPEATS * w.rows * w.row_bytes / sec / 1e9);
	}
	return 0;
}

// Interactive chat in the Llama 3 turn format. The KV cache carries the conversation between turns.
fn int chat(Gguf* g, Tokenizer* tok, Model* m, float temperature)
{
	Sampler s = { .temperature = temperature, .top_k = 40, .top_p = 0.95f, .state = (ulong)clock::now() | 1 };
	List{uint} ids;
	ids.init(mem);
	if (try bos = g.get("tokenizer.ggml.bos_token_id")) ids.push((uint)bos.as_uint());
	sz pos;
	io::printn("Type a message and press Enter. An empty line exits.");
	while (true)
	{
		io::print("\n> ");
		(void)io::stdout().flush();
		String? line = io::treadline();
		if (catch err = line) break;
		String text = line.trim();
		if (!text.len) break;

		tok.encode("<|start_header_id|>user<|end_header_id|>\n\n", &ids);
		tok.encode_plain(text, &ids); // plain: typed text cannot inject special tokens
		tok.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", &ids);

		Clock timer = clock::now();
		double prompt_sec;
		sz? made = turn(m, tok, &s, ids.array_view(), &pos, &timer, &prompt_sec);
		if (catch err = made)
		{
			if (err == model::CONTEXT_FULL) { io::printfn("\n[context of %d tokens is full, restart to continue]", m.max_seq); return 0; }
			io::printfn("\nforward failed: %s", err);
			return 1;
		}
		print_speed(ids.len(), prompt_sec, made, timer.mark().to_sec());
		ids.clear();
		tok.encode("<|eot_id|>", &ids); // closes the assistant turn at the start of the next prompt
	}
	return 0;
}

fn int main(String[] args)
{
	if (args.len == 2 && args[1] == "selftest") { selftest(); return 0; }
	String cmd = args.len > 1 ? args[1] : "";
	bool ok = (cmd == "inspect" && args.len == 3) || (cmd == "tokenize" && args.len == 4)
		|| (cmd == "generate" && (args.len == 5 || args.len == 6)) || (cmd == "chat" && (args.len == 3 || args.len == 4))
		|| (cmd == "bench" && (args.len == 3 || args.len == 4));
	if (!ok)
	{
		io::printn("usage: c3llm chat <model.gguf> [temperature]\n       c3llm generate <model.gguf> <new tokens> <prompt> [threads]\n       c3llm bench <model.gguf> [threads]\n       c3llm tokenize <model.gguf> <text>\n       c3llm inspect <model.gguf>\n       c3llm selftest");
		return 1;
	}
	Gguf? g = gguf::open(args[2]);
	if (catch err = g)
	{
		io::printfn("cannot read %s: %s", args[2], err);
		return 1;
	}
	if (cmd == "inspect") { inspect(&g); return 0; }
	Tokenizer? tok = tokenizer::load(&g);
	if (catch err = tok) { io::printfn("cannot load tokenizer: %s", err); return 1; }
	if (cmd == "tokenize") { show_tokens(&tok, args[3]); return 0; }

	// The weights stream from memory faster than the cores can use them until nearly every logical CPU is
	// busy, but the last one or two are worth more left free for the rest of the system.
	// shortcut: tuned on one 6-core/12-thread laptop; measure with the bench command on other machines.
	sz cpus = os::num_cpu();
	sz threads = cpus > 2 ? cpus - (cpus >= 12 ? 2 : 1) : 1;
	sz thread_arg = cmd == "generate" ? 5 : cmd == "bench" ? 3 : args.len;
	if (args.len > thread_arg)
	{
		int? t = args[thread_arg].to_int();
		if (catch err = t) { io::printfn("bad thread count: %s", args[thread_arg]); return 1; }
		if (t < 1 || t > model::MAX_THREADS) { io::printfn("thread count must be 1 to %d", model::MAX_THREADS); return 1; }
		threads = t;
	}
	if (catch err = model::start_threads(threads)) { io::printfn("cannot start threads: %s", err); return 1; }
	Model? m = model::load(&g);
	if (catch err = m) { io::printfn("cannot load model: %s", err); return 1; }
	if (cmd == "generate") return generate(&g, &tok, &m, args[3], args[4]);
	if (cmd == "bench") return bench(&m, threads);

	float temperature = 0.8f;
	if (args.len == 4)
	{
		float? t = args[3].to_float();
		if (catch err = t) { io::printfn("bad temperature: %s", args[3]); return 1; }
		if (!(t >= 0 && t <= 5)) { io::printfn("temperature must be 0 to 5"); return 1; }
		temperature = t;
	}
	return chat(&g, &tok, &m, temperature);
}