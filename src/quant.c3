module c3llm::quant;
import std::math;

// GGML tensor types this engine can read
const uint F32 = 0;
const uint Q4_0 = 2;
const uint Q4_1 = 3;
const uint Q6_K = 14;

// Bytes taken by n weights of a type, or 0 if the type or length is unsupported.
fn sz row_bytes(uint type, sz n)
{
	switch (type)
	{
		case F32: return n * 4;
		case Q4_0: return n % 32 ? 0 : n / 32 * 18;
		case Q4_1: return n % 32 ? 0 : n / 32 * 20;
		case Q6_K: return n % 256 ? 0 : n / 256 * 210;
		default: return 0;
	}
}

// Half-float to float through a table, as llama.cpp does. Converting each block's scale with the CPU's
// own instruction took as long as all the rest of a 32-weight block on a Zen 2 core.
float[65536] half_table;

fn void fill_half_table() @init
{
	foreach (i, &f : half_table) *f = (float)bitcast((ushort)i, float16);
}

fn float f16(char* p) @inline => half_table[mem::load((ushort*)p, 1)];

// 16 floats: two AVX2 registers, which the compiler splits for us.
alias Vec = float[<16>];

fn float dot_f32(float* a, float* b, sz n)
{
	Vec acc;
	sz i;
	for (; i + 16 <= n; i += 16) acc = mem::load((Vec*)(a + i), 4).fma(mem::load((Vec*)(b + i), 4), acc);
	float sum = acc.sum();
	for (; i < n; i++) sum += a[i] * b[i];
	return sum;
}

// Q6_K block, 210 bytes for 256 weights: ql[128] low 4 bits, qh[64] high 2 bits, scales[16] int8, f16 d
fn void dequant_q6k(char* p, float* y)
{
	char* ql = p;
	char* qh = p + 128;
	ichar* sc = (ichar*)(p + 192);
	float d = f16(p + 208);
	for (int half = 0; half < 2; half++)
	{
		for (int l = 0; l < 32; l++)
		{
			int s = l / 16;
			int q1 = (int)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
			int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
			int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
			int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
			y[l] = d * sc[s] * q1;
			y[l + 32] = d * sc[s + 2] * q2;
			y[l + 64] = d * sc[s + 4] * q3;
			y[l + 96] = d * sc[s + 6] * q4;
		}
		y += 128; ql += 64; qh += 32; sc += 8;
	}
}

// Expands n weights starting at p into floats. Used for embedding rows and as the reference in tests.
fn void dequant(uint type, char* p, float* out, sz n)
{
	if (type == F32)
	{
		mem::copy(out, p, n * 4);
	}
	else if (type == Q4_0)
	{
		for (sz b = 0; b < n; b += 32, p += 18)
		{
			float d = f16(p);
			for (int j = 0; j < 16; j++)
			{
				out[b + j] = d * (float)((int)(p[2 + j] & 0xF) - 8);
				out[b + j + 16] = d * (float)((int)(p[2 + j] >> 4) - 8);
			}
		}
	}
	else if (type == Q4_1)
	{
		for (sz b = 0; b < n; b += 32, p += 20)
		{
			float d = f16(p);
			float m = f16(p + 2);
			for (int j = 0; j < 16; j++)
			{
				out[b + j] = d * (p[4 + j] & 0xF) + m;
				out[b + j + 16] = d * (p[4 + j] >> 4) + m;
			}
		}
	}
	else if (type == Q6_K)
	{
		for (sz b = 0; b < n; b += 256, p += 210) dequant_q6k(p, out + b);
	}
}

// 16 products added in adjacent pairs. LLVM compiles exactly this shape to a single vpmaddwd.
macro int[<8>] madd(short[<16>] a, short[<16>] b)
{
	int[<16>] m = (int[<16>])a * (int[<16>])b;
	return swizzle(m, 0, 2, 4, 6, 8, 10, 12, 14) + swizzle(m, 1, 3, 5, 7, 9, 11, 13, 15);
}

macro short[<16>] shorts(short* p) => mem::load((short[<16>]*)p, 2);

<*
 Activations rounded to integers so the kernels multiply integers instead of floats (as llama.cpp does, with
 8 bits; here 15 bits for the 4-bit formats and 10 for Q6_K). Each block has its own scale, chosen so the
 block's largest value maps to the top of the range. Everything here is
 computed once per matrix product and shared by all rows, so the per-row kernels do as little as possible.
*>
struct Acts
{
	short[] q;     // round(x / scale), 16 bits so vpmaddwd can read them straight from memory
	float[] scale; // one per block
	float[] sum;   // 32-blocks: scale * sum(q), for the Q4_1 minimum
	int[] off;     // 32-blocks: 8 per block, 8 * the pair sums of q, which is the Q4_0 offset of -8 in vpmaddwd's lanes
	short[] bsum;  // 256-blocks: sum(q) over each 16 values, for the Q6_K offset of -32
}

fn Acts new_acts(sz max_n)
{
	sz blocks = max_n / 32 + 1;
	return {
		mem::new_array(short, max_n),
		mem::new_array(float, blocks),
		mem::new_array(float, blocks),
		mem::new_array(int, blocks * 8),
		mem::new_array(short, max_n / 16 + 16),
	};
}

// block is the block size of the weight format: 32 for Q4, 256 for Q6_K. n must be a multiple of it.
fn void Acts.set(&self, float* x, sz n, sz block)
{
	short[<16>] ones = 1;
	for (sz b = 0; b * block < n; b++)
	{
		float* xs = x + b * block;
		short* qs = &self.q[b * block];
		Vec biggest;
		for (sz i = 0; i < block; i += 16) biggest = math::max(biggest, math::abs(mem::load((Vec*)(xs + i), 4)));
		float amax = biggest.max();
		// 16-bit storage allows far finer steps than 8 bits at no cost. The limits keep every integer sum
		// in the kernels inside 32 bits (see dot_q4 and dot_q6k) and are multiples of 127 so tests stay exact.
		float limit = block == 32 ? 32004 : 1016;
		float inv = amax > 0 ? limit / amax : 0;
		for (sz i = 0; i < block; i += 16)
		{
			int[<16>] r = (int[<16>])(mem::load((Vec*)(xs + i), 4) * inv).rint();
			mem::store((short[<16>]*)(qs + i), (short[<16>])r, 2);
		}
		self.scale[b] = amax / limit;
		if (block == 32)
		{
			int[<8>] pairs = madd(ones, shorts(qs)) + madd(ones, shorts(qs + 16));
			mem::store((int[<8>]*)&self.off[b * 8], pairs * 8, 4);
			self.sum[b] = self.scale[b] * pairs.sum();
		}
		else
		{
			for (sz k = 0; k < block / 16; k++) self.bsum[b * (block / 16) + k] = shorts(qs + k * 16).sum();
		}
	}
}

<*
 Q4_0 block: f16 d, 16 bytes of nibbles.        weight = d * (nibble - 8)
 Q4_1 block: f16 d, f16 m, 16 bytes of nibbles. weight = d * nibble + m
 Low nibbles are elements 0-15, high nibbles 16-31.

 Returns sum(nibble * q) for one block as 8 lanes, with the Q4_0 offset already taken off.
*>
macro float[<8>] q4_lanes(char* nibbles, short* q, Acts* a, sz b, bool has_min)
{
	char[<16>] nib = mem::load((char[<16>]*)nibbles, 1);
	int[<8>] s = madd((short[<16>])(nib & 0xF), shorts(q)) + madd((short[<16>])(nib >> 4), shorts(q + 16));
	if (!has_min) s -= mem::load((int[<8>]*)&a.off[b * 8], 4);
	return (float[<8>])s;
}

// Two blocks per step into separate accumulators, so consecutive fused multiply-adds do not wait on each other.
fn float dot_q4(char* p, Acts* a, sz n, bool has_min)
{
	sz step = has_min ? 20 : 18;
	sz blocks = n / 32;
	float[<8>] acc0;
	float[<8>] acc1;
	float offset = 0;
	short* q = a.q.ptr;
	sz b = 0;
	for (; b + 1 < blocks; b += 2, p += 2 * step, q += 64)
	{
		float[<8>] d0 = f16(p) * a.scale[b];
		float[<8>] d1 = f16(p + step) * a.scale[b + 1];
		acc0 = d0.fma(q4_lanes(p + step - 16, q, a, b, has_min), acc0);
		acc1 = d1.fma(q4_lanes(p + 2 * step - 16, q + 32, a, b + 1, has_min), acc1);
		if (has_min) offset += f16(p + 2) * a.sum[b] + f16(p + step + 2) * a.sum[b + 1];
	}
	if (b < blocks)
	{
		float[<8>] d0 = f16(p) * a.scale[b];
		acc0 = d0.fma(q4_lanes(p + step - 16, q, a, b, has_min), acc0);
		if (has_min) offset += f16(p + 2) * a.sum[b];
	}
	return (acc0 + acc1).sum() + offset;
}

<*
 See dequant_q6k for the layout. Each 6-bit weight is rebuilt as 0..63 and multiplied by its group's int8
 scale (at most 63 * 128, which fits 16 bits). A block's total is at most 256 * 32 * 128 * 1016, which fits
 32 bits; partial sums may wrap on the way, which is harmless.
 The stored offset of -32 is applied once per block to the activation totals of the 16 groups.
*>
fn float dot_q6k(char* p, Acts* a, sz n)
{
	float sum = 0;
	short* q = a.q.ptr;
	for (sz b = 0; b < n / 256; b++, p += 210, q += 256)
	{
		short[<16>] group_scales = (short[<16>])mem::load((ichar[<16>]*)(p + 192), 1);
		short[16] scales @noinit;
		mem::store((short[<16>]*)&scales, group_scales, 2);
		int[<8>] acc = madd(group_scales, shorts(&a.bsum[b * 16])) * -32;
		// four groups of 16 positions; each yields 4 x 16 weights
		for (int g = 0; g < 4; g++)
		{
			int half = g / 2;
			int off = (g % 2) * 16;
			short* s = &scales[half * 8 + g % 2];
			short* qs = q + half * 128 + off;
			char[<16>] la = mem::load((char[<16>]*)(p + half * 64 + off), 1);
			char[<16>] lb = mem::load((char[<16>]*)(p + half * 64 + off + 32), 1);
			char[<16>] h = mem::load((char[<16>]*)(p + 128 + half * 32 + off), 1);
			short[<16>] w1 = (short[<16>])((la & 0xF) | ((h & 3) << 4)) * s[0];
			short[<16>] w2 = (short[<16>])((lb & 0xF) | ((h & 0xC) << 2)) * s[2];
			short[<16>] w3 = (short[<16>])((la >> 4) | (h & 0x30)) * s[4];
			short[<16>] w4 = (short[<16>])((lb >> 4) | ((h >> 2) & 0x30)) * s[6];
			acc += madd(w1, shorts(qs)) + madd(w2, shorts(qs + 32)) + madd(w3, shorts(qs + 64)) + madd(w4, shorts(qs + 96));
		}
		sum += (float)acc.sum() * (f16(p + 208) * a.scale[b]);
	}
	return sum;
}

// Dot product of one row of n weights at p with the activations: x as floats, a as their integer form.
fn float dot(uint type, char* p, float* x, Acts* a, sz n)
{
	switch (type)
	{
		case Q4_0: return dot_q4(p, a, n, false);
		case Q4_1: return dot_q4(p, a, n, true);
		case Q6_K: return dot_q6k(p, a, n);
		default: return dot_f32((float*)p, x, n);
	}
}
