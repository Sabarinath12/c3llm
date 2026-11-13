module c3llm::gguf;
import std::io::file;

excuse BAD_MAGIC, BAD_VERSION, TRUNCATED, BAD_TYPE, NOT_FOUND;

// GGUF metadata value types
const uint T_UINT8 = 0;
const uint T_INT8 = 1;
const uint T_UINT16 = 2;
const uint T_INT16 = 3;
const uint T_UINT32 = 4;
const uint T_INT32 = 5;
const uint T_FLOAT32 = 6;
const uint T_BOOL = 7;
const uint T_STRING = 8;
const uint T_ARRAY = 9;
const uint T_UINT64 = 10;
const uint T_INT64 = 11;
const uint T_FLOAT64 = 12;

const char[13] SCALAR_SIZE = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

// A view into the mapped file, never a copy.
struct Value
{
	uint type;
	uint elem_type; // arrays only
	ulong count;    // arrays only
	char[] raw;     // scalar bytes, string bytes, or the whole array payload
}

struct Meta
{
	String key;
	Value value;
}

struct Tensor
{
	String name;
	uint n_dims;
	ulong[4] dims;
	uint type;
	ulong offset; // absolute offset in the file
}

struct Gguf
{
	FileMmap map;
	char[] data;
	uint version;
	uint alignment;
	Meta[] meta;
	Tensor[] tensors;
}

struct Reader
{
	char[] data;
	sz pos;
}

fn char[]? Reader.bytes(&self, ulong n)
{
	if (n > (ulong)(self.data.len - self.pos)) return TRUNCATED~;
	char[] out = self.data[self.pos:(sz)n];
	self.pos += (sz)n;
	return out;
}

// Unaligned little-endian read; the file gives no alignment guarantee.
macro load(char[] raw, $Type)
{
	$Type v;
	mem::copy(&v, raw.ptr, $Type::size);
	return v;
}

macro Reader.read(&self, $Type)
{
	return load(self.bytes($Type::size)!, $Type);
}

fn String? Reader.string(&self)
{
	ulong len = self.read(ulong)!;
	return (String)self.bytes(len)!;
}

fn Value? Reader.value(&self, uint type)
{
	if (type == T_STRING) return { .type = type, .raw = (char[])self.string()! };
	if (type == T_ARRAY)
	{
		uint elem = self.read(uint)!;
		ulong count = self.read(ulong)!;
		sz start = self.pos;
		if (elem == T_STRING)
		{
			for (ulong i = 0; i < count; i++) self.string()!;
		}
		else
		{
			// shortcut: nested arrays rejected, add them if a model ever uses one
			if (elem >= SCALAR_SIZE.len || SCALAR_SIZE[elem] == 0) return BAD_TYPE~;
			if (count > (ulong)self.data.len) return TRUNCATED~; // keeps the multiply below from overflowing
			self.bytes(count * SCALAR_SIZE[elem])!;
		}
		return { .type = type, .elem_type = elem, .count = count, .raw = self.data[start:self.pos - start] };
	}
	if (type >= SCALAR_SIZE.len) return BAD_TYPE~;
	return { .type = type, .raw = self.bytes(SCALAR_SIZE[type])! };
}

fn ulong Value.as_uint(&self)
{
	switch (self.type)
	{
		case T_UINT8: case T_BOOL: return self.raw[0];
		case T_INT8: return (ulong)(long)load(self.raw, ichar);
		case T_UINT16: return load(self.raw, ushort);
		case T_INT16: return (ulong)(long)load(self.raw, short);
		case T_UINT32: return load(self.raw, uint);
		case T_INT32: return (ulong)(long)load(self.raw, int);
		case T_UINT64: case T_INT64: return load(self.raw, ulong);
		default: return 0;
	}
}

fn double Value.as_float(&self)
{
	if (self.type == T_FLOAT32) return load(self.raw, float);
	if (self.type == T_FLOAT64) return load(self.raw, double);
	return (double)self.as_uint();
}

// Views of every string in a string array. The parser already walked them, so the reads cannot fail on a parsed file.
fn String[]? Value.strings(&self)
{
	if (self.type != T_ARRAY || self.elem_type != T_STRING) return BAD_TYPE~;
	Reader r = { .data = self.raw };
	String[] out = mem::new_array(String, (sz)self.count);
	foreach (&s : out) *s = r.string()!;
	return out;
}

fn Value? Gguf.get(&self, String key)
{
	foreach (m : self.meta) if (m.key == key) return m.value;
	return NOT_FOUND~;
}

fn Gguf? parse(char[] data)
{
	Reader r = { .data = data };
	char[] magic = r.bytes(4)!;
	if ((String)magic != "GGUF") return BAD_MAGIC~;
	Gguf g = { .data = data, .alignment = 32 };
	g.version = r.read(uint)!;
	if (g.version != 3) return BAD_VERSION~;
	ulong n_tensors = r.read(ulong)!;
	ulong n_meta = r.read(ulong)!;
	// every entry takes at least one byte, so a count above the file size is corrupt
	if (n_tensors > (ulong)data.len || n_meta > (ulong)data.len) return TRUNCATED~;

	g.meta = mem::new_array(Meta, (sz)n_meta);
	foreach (&m : g.meta)
	{
		m.key = r.string()!;
		m.value = r.value(r.read(uint)!)!;
	}
	if (try a = g.get("general.alignment"))
	{
		g.alignment = (uint)a.as_uint();
		if (g.alignment == 0) return BAD_TYPE~;
	}

	g.tensors = mem::new_array(Tensor, (sz)n_tensors);
	foreach (&t : g.tensors)
	{
		t.name = r.string()!;
		t.n_dims = r.read(uint)!;
		if (t.n_dims > 4) return BAD_TYPE~;
		for (uint d = 0; d < t.n_dims; d++) t.dims[d] = r.read(ulong)!;
		t.type = r.read(uint)!;
		t.offset = r.read(ulong)!;
	}
	ulong base = ((ulong)r.pos + g.alignment - 1) / g.alignment * g.alignment;
	foreach (&t : g.tensors)
	{
		if (base > (ulong)data.len || t.offset > (ulong)data.len - base) return TRUNCATED~;
		t.offset += base;
	}
	return g;
}

fn Gguf? open(String path)
{
	FileMmap map = file::mmap_open(path, "rb")!;
	Gguf? g = parse(map.bytes());
	if (catch err = g)
	{
		map.destroy();
		return err~;
	}
	g.map = map;
	return g;
}

fn String type_name(uint type)
{
	switch (type)
	{
		case 0: return "F32";
		case 1: return "F16";
		case 2: return "Q4_0";
		case 3: return "Q4_1";
		case 6: return "Q5_0";
		case 7: return "Q5_1";
		case 8: return "Q8_0";
		case 10: return "Q2_K";
		case 11: return "Q3_K";
		case 12: return "Q4_K";
		case 13: return "Q5_K";
		case 14: return "Q6_K";
		case 30: return "BF16";
		default: return "other";
	}
}
