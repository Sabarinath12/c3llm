module c3llm::tokenizer;
import std::collections::map, std::collections::list;
import c3llm::gguf;

excuse UNSUPPORTED_TOKENIZER;

// GPT-2 style byte-level BPE, as used by Llama 3.
struct Tokenizer
{
	String[] vocab;
	HashMap{String, uint} ids;
	HashMap{String, uint} ranks; // "left right" -> merge priority, lower merges first
}

fn Tokenizer create(String[] vocab, String[] merges)
{
	Tokenizer t = { .vocab = vocab };
	t.ids.init(mem, vocab.len * 2 + 16);
	t.ranks.init(mem, merges.len * 2 + 16);
	foreach (i, v : vocab) t.ids.set(v, (uint)i);
	foreach (i, m : merges) t.ranks.set(m, (uint)i);
	return t;
}

fn Tokenizer? load(Gguf* g)
{
	Value kind = g.get("tokenizer.ggml.model")!;
	if (kind.type != gguf::T_STRING || (String)kind.raw != "gpt2") return UNSUPPORTED_TOKENIZER~;
	return create(g.get("tokenizer.ggml.tokens")!.strings()!, g.get("tokenizer.ggml.merges")!.strings()!);
}

// GPT-2 writes every byte as a printable character: printable Latin-1 bytes stay, the rest move to U+0100 and up.
fn uint byte_cp(char b)
{
	if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE) return b;
	if (b <= 0x20) return 256 + (uint)b;
	if (b <= 0xA0) return 289 + (uint)(b - 0x7F);
	return 323; // 0xAD
}

fn int cp_byte(uint cp)
{
	if (cp < 256) return byte_cp((char)cp) == cp ? (int)cp : -1;
	if (cp <= 288) return (int)(cp - 256);
	if (cp <= 322) return (int)(cp - 289 + 0x7F);
	return cp == 323 ? 0xAD : -1;
}

fn bool is_letter(char c) => ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') || c >= 0x80;
fn bool is_digit(char c) => c >= '0' && c <= '9';
fn bool is_space(char c) => c == ' ' || (c >= 0x09 && c <= 0x0D);
fn bool is_newline(char c) => c == '\r' || c == '\n';
fn bool is_other(char c) => !is_letter(c) && !is_digit(c) && !is_space(c);

<*
 End of the pre-token starting at i, following the llama-bpe pattern:
 contractions | [^\r\n L N]? L+ | N{1,3} | ' '? [^\s L N]+ [\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+

 shortcut: every non-ASCII byte counts as a letter, so non-ASCII punctuation, digits, spaces and emoji
 can split differently from llama.cpp. Upgrade to real Unicode categories when that matters.
*>
fn sz word_end(String s, sz i)
{
	sz n = s.len;
	char c = s[i];
	if (c == '\'' && i + 1 < n)
	{
		char a = s[i + 1] | 0x20;
		if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
		if (i + 2 < n)
		{
			char b = s[i + 2] | 0x20;
			if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
		}
	}
	sz j = i;
	if (!is_letter(c) && !is_digit(c) && !is_newline(c) && i + 1 < n && is_letter(s[i + 1])) j = i + 1;
	if (is_letter(s[j]))
	{
		while (j < n && is_letter(s[j])) j++;
		return j;
	}
	if (is_digit(c))
	{
		while (j < n && j < i + 3 && is_digit(s[j])) j++;
		return j;
	}
	if (c == ' ' && i + 1 < n && is_other(s[i + 1])) j = i + 1;
	if (is_other(s[j]))
	{
		while (j < n && is_other(s[j])) j++;
		while (j < n && is_newline(s[j])) j++;
		return j;
	}
	// only whitespace is left
	sz last_newline = -1;
	while (j < n && is_space(s[j]))
	{
		if (is_newline(s[j])) last_newline = j;
		j++;
	}
	if (last_newline >= 0) return last_newline + 1;
	if (j == n || j - i == 1) return j;
	return j - 1; // leave one space to attach to the next word
}

fn void Tokenizer.encode_word(&self, String word, List{uint}* out)
{
	// each byte becomes one symbol, written as its mapped character (1 or 2 UTF-8 bytes)
	char[] buf = mem::new_array(char, word.len * 2);
	sz[] bound = mem::new_array(sz, word.len + 1); // symbol k is buf[bound[k] .. bound[k + 1])
	defer mem::free(buf);
	defer mem::free(bound);
	sz len;
	foreach (k, b : word)
	{
		bound[k] = len;
		uint cp = byte_cp(b);
		if (cp < 0x80)
		{
			buf[len++] = (char)cp;
		}
		else
		{
			buf[len++] = (char)(0xC0 | (cp >> 6));
			buf[len++] = (char)(0x80 | (cp & 0x3F));
		}
	}
	sz count = word.len;
	bound[count] = len;

	// repeatedly merge the adjacent pair with the lowest rank
	char[600] key;
	while (count > 1)
	{
		uint best_rank = uint::max;
		sz best;
		for (sz k = 0; k + 1 < count; k++)
		{
			sz left = bound[k + 1] - bound[k];
			sz right = bound[k + 2] - bound[k + 1];
			if (left + right + 1 > key.len) continue; // longer than any merge
			mem::copy(&key, &buf[bound[k]], left);
			key[left] = ' ';
			mem::copy(&key[left + 1], &buf[bound[k + 1]], right);
			if (try rank = self.ranks.get((String)key[:left + 1 + right]))
			{
				if (rank < best_rank) { best_rank = rank; best = k; }
			}
		}
		if (best_rank == uint::max) break;
		for (sz k = best + 1; k < count; k++) bound[k] = bound[k + 1];
		count--;
	}

	for (sz k = 0; k < count; k++)
	{
		String sym = (String)buf[bound[k]:bound[k + 1] - bound[k]];
		if (try id = self.ids.get(sym))
		{
			out.push(id);
			continue;
		}
		// unknown symbol: fall back to its single-byte tokens
		for (sz p = 0; p < sym.len;)
		{
			sz width = sym[p] < 0x80 ? 1 : 2;
			if (try id = self.ids.get(sym[p:width])) out.push(id);
			p += width;
		}
	}
}

fn void Tokenizer.encode_plain(&self, String text, List{uint}* out)
{
	for (sz i = 0; i < text.len;)
	{
		sz end = word_end(text, i);
		self.encode_word(text[i:end - i], out);
		i = end;
	}
}

// Appends the token ids for text. Special tokens written as <|name|> map straight to their id.
fn void Tokenizer.encode(&self, String text, List{uint}* out)
{
	sz start;
	for (sz i = 0; i < text.len; i++)
	{
		if (text[i] != '<' || i + 1 >= text.len || text[i + 1] != '|') continue;
		sz close = -1;
		for (sz k = i + 2; k + 1 < text.len && k < i + 64; k++)
		{
			if (text[k] == '|' && text[k + 1] == '>') { close = k; break; }
		}
		if (close < 0) continue;
		if (try id = self.ids.get(text[i:close + 2 - i]))
		{
			self.encode_plain(text[start:i - start], out);
			out.push(id);
			start = close + 2;
			i = close + 1;
		}
	}
	self.encode_plain(text[start..], out);
}

// Writes the bytes a token stands for into buf and returns them. Output never exceeds the token's own length.
fn String Tokenizer.decode(&self, uint id, char[] buf)
{
	if (id >= self.vocab.len) return "";
	String piece = self.vocab[id];
	sz len;
	for (sz p = 0; p < piece.len && len < buf.len; p++)
	{
		char c = piece[p];
		if (c >= 0xC0 && c < 0xE0 && p + 1 < piece.len)
		{
			int b = cp_byte(((uint)(c & 0x1F) << 6) | (piece[p + 1] & 0x3F));
			if (b >= 0)
			{
				buf[len++] = (char)b;
				p++;
				continue;
			}
		}
		buf[len++] = c;
	}
	return (String)buf[:len];
}
