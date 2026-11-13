module c3llm::sampler;
import std::math;

const int MAX_K = 256;

struct Sampler
{
	float temperature; // 0 or less picks the most likely token every time
	int top_k;         // keep only the k most likely tokens
	float top_p;       // then keep the smallest set whose probability reaches p
	ulong state;       // random state, must not be 0
}

// xorshift64: uniform in [0, 1)
fn float Sampler.random(&self)
{
	self.state ^= self.state << 13;
	self.state ^= self.state >> 7;
	self.state ^= self.state << 17;
	return (float)(self.state >> 40) / 16777216.0f;
}

fn uint Sampler.pick(&self, float[] logits)
{
	int k = self.temperature <= 0 ? 1 : self.top_k;
	if (k < 1) k = 1;
	if (k > MAX_K) k = MAX_K;

	// the k largest logits, sorted from largest to smallest
	uint[MAX_K] ids;
	float[MAX_K] vals;
	int n;
	foreach (id, l : logits)
	{
		if (l != l) continue; // NaN
		if (n == k && l <= vals[n - 1]) continue;
		int j = n < k ? n++ : k - 1;
		for (; j > 0 && vals[j - 1] < l; j--)
		{
			vals[j] = vals[j - 1];
			ids[j] = ids[j - 1];
		}
		vals[j] = l;
		ids[j] = (uint)id;
	}
	if (n <= 1) return ids[0];

	float top = vals[0];
	float total = 0;
	for (int i = 0; i < n; i++)
	{
		vals[i] = math::exp((vals[i] - top) / self.temperature);
		total += vals[i];
	}
	float kept = 0;
	int m = 0;
	while (m < n)
	{
		kept += vals[m++];
		if (kept >= self.top_p * total) break;
	}
	float r = self.random() * kept;
	for (int i = 0; i < m; i++)
	{
		r -= vals[i];
		if (r < 0) return ids[i];
	}
	return ids[m - 1];
}
