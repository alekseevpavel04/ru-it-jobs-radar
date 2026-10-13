// The student's prompt for a pasted text, rendered the way radar.train.dataset renders it in training:
// the input cut after its first max_input_tokens tokens of the Qwen3 tokenizer, then the system prompt
// and the Qwen3 chat template with thinking off (an empty think block).
//
// Python cuts at the character offset where token number max_input_tokens ends (tokenizers offsets).
// tokenizers.js gives tokens, not offsets, so the offset is rebuilt from the bytes of the tokens: a
// byte-level token stands for the UTF-8 bytes of the NFC-normalized text, an added token for its own
// text. A token that ends inside a character covers that whole character, as the Rust offsets do.

const BYTE_OF_CHAR = (() => {
  // GPT-2 byte-to-unicode table, inverted: the character a byte-level token uses for each byte.
  const printable = [];
  for (let b = 33; b <= 126; b++) printable.push(b);
  for (let b = 161; b <= 172; b++) printable.push(b);
  for (let b = 174; b <= 255; b++) printable.push(b);
  const map = new Map();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    const code = printable.includes(b) ? b : 256 + extra++;
    map.set(String.fromCodePoint(code), b);
  }
  return map;
})();

const utf8 = new TextEncoder();

function tokenBytes(token, added) {
  if (added.has(token)) return utf8.encode(token).length;
  let n = 0;
  for (const ch of token) {
    if (!BYTE_OF_CHAR.has(ch)) throw new Error(`not a byte-level token: ${JSON.stringify(token)}`);
    n++;
  }
  return n;
}

// UTF-16 length of the shortest prefix of `text` whose UTF-8 form holds at least `bytes` bytes.
function charsCoveringBytes(text, bytes) {
  let seen = 0;
  let i = 0;
  for (const ch of text) {
    if (seen >= bytes) break;
    seen += utf8.encode(ch).length;
    i += ch.length;
  }
  return i;
}

export function addedTokens(tokenizerJson) {
  return new Set(tokenizerJson.added_tokens.map((t) => t.content));
}

// [input, truncated]: `text` cut after its first `maxTokens` tokens.
export function capInput(tokenizer, added, text, maxTokens) {
  const normalized = text.normalize("NFC");
  const tokens = tokenizer.tokenize(normalized, {add_special_tokens: false});
  if (tokens.length <= maxTokens) return [text, false];
  let bytes = 0;
  for (const token of tokens.slice(0, maxTokens)) bytes += tokenBytes(token, added);
  const cut = charsCoveringBytes(normalized, bytes);
  if (normalized === text) return [text.slice(0, cut), true];
  // The text changes under NFC: the shortest original prefix that normalizes to the cut prefix.
  const want = normalized.slice(0, cut);
  for (let end = cut; end <= text.length; end++) {
    if (text.slice(0, end).normalize("NFC") === want) return [text.slice(0, end), true];
  }
  throw new Error("no prefix of the text normalizes to the cut");
}

export function renderPrompt(system, user) {
  return (
    `<|im_start|>system\n${system}<|im_end|>\n` +
    `<|im_start|>user\n${user}<|im_end|>\n` +
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  );
}

export function promptFor(tokenizer, added, meta, text) {
  const [user, truncated] = capInput(tokenizer, added, text, meta.max_input_tokens);
  return {prompt: renderPrompt(meta.system_prompt, user), truncated};
}
