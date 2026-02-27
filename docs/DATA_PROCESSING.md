# Data Processing (BPE Tokenizer)

The tokenizer follows the standard Byte‑Pair Encoding algorithm, optimized for multilingual support as seen in Qwen/Llama-3.

## 1. Vocabulary Construction
- Start from the byte‑level alphabet (256 symbols).
- Iteratively merge the most frequent pair of symbols tokens in the training corpus.
- Stop when the vocabulary size reaches `vocab_size`.

## 2. Special Tokens
- `<bos>` (begin‑of‑sentence): Marks the start of a prompt.
- `<eos>` (end‑of‑sentence): Marks the completion of a sequence.
- `<pad>`: Used to align batch sequences to the same length.
- `<unk>`: Representing tokens not in the vocabulary (rare in modern BPE implementations but occasionally used).

## 3. Encoding Pipeline
1. **Normalization**: NFKC normalization or similar pre-processing.
2. **Byte-Level Pre-tokenization**: Input text is UTF‑8 encoded and split into bytes.
3. **BPE Merges**: The learned merges are applied to produce token IDs.

## 4. Masking & Padding Strategy
- **Causal Masking**: A triangular mask $M_{i,j}$ where $M_{i,j}=1$ if $j \le i$ and $0$ otherwise ensures that position $i$ can only attend to positions $1 \dots i$.
- **Padding**: Sequences shorter than `seq_len` are padded. The attention mask should exclude padding tokens from being attended to.
- **Truncation**: Sequences longer than `seq_len` are truncated to the maximum allowed length.
