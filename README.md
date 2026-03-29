# ai-toolkit

A command-line toolkit for everyday AI/ML tasks. Compare embeddings, count tokens, diff model outputs, and benchmark prompts — all from your terminal.

## Why This Exists

Working with LLMs means constantly doing small, repetitive checks: How many tokens is this prompt? How similar are these two embeddings? Did the model output change after I tweaked the system message? This toolkit wraps those tasks into fast CLI commands so you never have to leave your terminal.

## Installation

```bash
pip install -e .
```

Or install from source:

```bash
git clone https://github.com/nripankadas07/ai-toolkit.git
cd ai-toolkit
pip install -e ".[dev]"
```

## Commands

### `aitk embeddings compare`

Compare the cosine similarity between two text strings using sentence embeddings.

```bash
aitk embeddings compare "machine learning" "deep learning"
# Similarity: 0.8734

aitk embeddings compare "python programming" "french cooking"
# Similarity: 0.1203

# Use a specific model
aitk embeddings compare "AI safety" "alignment research" --model all-MiniLM-L6-v2
```

### `aitk tokens count`

Count tokens for any text using common tokenizer encodings.

```bash
aitk tokens count "Hello, world!"
# Tokens (cl100k_base): 4

aitk tokens count --file prompt.txt
# Tokens (cl100k_base): 347

# Specify encoding
aitk tokens count "Some text" --encoding p50k_base

# Estimate cost
aitk tokens count --file prompt.txt --cost --model gpt-4
```

### `aitk tokens split`

Visualize how text gets split into tokens.

```bash
aitk tokens split "The quick brown fox"
# ['The', ' quick', ' brown', ' fox']
```

### `aitk diff outputs`

Compare two model outputs side-by-side with colored diffs.

```bash
aitk diff outputs response_v1.txt response_v2.txt
# Shows word-level diff with additions/deletions highlighted

aitk diff outputs --text "First response" "Second response"
```

### `aitk bench prompt`

Benchmark a prompt across multiple runs, measuring latency and output variance.

```bash
aitk bench prompt "Explain quantum computing in one sentence" --runs 5
# Avg latency: 1.23s | Std dev: 0.15s | Output variance: 0.34
```

### `aitk text stats`

Quick text statistics — word count, sentence count, reading level, and more.

```bash
aitk text stats "Your long document text here"
# Words: 142 | Sentences: 8 | Avg words/sentence: 17.8 | Reading level: Grade 11

aitk text stats --file article.md
```

## Project Structure

```
ai-toolkit/
├── ai_toolkit/
│   ├── __init__.py
│   ├── cli.py              # Main CLI entry point
│   ├── embeddings.py        # Embedding comparison utilities
│   ├── tokens.py            # Token counting and splitting
│   ├── diff.py              # Output diffing
│   ├── bench.py             # Prompt benchmarking
│   └── text_stats.py        # Text statistics
├── tests/
│   ├── test_embeddings.py
│   ├── test_tokens.py
│   ├── test_diff.py
│   └── test_text_stats.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy ai_toolkit/
```

## License

MIT
