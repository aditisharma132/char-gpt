# char-gpt

A character-level GPT trained on the Tiny Shakespeare dataset.

This is a replicate of the GPT implementation from Andrej Karpathy's
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
lecture — the same `Head` / `MultiHeadAttention` / `Block` transformer
structure, trained on the same [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
dataset. I rebuilt it as a learning exercise and then optimized it after
noticing the model was overfitting.

## What's different from the original

The initial version used a GPT-2 BPE tokenizer (`tiktoken`, vocab size
~50k) instead of the original's character-level tokenizer. On a ~1MB
dataset, that meant most of the model's parameters lived in the token
embedding and output head alone, so it memorized the training set almost
immediately — train loss kept falling while validation loss bottomed out
early and then climbed for the rest of training.

Changes made to fix that and improve training generally:

- **Character-level tokenizer** (vocab ~50k → ~65) — removes the
  oversized embedding/output table that was the main cause of
  overfitting on a dataset this small.
- **Weight tying** between the token embedding and output head.
- **Weight decay + gradient clipping** on the optimizer.
- **Cosine learning rate decay** instead of a constant LR.
- **Early stopping** with best-checkpoint restore, based on validation
  loss, instead of always training for the full fixed number of steps.
- **Temperature + top-k sampling** at generation time for more coherent
  output.
- Bumped model capacity back up (`n_embd`, `n_head`, `n_layer`,
  `block_size`) now that the tokenizer change freed up parameter budget
  without reintroducing the overfitting problem.

## Usage

```bash
pip install -r requirements.txt
python nano_gpt.py
```

The Tiny Shakespeare dataset is downloaded automatically on first run.
Training prints train/val loss every `eval_interval` steps, stops early
once validation loss stops improving, and generates a sample of text
from the best checkpoint at the end.
