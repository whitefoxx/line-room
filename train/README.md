# Fine-tuning Needle for this room

**Read this first: the fine-tuned model lost and is not shipped.** It improved
what it was trained on — targeted commands 4→5 of 8, paraphrases 7→9 of 16 —
and destroyed the model's willingness to decline, 4/5 refusals down to 0/5.
Asked the population of Tokyo, it called `set_fan(speed=high)` at confidence
0.45. The base weights are what the room runs.

This directory is kept because the negative result is the useful part, and
because the dataset generator and the eval harness are what produced every
number in the top-level README. What follows is how it was done.

The goal was to teach the base model this room's four tools so it would stop
declining things it can actually do, and stop guessing the desk lamp when you
say "the lights".

Everything here runs on the same Intel Mac the room runs on. Read
`../docs/pitfalls.md` first — three separate walls stand between a plain
`pip install cactus-needle` and a working trainer on this platform.

## Setup

```sh
uv venv --python 3.12 ../.venv

# cactus-needle pins flax>=0.12.8, which needs a jax with no Intel-Mac wheels.
# The fine-tuning path only imports flax.linen, so the pin is stricter than the
# code and can be satisfied by hand.
uv pip install --python ../.venv/bin/python --no-deps cactus-needle
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
uv pip install --python ../.venv/bin/python \
  "jax==0.4.38" "jaxlib==0.4.38" "flax==0.10.2" optax sentencepiece \
  huggingface_hub "numpy<2.2"

# The library auto-downloads checkpoints/needle2.pkl and 404s: the repo
# publishes it under weights/.
mkdir -p checkpoints
curl -L -o checkpoints/needle2.pkl \
  https://huggingface.co/Cactus-Compute/needle2/resolve/main/weights/needle2.pkl
```

## The float16 problem

The published checkpoint is float16 — all 56 tensors. The base forward pass
works, but it runs close to the edge: logits reach ~7.6e3 against fp16's 65504
ceiling. LoRA plus AdamW at lr 1e-4 keeps every intermediate in fp16 (the
adapters are created with `weight.dtype` and `merge_lora` casts back), and
within four steps the loss is `nan`.

So train from an fp32 copy. Quantization happens at export anyway, so nothing
is lost by carrying more precision through training:

```sh
../.venv/bin/python -c "
import pickle, numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
ck = pickle.load(open('checkpoints/needle2.pkl','rb'))
flat = {k: np.asarray(v, np.float32) for k, v in flatten_dict(ck['params']).items()}
ck['params'] = unflatten_dict(flat)
pickle.dump(ck, open('checkpoints/needle2-fp32.pkl','wb'), protocol=4)
"
```

## Data

`needle generate-data` wants an OPENROUTER_API_KEY to have an LLM invent
examples. There isn't one here, and authoring the set directly is the better
trade anyway: we had already measured exactly which utterances the base model
gets wrong, so it can be aimed at those.

```sh
python3 make_dataset.py     # -> data/train.jsonl, data/heldout.json
```

`make_dataset.py` refuses to build if any held-out phrasing has leaked into
training. That guard matters: "we fixed it" measured on a sentence that was
trained on is worth nothing.

Sequences come out at median 445 tokens, p95 469 — so `--max-len 512` covers
the set and costs half of the 1024 default.

## Train and export

```sh
../.venv/bin/python -u -m needle.cli finetune data/train.jsonl \
  --checkpoint checkpoints/needle2-fp32.pkl \
  --epochs 3 --batch-size 8 --max-len 512 \
  --lora-rank 16 --lora-alpha 32 --lr 1e-4 \
  --out checkpoints/room_lora.pkl

../.venv/bin/needle build checkpoints/needle2-fp32.pkl \
  --lora checkpoints/room_lora.pkl \
  --out ../needle2-tuned.cact
```

Do **not** pass `--bits 2`. The shipped model is mixed precision *averaging*
two bits, and that per-tensor map lives in the checkpoint config as
`weight_bits`. `build` uses it automatically, but only when `--bits` is absent
— passing `--bits 2` overrides the map with uniform 2-bit and quietly produces
a differently-quantized model than the base you are comparing against.

Two invocation traps, both of which cost time here:

- Use the `needle` console script, not `python -m needle.cli`. That module has
  no `__main__` guard, so `-m` exits 0 having done nothing at all.
- Redirect to a file; don't pipe through `tail`. A pipe buffers the entire run,
  so a job that is working looks identical to one that is hung.

The `.cact` drops straight into the wasm engine with no recompilation.

## Evaluate

The page takes a `?weights=` override, so base and tuned can be compared
through the real browser engine:

```
http://127.0.0.1:8231/?weights=needle2.cact
http://127.0.0.1:8231/?weights=needle2-tuned.cact
```

Paste `eval.js` into the console and `await lineRoomEval()`. It drives
`window.__lineRoom.ask`, which returns raw tool calls without touching the
room, so a score never depends on what state the room happens to be in.

Three buckets, and only two of them mean anything on their own:

- **targeted** — phrasings that ARE in training. Says whether the specific
  failures got fixed. Nothing more.
- **paraphrase** — phrasings in training NOWHERE. This is the number that says
  whether anything generalised.
- **refusal** — must come back empty. Watched closely: a LoRA that pulls
  everything toward calling a tool would regress this, and that regression
  would be worse than the failures being fixed.
