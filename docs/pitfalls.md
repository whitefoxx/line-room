# Pitfalls

Things that actually cost debugging time.

---

## Fine-tuning on an Intel Mac: three walls in a row

**1. `jaxlib` ships no macOS x86_64 wheels.** `pip install cactus-needle` fails
outright with a resolver essay. Wheels exist for manylinux x86_64/aarch64,
`macosx_11_0_arm64` and `win_amd64` — Apple Silicon yes, Intel Mac no. The last
`jaxlib` with Intel-Mac wheels is **0.4.38**, and `cactus-needle` pins
`flax>=0.12.8`, which requires a much newer jax.

The pin is stricter than the code. Every module on the fine-tuning path
(`finetune.py`, `architecture.py`, `quantize.py`, `export.py`) imports only
`flax.linen` — the old stable API — plus `jax`, `jax.numpy`, `optax`, `numpy`.
So install without dependency resolution and pick versions by hand:

```sh
uv pip install --no-deps cactus-needle
uv pip install "jax==0.4.38" "jaxlib==0.4.38" "flax==0.10.2" \
               optax sentencepiece huggingface_hub "numpy<2.2"
```

That runs natively. Docker with a `linux/amd64` image also works (the host is
x86_64, so there is no emulation penalty) but it is the worse trade on a 16 GB
machine — see below.

**2. PyPI direct is unusable from mainland China.** The same install that hung
past ten minutes and had to be killed finished in seconds through the Tsinghua
mirror:

```sh
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install ...
```

Symptom to recognise: no error, no output, no progress — the resolver has
already finished and it is the wheel download that is stalled.

**3. The base checkpoint is not where the library looks.** `needle finetune`
auto-downloads `checkpoints/needle2.pkl` from `Cactus-Compute/needle2` and gets
a 404, because the repo actually publishes it at **`weights/needle2.pkl`**
(90 MB). Note the failure is a clean `RemoteEntryNotFoundError`, not a network
error — that distinction is what tells you the path is wrong rather than the
connection. Fetch it yourself and pass `--checkpoint` explicitly:

```sh
curl -L -o checkpoints/needle2.pkl \
  https://huggingface.co/Cactus-Compute/needle2/resolve/main/weights/needle2.pkl
```

Also note `hf-mirror.com` did not have it; huggingface.co direct did.

---

## Turing GPUs (T4, RTX 20-series) cannot run this fine-tune

```
jax.errors.JaxRuntimeError: INTERNAL:
SDPA FP16/BF16 requires SM80 (Ampere) or newer architecture
```

Needle's attention dispatches to cuDNN's fused scaled-dot-product-attention
under jax 0.11 / flax 0.12, and that kernel needs compute capability 8.0+.
A Tesla T4 (Colab's free tier) is SM75. **So is an RTX 2060** — same generation,
same wall. "Get a GPU" only helps from Ampere onward: RTX 30-series, 40-series,
L4, A100.

`XLA_FLAGS=--xla_gpu_enable_cudnn_fmha=false` does *not* fix it — that toggles
an XLA fusion pass, while the choice is made inside
`jax.nn.dot_product_attention`'s implementation selection. Untested ways out:
pin the versions that work on CPU here (`jax 0.4.38` / `flax 0.10.2`) against a
CUDA build, or fall back to `JAX_PLATFORMS=cpu` and lose the GPU entirely.

Everything *else* about Colab worked well: `pip install cactus-needle[gpu]` in
33s, checkpoint in ~3s, fp32 conversion instant. The blocker is the card.

Two smaller Colab notes:

- **Line continuations break in IPython `!` magic.** A `!wget -O out \` across
  two lines silently produced a 0-byte file; `-q` hid the error. Keep `!`
  commands on one line.
- **The upload widgets cannot be driven programmatically.** `files.upload()`
  renders in a cross-origin output iframe, and the sidebar's "Upload to session
  storage" ignores a synthetic `change` event even after the file is attached to
  the input. Uploading has to be done by hand — or avoided entirely by cloning
  the repo from git.

---

## Starting Docker Desktop starts everything else too

`open -a Docker` brought up a full Supabase stack from an unrelated project
(twelve containers, ~2 GB) that had been left with a restart policy. On a 16 GB
Intel Mac already swapping 8.5 GB, that made the Docker API itself take minutes
to answer a `docker stats`. If you launch Docker to run one container, check
`docker ps` first and stop what you did not ask for — then quit Docker Desktop
when you are done, because its GUI helper alone sat at 48% CPU.

---

Read the rest before touching the wasm engine.

---

## `needle_load` is zero-copy — never free the weights buffer

**Symptom.** The wasm engine loads and initialises cleanly: `needle_load`
returns 0, `needle_init` returns the exact same value as the native library
(same prompt token count, so the tokenizer and grammar compiler are demonstrably
fine). Then every single query comes back as a refusal:

```json
{"type":"call","success":true,"function_calls":[],
 "reasoning":null,"confidence":0.2000}
```

Identical `confidence: 0.2` on every input, including trivially answerable ones.
It looks like a broken or mismatched model file.

**Root cause.** `needle_load(const unsigned char* cact, unsigned long long n)`
does not copy. The engine reads 2-bit weight codes straight out of the buffer
you handed it and expands them inside vector registers — that is exactly what
"resident memory stays at blob size" means on the product page. Calling
`Module._free(ptr)` after `needle_load` therefore hands the allocator 13.7 MB
that the engine is still reading through. The forward pass runs on whatever
lands there next, the grammar keeps the output well-formed anyway, and the
shortest legal completion happens to be an empty call list. So a memory bug
disguises itself as a polite refusal.

**Fix.** Allocate the `.cact` once and keep the pointer alive for the whole
session:

```js
const cact = new Uint8Array(await (await fetch('needle2.cact')).arrayBuffer());
const ptr = Module._malloc(cact.length);
Module.HEAPU8.set(cact, ptr);
Module._needle_load(ptr, BigInt(cact.length));  // NOT a Number — see below
// deliberately never freed; it is the model
```

**Lesson.** The Python binding hides this: it passes a `bytes` object that stays
referenced for the lifetime of the script, so nobody upstream ever hit it. Any
manual FFI caller — wasm, Rust, C — has to know. When a model returns
structurally valid but semantically empty output at a suspiciously constant
confidence, suspect the weights, not the prompt.

---

## `needle_load`'s size argument must be a BigInt

The parameter is `unsigned long long`, which crosses the wasm boundary as i64.
Passing a JS Number throws `TypeError: Cannot convert 13737679 to a BigInt`.
Wrap it: `BigInt(cact.length)`.

---

## Only `ccall`, `cwrap` and `UTF8ToString` are exported on the Module

`lengthBytesUTF8` and `stringToUTF8` are **not**. Either go through `ccall`
(which converts `'string'` arguments for you, and maps `null` to a NULL pointer,
which is what `tool_index_path` wants when unused), or encode into the heap
yourself with `TextEncoder` + `HEAPU8` and a trailing zero byte.

---

## This build ignores `Module.locateFile`

**Symptom.** `WebAssembly.instantiate(): expected magic word 00 61 73 6d, found
3c 21 44 4f`. Those four bytes are `<!DO` — the server's 404 page.

**Root cause.** The glue defines its own private `locateFile(path) {
return scriptDirectory + path }` and never consults `Module.locateFile`.
`scriptDirectory` comes from `self.location.href`, which inside a worker is the
**worker's** URL, not the engine's. So it looked for `/src/needle.wasm`.

**Fix.** Skip path resolution: fetch the wasm yourself and pass the bytes as
`Module.wasmBinary`, which this build does honour.

---

## Relative URLs inside the worker resolve against the worker, not the page

`'models/needle2.cact'` became `/src/models/needle2.cact` → 404. Resolve any URL
on the main thread (`new URL(path, document.baseURI).href`) before posting it in.
The weights URL is also the Cache API key, so it has to stay stable anyway.

---

## Inference blocks; keep it off the main thread

`needle_complete` is synchronous and takes about a second in wasm. On the main
thread that freezes the 60 fps render loop for the entire call — the room
visibly stalls. It belongs in a Web Worker, with the 13.7 MB blob living in the
worker's heap.

A worker is necessary but not sufficient: the two still compete for CPU.
Measured on an Intel Mac, same command three times:

| render loop | latency | decode |
|---|---|---|
| 60 fps | 1.7–2.2 s | 16–17 tok/s |
| stopped entirely | 1.1 s | 25–28 tok/s |
| capped at 20 fps while thinking | 1.0 s | 30 tok/s |

Capping is what `room.setFrameCap(20)` does during a turn. Devices keep
animating; only the render is skipped, so the room stays alive rather than
freezing. Note that stopping the loop outright was *worse* than capping it —
the numbers are noisy at this granularity, so treat them as "roughly half the
latency", not as three significant figures.

---

## Don't build while a dev server is serving the same directory

Applies to this repo the moment it grows a bundler. Serving and building share
the output directory and will clobber each other's chunks; run them serially.

---

## Chrome's speech synthesis can wedge, and it lies about it

line-room does not speak any more — a scene you keep open while you read should
not talk at you — but the finding is worth keeping, because nothing about it is
visible from inside the page. Measured on a fresh load, focused tab, an
explicitly selected local voice (Samantha, `localService: true`):

```
speak()  ->  speaking = true       immediately
             pending  = false
             onstart  never fires
             onend    never fires
             speaking = true       still, minutes later
```

The engine accepts the utterance, reports itself busy forever, and produces no
audio. Every later `speak()` queues behind the stuck one and is swallowed too,
so the failure is permanent and silent. `speechSynthesis.cancel()` clears the
queue; only restarting Chrome restores audio.

If you ever put a voice back in, three things are not optional:

- **Never trust `onstart`/`onend`.** Poll `speaking`/`pending`, and keep a hard
  ceiling so the UI cannot be left mid-animation. Chrome skips `onstart`
  entirely for short utterances even when it is working.
- **Hold a reference to every live utterance.** Chrome collects utterances that
  nothing in JS still references, and a collected one goes quiet mid-word.
- **Do not `cancel()` and `speak()` in the same task** — the new utterance is
  lost. Cancel only when something is genuinely in flight, and give it a tick.

Note that a backgrounded tab keeps `speaking` true legitimately, so any wedge
check has to stand down on `document.hidden` — which also means **the check
cannot be verified through browser automation**, since automation backgrounds
the tab. Override the getter to exercise it.

---

## Chrome's focus ring was the only blue on the page

`outline: auto` renders as `rgb(0, 95, 204)`. Around a 96px circular dial it
draws a blue rounded *square* — in a black-and-white line drawing it reads as
debris rather than as focus. `:focus { outline: none }` plus a focus indicator
in the page's own vocabulary (dashed rim on the dial, inverted tray button)
keeps keyboard access without the colour.

---

## `vector-effect: non-scaling-stroke` silently breaks `stroke-dasharray`

The dial's rim came up permanently half-drawn — and only on a retina display.

Chrome measures the dash pattern in **device pixels** when the element has
`non-scaling-stroke`, while the path length stays in user units, and it ignores
`pathLength` (tested: setting `pathLength="100"` with `stroke-dasharray: 100`
made it *worse*, not length-agnostic). The rim is 294.8 user units drawn at
96/100 scale, so on a 2x screen it is 566 device px, and the 296 dash covered
52% of it.

```
294.8 units  ->  283.0 css px  ->  566.1 device px      dash 296 = 52%
```

At `devicePixelRatio: 1` the same CSS draws the full circle, which is exactly
why this survives review on one machine and not another. Any element with a
dash pattern has to opt out of `non-scaling-stroke`; at this scale (0.96) the
hairline is unchanged to the eye.

---

## Invisible hit boxes go wrong quietly, and only on some screens

Objects are picked by raycasting against invisible proxy boxes, because hitting
a 1px line with a mouse is hopeless. The cost is that the thing you click and
the thing you see are two different shapes, and nothing on screen tells you when
they disagree — the object simply stops responding and you assume you missed.

Three were found this way, all by measurement rather than by clicking around:

```
414x896   ceiling_light < fan     the light switch, i.e. the main control
414x896   door          < fan
1440x788  curtains      < mobile
```

The ceiling fan's proxy was a 4 x 1 x 4 slab at ceiling height. On a phone the
camera pulls a long way back to fit the room in a narrow frame, the projection
flattens toward orthographic, and that slab sweeps across the entire back wall.
It was fine on the machine it was drawn on and broken on a phone.

The diagnostic that finds them: for every device, project its proxy's bounding
box to screen, sample a grid across that footprint, raycast each sample, and
count how many come back with *this* device as the nearest hit. Run it at
several viewport sizes. A device that wins zero samples is unclickable.

Aiming only at the centre is too strict — the globe genuinely sits in front of
the middle of the cabinet, and the cabinet is still clickable across the rest of
its face (24 of 58 samples). Count area, not centres, or you will shrink boxes
that were never broken.

---

## Folding modules into one file collapses their scopes

Obvious in hindsight, and it still bit within the hour. Adding a local constant
to the framing code:

```js
const FILL = 0.94;      // how much of the frame the room may span
```

`FILL` is also the shared white fill material, defined at the top of the file
and used by every object in the room. Inside one module that is two names in two
scopes; inside one file it is one name, and the room booted to
`Cannot read properties of undefined (reading 'set')` from a line that had not
been touched.

ES modules make each file a scope for free and hide this class of collision
completely. A single-file page has one scope, so a new name has to be checked
against the whole file, not the section you are editing. The merge itself was
safe — every top-level name across the thirteen modules was verified unique
before they were concatenated — but that guarantee expires the moment you add
the next line.

Two things make it survivable rather than mysterious:

- **Wrap the whole app in try/catch and print the error onto the page.** A blank
  white page is the worst failure a demo can have, because it looks like nothing
  and nobody reports it. This one named its file, line and message immediately.
- **Prefix names that are only meaningful locally** — `FRAME_FILL`, not `FILL`.
