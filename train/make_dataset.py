"""Build the LoRA training set for line-room's five tools.

`needle generate-data` wants an OPENROUTER_API_KEY to have an LLM invent
examples. There isn't one here, and hand-authoring is the better trade anyway:
we already measured exactly which utterances the base model gets wrong, so the
set can be aimed at those instead of at a generic distribution.

Two files come out:

  data/train.jsonl   what the fine-tune sees
  data/heldout.json  evaluation phrasings that appear NOWHERE in training,
                     split into `paraphrase` (same intent, unseen wording) and
                     `inference` (the intent has to be worked out from a
                     complaint rather than named). Reporting "we fixed it"
                     using a sentence that was trained on would be worthless.

Every row carries the same five tools, because this is a specialist for one
room, not a general tool caller.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
TOOLS = json.loads((HERE / "tools.json").read_text())
SEED = 20260812

# ---------------------------------------------------------------- helpers

def light(which, on):
    return {"name": "set_light", "arguments": {"light": which, "on": on}}

def fan(speed):
    return {"name": "set_fan", "arguments": {"speed": speed}}

def opened(target, on):
    return {"name": "set_open", "arguments": {"target": target, "open": on}}

def running(target, on):
    return {"name": "set_running", "arguments": {"target": target, "on": on}}

def poke(target):
    return {"name": "poke", "arguments": {"target": target}}


# --------------------------------------------------------------- intents
# (phrasings, answers, reasoning). Phrasings are the training surface forms;
# unseen ones live in HELDOUT below.

INTENTS = [
    # ---- lights. "the lights" means the ceiling light, which the base model
    # ---- got wrong by reaching for the desk lamp.
    (["turn on the desk lamp", "desk lamp on", "switch the desk lamp on",
      "turn the desk lamp on", "light the desk lamp", "put the desk lamp on",
      "turn on the reading lamp", "lamp on"],
     [light("desk_lamp", True)], "'desk lamp' -> light 'desk_lamp'; 'on' -> on true"),

    (["turn off the desk lamp", "desk lamp off", "switch the desk lamp off",
      "turn the desk lamp off", "shut the desk lamp off", "put the lamp out",
      "turn off the reading lamp", "lamp off"],
     [light("desk_lamp", False)], "'desk lamp' -> light 'desk_lamp'; 'off' -> on false"),

    (["turn on the lights", "lights on", "turn the lights on", "switch the lights on",
      "turn on the ceiling light", "ceiling light on", "turn on the overhead light",
      "turn on the main light", "turn the room light on", "give me some light"],
     [light("ceiling_light", True)],
     "'lights' means the room's ceiling light -> light 'ceiling_light'; 'on' -> on true"),

    (["turn off the lights", "lights off", "turn the lights off", "switch the lights off",
      "turn off the ceiling light", "ceiling light off", "turn off the overhead light",
      "turn off the main light", "kill the lights", "lights out"],
     [light("ceiling_light", False)],
     "'lights' means the room's ceiling light -> light 'ceiling_light'; 'off' -> on false"),

    # ---- fan. enum words, never numbers.
    (["turn the fan off", "fan off", "stop the fan", "switch the fan off",
      "shut the fan off", "kill the fan", "turn off the ceiling fan"],
     [fan("off")], "'off' -> speed 'off'"),
    (["set the fan to low", "fan on low", "put the fan on low", "slow the fan down",
      "fan to its lowest", "just a little fan", "gentle fan"],
     [fan("low")], "'low' -> speed 'low'"),
    (["set the fan to medium", "fan on medium", "put the fan on medium",
      "fan halfway", "medium fan"],
     [fan("medium")], "'medium' -> speed 'medium'"),
    (["crank the fan to max", "fan on high", "set the fan to high", "turn the fan up",
      "fan at full blast", "max out the fan", "fan all the way up", "turn the fan on"],
     [fan("high")], "'max' -> speed 'high', the maximum"),

    # ---- open / close
    (["open the door", "open up the door", "get the door", "door open"],
     [opened("door", True)], "'door' -> target 'door'; 'open' -> open true"),
    (["close the door", "shut the door", "door closed", "close up the door"],
     [opened("door", False)], "'door' -> target 'door'; 'close' -> open false"),

    (["open the window", "open the window up", "window open", "crack the window open"],
     [opened("window", True)], "'window' -> target 'window'; 'open' -> open true"),
    (["close the window", "shut the window", "window closed", "close the window up"],
     [opened("window", False)], "'window' -> target 'window'; 'close' -> open false"),

    (["raise the blinds", "open the blinds", "blinds up", "pull the blinds up",
      "lift the blinds"],
     [opened("blinds", True)], "'raise' the blinds -> open true"),
    (["lower the blinds", "close the blinds", "shut the blinds", "blinds down",
      "drop the blinds", "pull the blinds down"],
     [opened("blinds", False)], "'shut' the blinds -> open false"),

    (["open the drawer", "pull the drawer out", "open the desk drawer", "drawer open"],
     [opened("drawer", True)], "'drawer' -> target 'drawer'; 'open' -> open true"),
    (["close the drawer", "shut the drawer", "push the drawer in", "drawer closed"],
     [opened("drawer", False)], "'drawer' -> target 'drawer'; 'close' -> open false"),

    (["open the cabinet", "open the cabinet up", "cabinet open", "open the cupboard"],
     [opened("cabinet", True)], "'cabinet' -> target 'cabinet'; 'open' -> open true"),
    (["close the cabinet", "shut the cabinet", "cabinet closed", "close the cupboard"],
     [opened("cabinet", False)], "'cabinet' -> target 'cabinet'; 'close' -> open false"),

    (["pull the chair out", "pull out the chair", "chair out", "slide the chair out"],
     [opened("chair", True)], "chair pulled out -> open true"),
    (["tuck the chair in", "push the chair in", "chair in", "put the chair back"],
     [opened("chair", False)], "chair tucked in -> open false"),

    (["take a book out", "pull a book out", "grab a book", "get me a book", "book out"],
     [opened("book", True)], "book pulled out -> open true"),
    (["put the book back", "shelve the book", "put the book away", "book back"],
     [opened("book", False)], "book returned to the shelf -> open false"),

    (["tilt the picture", "knock the picture crooked", "tip the picture"],
     [opened("picture", True)], "picture tilted -> open true"),
    (["straighten the picture", "level the picture", "fix the picture", "picture straight"],
     [opened("picture", False)], "picture straightened -> open false"),

    # ---- running. "music" must reach the record player: the base model
    # ---- refused "play some music" outright.
    (["play some music", "put some music on", "play music", "start the music",
      "put a record on", "play a record", "start the record player",
      "put on a record", "let's have some music", "music on"],
     [running("record_player", True)],
     "music comes from the record player -> target 'record_player'; play -> on true"),
    (["stop the music", "turn the music off", "stop the record", "music off",
      "stop the record player", "turn off the record player", "cut the music"],
     [running("record_player", False)],
     "music comes from the record player -> target 'record_player'; stop -> on false"),

    (["spin the globe", "start the globe", "give the globe a spin", "get the globe going",
      "set the globe spinning"],
     [running("globe", True)], "'spin' the globe -> on true"),
    (["stop the globe", "hold the globe still", "stop the globe spinning", "globe still"],
     [running("globe", False)], "'stop' the globe -> on false"),

    (["start the mobile", "spin the mobile", "get the mobile going", "start the chimes"],
     [running("mobile", True)], "'start' the mobile -> on true"),
    (["stop the mobile", "stop the chimes", "quiet the chimes", "still the mobile"],
     [running("mobile", False)], "'stop' the mobile -> on false"),

    (["start the clock", "get the clock ticking", "start the pendulum", "wind the clock up"],
     [running("wall_clock", True)], "'start' the clock -> on true"),
    (["stop the clock", "stop the ticking", "silence the clock", "stop the pendulum"],
     [running("wall_clock", False)], "'stop' the clock -> on false"),

    (["make the plant sway", "start the plant swaying", "get the plant moving"],
     [running("plant", True)], "'sway' the plant -> on true"),
    (["stop the plant swaying", "hold the plant still", "still the plant"],
     [running("plant", False)], "'stop' the plant -> on false"),

    (["make the coffee steam", "heat the coffee up", "start the steam", "fresh coffee"],
     [running("coffee_mug", True)], "steam from the mug -> target 'coffee_mug'; on true"),
    (["let the coffee cool", "stop the steam", "cool the coffee down"],
     [running("coffee_mug", False)], "steam from the mug -> target 'coffee_mug'; on false"),

    # ---- poke
    (["bounce the ball", "throw the ball", "give the ball a bounce", "drop the ball",
      "kick the ball"],
     [poke("ball")], "'bounce the ball' -> poke 'ball'"),
    (["toss the pillow", "throw the pillow", "flip the pillow", "fluff the pillow",
      "toss the cushion"],
     [poke("pillow")], "'toss the pillow' -> poke 'pillow'"),
]

# Multi-call. The base model dropped the second call whenever the tool count
# went up, so these carry weight.
MULTI = [
    (["open the window and put a record on", "put a record on and open the window"],
     [opened("window", True), running("record_player", True)],
     "two requests: open the window, and start the record player"),
    (["turn off the lights and close the door", "close the door and turn off the lights"],
     [light("ceiling_light", False), opened("door", False)],
     "two requests: ceiling light off, and the door closed"),
    (["raise the blinds and open the window"],
     [opened("blinds", True), opened("window", True)],
     "two requests, both open true"),
    (["turn on the desk lamp and put some music on"],
     [light("desk_lamp", True), running("record_player", True)],
     "two requests: desk lamp on, and the record player playing"),
    (["close the window and turn the fan on"],
     [opened("window", False), fan("high")],
     "two requests: window closed, fan to high"),
    (["turn off the lights and stop the music"],
     [light("ceiling_light", False), running("record_player", False)],
     "two requests: ceiling light off, record player stopped"),
    (["spin the globe and bounce the ball"],
     [running("globe", True), poke("ball")],
     "two requests: globe spinning, and a poke at the ball"),
    (["shut the blinds and close the window"],
     [opened("blinds", False), opened("window", False)],
     "two requests, both open false"),
    (["open the door and open the cabinet"],
     [opened("door", True), opened("cabinet", True)],
     "two requests, both open true"),
    (["turn the fan off and close the window"],
     [fan("off"), opened("window", False)],
     "two requests: fan off, window closed"),
    (["put the book back and close the cabinet"],
     [opened("book", False), opened("cabinet", False)],
     "two requests, both open false"),
    (["turn on the desk lamp and pull the chair out"],
     [light("desk_lamp", True), opened("chair", True)],
     "two requests: desk lamp on, chair pulled out"),
    (["stop the clock and stop the chimes"],
     [running("wall_clock", False), running("mobile", False)],
     "two requests, both on false"),
    (["open the window, set the fan to low"],
     [opened("window", True), fan("low")],
     "two requests: window open, fan to low"),
    (["toss the pillow and bounce the ball"],
     [poke("pillow"), poke("ball")],
     "two pokes"),
    (["lights on and put a record on"],
     [light("ceiling_light", True), running("record_player", True)],
     "two requests: ceiling light on, record player playing"),
]

# Implicit requests: a complaint that still maps onto exactly one control.
# Kept deliberately few — a 45M model taught to over-infer starts inventing.
IMPLICIT = [
    (["it's stuffy in here", "it's so stuffy", "the air is stale in here"],
     [fan("high")], "stuffy air -> move air -> fan to high"),
    (["it's too dark to read", "I can't see anything in here"],
     [light("ceiling_light", True)], "too dark -> ceiling light on"),
    (["it's freezing with that window open", "I'm cold, the window"],
     [opened("window", False)], "cold from the window -> close it"),
    (["too much glare on the screen"],
     [opened("blinds", False)], "glare from the window -> lower the blinds"),
]

# Out of scope. Enough to hold the refusal behaviour, not enough to make the
# model refusal-happy — the base already declines correctly and the risk here
# is teaching it to decline things it can actually do.
REFUSALS = [
    "what is the capital of France?", "who wrote Moby Dick?", "what's 17 times 23?",
    "what's the weather tomorrow?", "read me the news", "what time is it?",
    "tell me a joke", "how are you?", "who are you?", "thanks",
    "order me a pizza", "call my mother", "send an email to Sam",
    "turn on the TV", "start the dishwasher", "open the fridge",
    "book me a flight", "what's on my calendar?", "play a video",
    "set an alarm for 7am", "add milk to my shopping list", "vacuum the floor",
]

# Phrasings that appear NOWHERE above, for honest measurement.
HELDOUT = {
    "paraphrase": [
        ("kill the desk lamp", [light("desk_lamp", False)]),
        ("hit the ceiling light", [light("ceiling_light", True)]),
        ("dial the fan back to low", [fan("low")]),
        ("wind the blinds up", [opened("blinds", True)]),
        ("swing the door open", [opened("door", True)]),
        ("shove the drawer shut", [opened("drawer", False)]),
        ("give me some tunes", [running("record_player", True)]),
        ("kill the record", [running("record_player", False)]),
        ("get that globe turning", [running("globe", True)]),
        ("give the ball a whack", [poke("ball")]),
        ("chuck the cushion", [poke("pillow")]),
        ("seal the window", [opened("window", False)]),
        ("crack open the cupboard", [opened("cabinet", True)]),
        ("scoot the chair back out", [opened("chair", True)]),
        ("halt the pendulum", [running("wall_clock", False)]),
        ("put a record on and kill the lights",
         [running("record_player", True), light("ceiling_light", False)]),
    ],
    "inference": [
        ("I'm going to sleep", None),          # ambiguous on purpose: any sane
        ("there's a glare on my book", None),  # answer or a decline is fine;
        ("I want to hear something", None),    # scored by hand
        ("this room is too bright", None),
    ],
    "refusal": [
        "what's the population of Tokyo?",
        "remind me to call the dentist",
        "turn on the air conditioning",
        "what's my wifi password?",
    ],
}

# --------------------------------------------------------------- assembly

POLITE_PREFIX = ["", "", "", "", "please ", "can you ", "could you ", "hey, "]
POLITE_SUFFIX = ["", "", "", "", "", " please", " for me", ", thanks"]


def decorate(rng, text):
    """Light politeness noise so the model keys on content, not on sentence
    shape. Applied to a minority of rows on purpose."""
    prefix = rng.choice(POLITE_PREFIX)
    suffix = rng.choice(POLITE_SUFFIX)
    if "please" in text.lower():          # no "please kill the lights please"
        prefix, suffix = "", ""
    out = prefix + text + suffix
    return out[0].upper() + out[1:] if rng.random() < 0.15 else out


def build():
    rng = random.Random(SEED)
    rows = []

    def add(query, answers, reasoning=None):
        row = {"query": query, "tools": TOOLS, "answers": answers}
        if reasoning:
            row["reasoning"] = reasoning
        rows.append(row)

    # Multi-call gets more variants per phrasing: it is where the base model
    # failed hardest, and there are fewer natural ways to say it.
    for group, extra in ((INTENTS, 0), (MULTI, 3), (IMPLICIT, 1)):
        for phrasings, answers, reasoning in group:
            for phrase in phrasings:
                add(phrase, answers, reasoning)
                add(decorate(rng, phrase), answers, reasoning)
                for _ in range(extra):
                    add(decorate(rng, phrase), answers, reasoning)
                if rng.random() < 0.35:
                    add(decorate(rng, phrase), answers, reasoning)

    # Declining correctly is something the base model is already good at, and
    # a LoRA that pulls everything toward calling a tool would regress it
    # visibly. Enough rows to hold the behaviour, not enough to amplify it.
    for phrase in REFUSALS:
        add(phrase, [], "no tool in this room covers that")
        for _ in range(2):
            add(decorate(rng, phrase), [], "no tool in this room covers that")

    # guard: nothing we plan to measure on may appear in training
    trained = {r["query"].lower().strip(" ,.") for r in rows}
    leaked = []
    for bucket in HELDOUT.values():
        for item in bucket:
            query = item[0] if isinstance(item, tuple) else item
            if query.lower().strip(" ,.") in trained:
                leaked.append(query)
    if leaked:
        raise SystemExit(f"held-out phrasings leaked into training: {leaked}")

    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build()
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    with (out / "train.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "heldout.json").write_text(json.dumps(HELDOUT, indent=2))

    calls = sum(len(r["answers"]) for r in rows)
    refusals = sum(1 for r in rows if not r["answers"])
    multi = sum(1 for r in rows if len(r["answers"]) > 1)
    print(f"{len(rows)} rows -> {out/'train.jsonl'}")
    print(f"  {calls} tool calls, {multi} multi-call rows, {refusals} refusals "
          f"({refusals * 100 // len(rows)}%)")
    print(f"  held-out: {sum(len(v) for v in HELDOUT.values())} phrasings, none in training")
