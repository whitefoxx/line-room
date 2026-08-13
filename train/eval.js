/* Browser eval harness. Paste into the console of a running line-room, or run
 * it through an automation tool; it drives window.__lineRoom.ask, which hits
 * the real wasm engine and returns raw tool calls without touching the room.
 *
 * Load a specific set of weights with ?weights=needle2-tuned.cact and
 * run the same suite twice to compare base against tuned. Scoring is exact
 * match on the set of calls, order-insensitive — the room does not care which
 * order two independent devices are set in.
 *
 *   await lineRoomEval()            -> summary object, also console.table'd
 */

const SUITE = {
  /* Phrasings that ARE in the training set. These measure whether tuning
     fixed the specific failures we set out to fix — nothing more. */
  targeted: [
    ['play some music', [['set_running', { target: 'record_player', on: true }]]],
    ['turn off the lights', [['set_light', { light: 'ceiling_light', on: false }]]],
    ['turn on the lights', [['set_light', { light: 'ceiling_light', on: true }]]],
    ['spin the globe', [['set_running', { target: 'globe', on: true }]]],
    ['shut the blinds', [['set_open', { target: 'blinds', open: false }]]],
    ['crank the fan to max', [['set_fan', { speed: 'high' }]]],
    ['open the window and put a record on',
      [['set_open', { target: 'window', open: true }], ['set_running', { target: 'record_player', on: true }]]],
    ['turn off the lights and close the door',
      [['set_light', { light: 'ceiling_light', on: false }], ['set_open', { target: 'door', open: false }]]],
  ],

  /* Phrasings that appear NOWHERE in training. This is the number that
     actually says whether anything generalised. */
  paraphrase: [
    ['kill the desk lamp', [['set_light', { light: 'desk_lamp', on: false }]]],
    ['hit the ceiling light', [['set_light', { light: 'ceiling_light', on: true }]]],
    ['dial the fan back to low', [['set_fan', { speed: 'low' }]]],
    ['wind the blinds up', [['set_open', { target: 'blinds', open: true }]]],
    ['swing the door open', [['set_open', { target: 'door', open: true }]]],
    ['shove the drawer shut', [['set_open', { target: 'drawer', open: false }]]],
    ['give me some tunes', [['set_running', { target: 'record_player', on: true }]]],
    ['kill the record', [['set_running', { target: 'record_player', on: false }]]],
    ['get that globe turning', [['set_running', { target: 'globe', on: true }]]],
    ['give the ball a whack', [['poke', { target: 'ball' }]]],
    ['chuck the cushion', [['poke', { target: 'pillow' }]]],
    ['seal the window', [['set_open', { target: 'window', open: false }]]],
    ['crack open the cupboard', [['set_open', { target: 'cabinet', open: true }]]],
    ['scoot the chair back out', [['set_open', { target: 'chair', open: true }]]],
    ['halt the pendulum', [['set_running', { target: 'wall_clock', on: false }]]],
    ['put a record on and kill the lights',
      [['set_running', { target: 'record_player', on: true }], ['set_light', { light: 'ceiling_light', on: false }]]],
  ],

  /* Must come back with no calls at all. Watched closely: a LoRA that pulls
     everything toward calling a tool would regress this, and that regression
     is worse than the failures we are trying to fix. */
  refusal: [
    ["what's the population of Tokyo?", []],
    ['remind me to call the dentist', []],
    ['turn on the air conditioning', []],
    ["what's my wifi password?", []],
    ['what is the capital of France?', []],
  ],
};

/* Synonym enum values resolve to the same device, so `target: "music"` and
   `target: "record_player"` drive identical animations. Score what the room
   does, not which spelling the model picked — otherwise the base model gets
   marked wrong for using a synonym we deliberately put in the schema. */
function deviceAliases(registry) {
  const map = new Map();
  for (const d of registry.list) {
    if (!d.alias) continue;
    map.set(d.alias, d.alias);
    for (const s of d.synonyms || []) map.set(s, d.alias);
  }
  return map;
}

const render = (aliases, name, args) => `${name}(${Object.keys(args || {}).sort()
  .map((k) => `${k}=${aliases.get(args[k]) ?? args[k]}`).join(',')})`;

const canon = (aliases, calls) => (calls || [])
  .map((c) => render(aliases, c.name, c.arguments))
  .sort().join(' + ');

const expectCanon = (aliases, pairs) => pairs
  .map(([name, args]) => render(aliases, name, args))
  .sort().join(' + ');

async function lineRoomEval({ suite = SUITE, verbose = true } = {}) {
  const room = window.__lineRoom;
  if (!room || !room.ready) throw new Error('model not ready yet');

  const aliases = deviceAliases(room.registry);
  const rows = [];
  const totals = {};

  /* Same frame cap the app uses during a turn, held for the whole run: at
     60fps the render loop takes ~40% of the worker's throughput, and a score
     sheet where latency depends on what the room happened to be animating is
     not worth reading. */
  room.setFrameCap(20);

  for (const [bucket, cases] of Object.entries(suite)) {
    let pass = 0;
    for (const [query, expected] of cases) {
      const r = await room.ask(query);
      const got = canon(aliases, r.calls);
      const want = expectCanon(aliases, expected);
      const ok = got === want;
      if (ok) pass++;
      rows.push({
        bucket,
        query,
        ok: ok ? 'pass' : 'FAIL',
        got: got || '(declined)',
        want: want || '(declined)',
        conf: Number(r.confidence).toFixed(2),
        ms: Math.round(r.ms),
      });
    }
    totals[bucket] = `${pass}/${cases.length}`;
  }

  room.setFrameCap(0);

  const passed = rows.filter((r) => r.ok === 'pass').length;
  totals.overall = `${passed}/${rows.length}`;
  totals.weights = room.weightsUrl.split('/').pop();
  totals.medianMs = rows.map((r) => r.ms).sort((a, b) => a - b)[Math.floor(rows.length / 2)];

  if (verbose) {
    console.table(rows);
    console.table(totals);
  }
  return { totals, rows };
}

window.lineRoomEval = lineRoomEval;
