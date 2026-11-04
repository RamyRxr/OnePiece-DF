# One Piece — Devil Fruit
## Project state, architecture, and work log

> **This is the living document. Read it before touching the code.**
>
> It explains what the project is, how it is built, where we stopped, and how
> we work. Every session that changes the project adds a dated entry to
> [§11 Work log](#11-work-log) and updates [§9 Where we stopped](#9-where-we-stopped).
> **Never delete history.** If something in here turns out to be wrong, correct
> it and say so in the entry; do not silently rewrite the past.

| | |
|---|---|
| Last updated | 2026-11-04 |
| Last feature commit | `18cd6b9` — *fire: harden re-light and thumb flame* |
| Working tree | Clean |
| Test suite | **267 passed**, no camera required |
| Current step | Ace's fire is lab-only: fist-open ignition, shake extinguish, sequential fingertip flames |
| Next step | Decide: put the fruit + fire on the protocol, or scaffold Unity (§12) |

---

## 1. What we are building

A One Piece Devil Fruit game. The player sits in front of a webcam, makes a
gesture, and a Devil Fruit power happens — first on a stylised avatar, later in
a game.

Two fruits exist today:

| Fruit | User | Power in the lab | Status |
|---|---|---|---|
| **Gum-Gum** (rubber) | Luffy | pinch a body part or finger, pull, release → elastic stretch | Mature perception + full lab effect |
| **Mera Mera** (fire) | Ace | fist then open to light, shake to put out, raise fingers one by one for fingertip flames, fist to swell, index up for a fireball | Lab effect done, not on the wire |

The long-term target is a Unity game driven by the same perception service.
The vision half is deliberately built and tuned **before** the avatar half,
because the hard problem is not stretching a rig — it is knowing *that the
player pulled their cheek, and by how much, from a 2D webcam*.

**The one-fruit invariant:** the lab never runs both powers at once. Key `f`
switches. The inactive fruit is not advanced, not drawn, and its state is
reset where needed, so gestures from the hidden fruit cannot fire.

---

## 2. The shape of the system

```
webcam ──► vision service ──► websocket ──► Unity ──► avatar
           (Python, 30Hz)    (~35 gameplay
                              numbers)
```

- The **webcam feed never leaves the machine.** Only derived, dimensionless
  numbers go on the wire.
- The **vision service** (`vision/`) is the only part that exists today. It
  captures, tracks, measures, calibrates, streams, and (in the lab window)
  draws the cosmetic effect.
- **Unity does not exist yet.** A reference client sits at
  `vision/protocol/Unity/StretchSignalClient.cs` and has never been compiled.
  Expect small fixes when it is first dropped into a project.
- **The lab window is a development tool, not the game.** Its effects are
  cosmetic. If an effect fails, the service must keep tracking and streaming
  and show the plain camera image.

---

## 3. Repository map

```
OnePiece-DF/
├─ README.md                     project pitch + roadmap (short)
├─ PROJECT-STATE.md              THIS FILE: architecture, state, rules, work log
├─ tools-explanation.md          unrelated: the machine's global Claude setup
└─ vision/
   ├─ README.md                 the vision service in depth (keep in sync)
   ├─ pyproject.toml            package, dependencies, pytest config
   ├─ models/                   MediaPipe .task/.tflite models (downloaded)
   ├─ records/                  recorded sessions and clips
   ├─ protocol/Unity/           reference C# websocket client (uncompiled)
   ├─ ofp_vision/
   │  ├─ config.py              EVERY tunable + channel lists + fruit config
   │  ├─ indices.py             landmark index constants (no deps)
   │  ├─ signals.py             landmarks → features (pure, no mediapipe)
   │  ├─ filters.py             OneEuroFilter + Slope
   │  ├─ channels.py            per-channel filter/activate/release state
   │  ├─ grab.py                pinch-grab-pull (pure, no mediapipe)
   │  ├─ gestures.py            fire signs: fist-open, shake, fist, point (pure)
   │  ├─ pipeline.py            frame → features → channels → message
   │  ├─ protocol.py            wire format (v3)
   │  ├─ landmarks.py           MediaPipe wrappers (the only mediapipe import)
   │  ├─ capture.py             camera acquisition, selected by NAME
   │  ├─ baseline.py            calibration capture + persistence
   │  ├─ synthetic.py           generated frames, incl. a synthetic grab
   │  ├─ server.py              capture loop + websocket + lab window + keys
   │  ├─ lab.py                 overlay window: landmarks, bars, status
   │  ├─ clips.py               record/replay real camera clips (key v)
   │  ├─ room.py                the picture behind the hands, remembered
   │  ├─ light.py               room light judged from the face
   │  ├─ skin.py                what counts as skin; hand silhouette
   │  ├─ skinmesh.py            skin grid + springs
   │  ├─ faceskin.py            face mesh glued to landmarks
   │  ├─ fingers.py             rubber finger as a chain/tube
   │  ├─ softbody.py            XPBD skin physics (numba)
   │  ├─ gpu.py                 draws the skin mesh (moderngl)
   │  ├─ stretch.py             ties mesh + physics + GPU together
   │  ├─ warp.py                CPU fallback stretch effect
   │  ├─ vfx.py                 release sparks in the lab
   │  ├─ fire.py                Ace's fire: lit hand, fingertip flames, embers, fireball
   │  └─ tools/                 fetch_models, probe, probe_client, replay, render_clip
   └─ tests/                    synthetic-landmark tests, no camera needed
```

**Why the layout matters:** `signals.py`, `grab.py` and `gestures.py` import
no MediaPipe. That keeps the risky math — the part that decides what the player
did — testable against fixtures and replayable offline.

---

## 4. Gum-Gum fruit — how perception works

### 4.1 The core problem

A webcam gives X and Y. A real pull is mostly a **Z** movement: grab your
cheek and tug sideways and your hand comes toward the lens while the 2D
landmarks barely move. Distance-based tracking reads almost nothing.

The fix is mathematical. Every measurement is divided by a **rigid,
body-local reference length**:

```
feature = |a - b| / scale_reference
```

- face → **interocular distance**
- hand → **palm length** (wrist to middle knuckle)
- body → **shoulder span**

Leaning in/out, turning, and moving closer all cancel. What survives is how
much the tissue stretched relative to the body it sits on.

Two more normalisations:

- **Baseline** — compare against a calibration snapshot of *this player's*
  neutral pose.
- **Asymmetry** — bilateral features also get `left/right − 1`, so a one-sided
  pull cancels scale in one step.

### 4.2 Calibration

- First run captures ~3 seconds of neutral pose.
- Frames where the player is **moving** are rejected. Stillness is scored per
  body group (face / each hand / pose), not averaged over all landmarks, so a
  single moving finger cannot slip through.
- The baseline is saved to `vision/baseline.json` and reused.
- Channels not visible during calibration (usually hands) learn a **lazy
  baseline** from their first stable frames.
- Press `c` to recalibrate, `r` to reset the baseline and recalibrate.

### 4.3 Channels and events

A **channel** is one dimensionless number the game can bind to a body part.
Each frame goes through:

```
raw feature → minus baseline → OneEuroFilter → value
                                          → Slope → velocity
                                          → activation / release
```

- **OneEuroFilter, not an EMA.** Heavy smoothing at rest, light smoothing when
  moving; the stretch tracks the hand without jitter.
- **Activation** is value past a threshold.
- **Release is the trigger.** A pull is a motion, so we watch velocity. The
  release fires edge-triggered when the value contracts fast enough or returns
  to rest, and carries the peak reached. One release per pull; the channel
  stays disarmed until it returns to rest.
- Timestamps are milliseconds on the wire; filters work in seconds.

Channel families:

| Group | Examples | Scale |
|---|---|---|
| `face.*` | cheek, ear, pullL/R, pullAsym, blendshapes | interocular |
| `hand.*` | `<digit>.reach / ext / straight`, meanExt, spread | palm |
| `body.*` | armExtend, armReach, shoulderRaise | shoulder |
| `grab.<hand>.*` | active, stretch, stretchX/Y, dirX/Y, target | interocular / palm |
| composites | `face.pullL/R` = norm of two rigid anchors | interocular |

The face **pull composites** exist because every single distance is blind to
one family of pull directions. Measuring the tragion against both the chin and
the forehead, then taking the norm, gives a channel that responds to a tug in
any direction.

### 4.4 Pinch-grab-pull (the primary interaction)

```
pinch  thumb tip meets index tip on a target   → grab
pull   move the pinched hand away              → grab.<hand>.stretch grows
let go open the pinch / pull past snap         → release event: fire
```

- Targets: ears, nose, chin, jaw sides, cheeks, mouth corners, lips, and any
  finger of the *other* hand.
- One hand or both; every grab channel names the pinching hand.
- Pinch hysteresis: closes below `pinch_on`, opens above `pinch_off`.
- A pinch only grabs **as it closes**; a hand already pinched grabs nothing.
- Tracking dropouts under `lost_grace_ms` keep the grab; longer cancels
  without firing.
- MediaPipe Left/Right labels are steadied first; a grab stays with the hand
  that made it, and two hands never share a label.
- Stretch is measured in interocular distances (palm lengths on a finger),
  and the anchor follows the face, so moving head + hand together is not a
  pull.

All thresholds live in `GrabConfig` (`config.py`).

### 4.5 Protocol (v3)

Local websocket at `ws://127.0.0.1:8765`, 30Hz, 35 gameplay channels.

```json
{
  "v": 3,
  "t": 15234,
  "ch": {
    "grab.Right.stretch": { "v": 0.6120, "m": 0.6120, "a": 1, "l": 0 },
    "face.cheekL": { "v": -0.1432, "m": 0.1432, "a": 1, "l": 0 }
  },
  "ev": [ { "ch": "grab.Left.stretch", "peak": 0.7311, "d": 1, "v": 0.0, "t": 15180 } ],
  "q": { "face": 1, "handL": 0, "handR": 1, "pose": 1, "fps": 30.0, "cal": 1 }
}
```

- `v` signed deviation from baseline; `m` magnitude (drive the rig with this)
- `a` active; `l` source lost (relax, do not freeze)
- `ev` release events with the peak that was reached
- `q` per-frame health flags

The game multiplies `m` by a per-channel gain to decide how many centimetres a
`0.1` pull becomes. Retuning game feel never touches perception.

**Important:** the active fruit is **not** on the wire yet. Both fruits' data
is still streamed. See §12.

---

## 5. Gum-Gum fruit — the lab effect

The lab draws the stretch on the camera image. It is cosmetic and must never
break tracking.

- A fine grid (~4,000 points) is laid over the skin around the pinch and
  **glued to the face's own landmarks** (moving least squares), so it follows
  head motion, mouth opening, smiles — without lagging like a mask.
- A **finger is a rubber tube**: a chain from knuckle to tip. The knuckle stays
  on the hand, the pinched part travels with the fingers, the middle stretches
  like rubber and thins a little.
- Only **real skin** moves: segmentation labels decide per pixel, per frame.
  Eyes are pinned, lips separated, teeth/hair/room never move.
- The room behind a moved hand is **remembered** and revealed, so a stretched
  finger leaves no ghost.
- Physics is XPBD on the GPU (`softbody.py`, `gpu.py`, `stretch.py`), painted
  with the player's own camera image and lit from its 3D shape.
- Fallback chain: GPU stretch → CPU `warp.py` → plain image. Failures are
  logged once and never stop the loop.
- Release fires a spark burst (`vfx.py`) so a let-go reads as an event.

The stretch **feel** lives in constants at the top of `softbody.py` (snap
speed, spread, damping). The fire's feel lives in `FireConfig`.

---

## 6. Mera Mera fruit — fire

Implemented in `gestures.py` (signs) and `fire.py` (effect), tuned by
`FireConfig`.

### 6.1 Signs (pure geometry, scale-invariant against palm length)

| Sign | Detection | Effect |
|---|---|---|
| **ignite** | a fist opens again within `ignite_window_ms` | light the hand |
| **shake** | rapid side-to-side wrist motion inside `shake_window_ms` | put the fire out |
| **fist** | all four fingers curled | the flame swells |
| **index_only** | index out, middle/ring/pinky curled, thumb anywhere | fireball above the fingertip |
| **snap** | thumb tip to index or middle tip closes, then opens within `snap_window_ms` | kept as a sign; no longer toggles the fire |

- Fingers use hysteresis (`out_*` to go out, looser `in_*` to stay out), so a
  sign on a threshold cannot flicker.
- **Ignition is edge-triggered:** the tracker remembers when a fist began; the
  first open frame inside the window fires once. A fist held longer is just a
  fist, not a delayed ignition. `ignite_cooldown_ms` stops a fast open/close
  from firing twice.
- **Shake** is measured on the wrist's horizontal position in palm lengths, so
  it is distance-invariant. It needs enough total travel (`shake_span`), enough
  significant steps (`shake_step`) and at least `shake_reversals` direction
  changes; a single sweep or ordinary drift cannot trigger it.
- A snap must be a **close then fast open**, and the snapped finger must come
  back out. That stops closing a fist from firing (the thumb brushes curled
  fingers, but no finger comes back out).
- `SignTracker` also refuses to re-arm while a pinch is still held: a pinch
  held longer than the snap window is not a snap, and letting go of it must
  not toggle the fire. This was a real bug fixed on 2026-11-04.

### 6.2 Effect

- The hand is **lit**, not painted over: its real silhouette (segmentation body
  skin, or the landmark glove as fallback) gets a hot rim, a faint body tint
  and flame streaks rising off its edge. The camera image of the hand stays
  readable under the fire.
- Each earned fingertip carries a teardrop flame, drawn with additive blending
  and its own flicker/embers.
- **Sequential fingertips:** opening the whole hand from a fist lights only the
  first finger; the count can only grow by exactly one. Lowering a finger hides
  its flame, but the earned count persists. The thumb is the capstone: it lights
  only after all four fingers, so all five cannot come on at once.
- Ignition: flash + ember burst. Extinguish: smoke puff + falling embers.
- The fireball floats above the index tip, follows it with lag, and goes out
  when the finger closes.
- Particles ride the hand's translation in image space; flames still rise.
- Everything advances on `Frame.t_ms`, never wall time, so recordings replay
  identically and tests can step it.
- `Fire.reset()` puts every flame out and forgets embers — used when the fruit
  is switched away.

### 6.3 One fruit at a time (added 2026-11-04)

- `Config.fruit.active` picks the starting fruit (`"rubber"` default).
- Key **`f`** toggles live between `"rubber"` and `"fire"`.
- `Lab._stretched()` advances and draws **only** the active effect.
- Switching away from fire calls `Fire.reset()`.
- In fire mode the Gum-Gum grab rings and pull lines are hidden
  (`LabOverlay.compose(show_targets=False)`); the hand landmarks stay.
- The status panel shows the active fruit and its controls.

---

## 7. How to run it

```powershell
cd vision

# one-time setup
uv venv --python 3.15 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m ofp_vision.tools.fetch_models

# the lab
.venv\Scripts\python.exe -m ofp_vision.server

# headless (no window)
.venv\Scripts\python.exe -m ofp_vision.server --no-window

# no camera at all: pipeline/protocol check
.venv\Scripts\python.exe -m ofp_vision.server --synthetic

# phone as camera (bypasses DirectShow)
.venv\Scripts\python.exe -m ofp_vision.server --url http://192.168.1.42:8080/video

# record / replay
.venv\Scripts\python.exe -m ofp_vision.server --record records/pull-cheek.jsonl
.venv\Scripts\python.exe -m ofp_vision.tools.replay records/pull-cheek.jsonl

# tests
.venv\Scripts\python.exe -m pytest -q
```

### Lab keys

| Key | Action |
|---|---|
| `c` | Recalibrate (3s neutral capture) |
| `r` | Reset baseline and recalibrate |
| `h` | Clean view: hide landmarks and grab drawings |
| `f` | **Switch devil fruit: rubber ↔ fire** |
| `v` | Start / stop a camera clip |
| `q` | Quit |

### Environment traps (this machine)

- Windows camera privacy can block every desktop app
  (`ConsentStore\webcam\NonPackaged = Deny`). Turn on
  **Settings → Privacy & security → Camera → "Let desktop apps access your camera"**.
- **OBS Virtual Camera** enumerates before the real webcam and silently returns
  a frozen frame. `capture.py` selects by **name**, never by index.
- **MediaPipe's GPU delegate is broken in the Windows wheels.** CPU (XNNPACK)
  is ~27ms/frame for face+hand+pose; that is fine. Do not fight it.
- Pins: `numpy>=2.5` (2.2.x cp315 wheel crashes on import),
  `opencv-contrib-python<5` (OpenCV 5 cannot open this webcam).

---

## 8. How we work

These are the rules the project is built by. Follow them unless a session
explicitly decides otherwise — and if you break one, say why in the work log.

### 8.1 Before coding

1. **Read this file first**, then the relevant module and its tests.
2. **Read the code you are about to change**, not just the function name.
   Effects and perception interact in non-obvious ways (e.g. hand labels,
   frame timing, fallbacks).
3. **Reproduce the current state.** Run the targeted tests. If something is
   already red, fix or record it before adding new work.

### 8.2 While coding

4. **Tunables go in `config.py`.** No magic thresholds in the code paths.
   New feel → new field in the relevant config dataclass.
5. **Keep the risky math pure.** Landmark → feature code (`signals.py`,
   `grab.py`, `gestures.py`) imports no MediaPipe and is tested against
   synthetic fixtures. MediaPipe stays in `landmarks.py`.
6. **Cosmetic effects must never break the service.** Wrap effect updates and
   draws so a failure logs once and falls back to the plain image. Tracking
   and streaming are sacred.
7. **One fruit at a time.** Any new fruit or power must join the switch, not
   run alongside the others. Update the invariant in `Lab` and the status UI.
8. **Respect the frame clock.** Anything animated or timed advances on
   `Frame.t_ms`, never `time.time()`/wall clock, so recordings replay.
9. **Scale-invariance is a requirement.** New features divide by a rigid
   body-local reference. Add a test that multiplies every landmark by 2 and
   asserts the feature is unchanged where applicable.
10. **Do not mutate shared state needlessly.** Prefer returning new values;
    hot per-frame arrays may be updated in place for speed, but say so.
11. **Small, focused changes.** One behaviour per change. Do not mix a
    refactor with a feature or a tuning pass.

### 8.3 Verification

12. **Test first when practical** (red → green → refactor). At minimum, add a
    test that would fail without the change.
13. **Run the targeted tests, then the full suite.** The full suite is
    ~2 minutes; do not skip it before declaring done.
14. **No camera needed for tests.** Everything must be verifiable with
    synthetic landmarks and fixtures.
15. **If the change affects feel or thresholds, record/replay real footage**
    (`--record`, `tools.replay`, `tools.render_clip`) rather than guessing.

### 8.4 After coding

16. **Update docs in the same change:**
    - `vision/README.md` for perception/protocol/effect behaviour and keys.
    - `README.md` for pitch/status/roadmap.
    - **This file**: add a dated entry to §11, update §9, and bump the header
      table (last commit, working tree, test count, current/next step).
17. **Report what changed, what was verified, and what is still open.**
    Do not claim "done" if the wire/game half is not done.
18. **Commit** once the suite is green. The history is informal so far; keep
    messages short and factual (e.g. *start work on fire df*). Do not commit
    a red suite.

### 8.5 Adding to this file

Every session appends an entry at the **top** of the log (newest first), using
this template:

```markdown
### YYYY-MM-DD — <short title>

**Goal:** what we set out to do.

**What changed:**
- `path/file.py` — what and why.
- ...

**Decisions:** anything we chose, and the alternative we rejected.

**Verification:** exact commands + result (e.g. `pytest -q` → 255 passed).

**Open / next:** what is still unfinished, and what the next session should
pick up.

**Commit:** hash + message, or "uncommitted".
```

Also update, in the same session:

- the header table at the top (date, commit, working tree, tests, current/next)
- §9 (done / not done / known gaps)
- §12 (backlog) if priorities changed

---

## 9. Where we stopped

**As of 2026-11-04 (after the fruit switch).**

### 9.1 Done and verified

- MediaPipe face + hand + pose + segmentation, CPU, ~30–40ms/frame.
- Scale-invariant feature maths, calibration with motion gating, lazy
  baselines, OneEuroFilter, activation/release state machine.
- Pinch-grab-pull end to end: one or both hands, fingers grabbed where
  pinched, hysteresis, dropouts, relabelling, snap distance.
- 35-channel gameplay wire protocol (v3) + reference Unity client
  (**uncompiled** — no Unity project yet).
- GPU skin-physics stretch effect with CPU fallback and full failure
  isolation.
- Ace's fire: fist-open ignition, shake extinguish, sequential fingertip
  flames, fist swell, index fireball, lit-hand rendering, embers, smoke,
  frame-clock driven.
- **One-fruit switch:** `f` toggles rubber ↔ fire; only the active effect is
  advanced and drawn; fire is reset on the way out; Gum-Gum rings hidden in
  fire mode.
- **267 tests pass** with no camera.

### 9.2 Not done

- **No Unity project.** No avatar, no game, no compiled C# client.
- **Fire is lab-only.** `fire.<hand>.*` is not in `GAMEPLAY_CHANNELS`, not in
  `protocol.py`, and not in the Unity client. The active fruit is not on the
  wire either.
- **No real-face recording has been tuned yet.** The grab thresholds are first
  guesses. The roadmap's "Signal Lab tuning" step is still open.
- **Face-deformation channels are unproven.** `face.cheek*`, `face.ear*` are
  secondary until real recordings show they respond. The ear proxy is the
  weakest signal and may be cut.
- **Packaging.** The vision service is not shipped as a companion app; players
  would still need Python.
- **Fire polish.** No visual regression tests for the fire (state and a
  draw-smoke test only); no true perspective, no skin detail at real size for
  the stretch. The new ignite/shake windows and fingertip thresholds have not
  been tuned against a real camera yet.

### 9.3 Known gaps / traps

- The root `README.md` and `vision/README.md` have repeatedly drifted behind
  the code. This file is the source of truth; when they disagree, trust this
  file and fix them.
- The snap re-arm bug (fixed 2026-11-04) showed how easy it is for a gesture
  to fire on the wrong edge. Any new gesture needs a held-input test.
- Effects are coupled to the lab window. The headless service never draws, so
  effect changes are invisible to `--no-window` and to the wire.
- The fire's count deliberately persists while fingers are lowered. If a real
  hand cannot produce a clean one-finger pose, tune `out_*`/`in_*` before
  changing the sequence.
- `tools-explanation.md` is **not** project documentation; it describes the
  machine's global Claude toolset. Ignore it when working on the game.

---

## 10. Decisions worth remembering

- **Read the hand, not the face.** A pinch is reliable through a webcam; a
  cheek deforming is not. The grab is the primary signal; face deformation is
  secondary until proven.
- **The hard part is the camera, not the fruit.** Hence perception and the
  Signal Lab before avatar work.
- **Ratios, not distances.** Every measurement divides by a rigid body-local
  length.
- **The vision service emits dimensionless numbers.** Unity decides what a
  value is worth in centimetres. Perception and game feel stay decoupled.
- **Calibration rejects movement.** A settling player would poison the
  baseline for the whole session.
- **Release is velocity, not pose.** A pull is a motion; the release event
  carries the peak.
- **One fruit at a time.** A fruit is a mode, not an additive layer. The
  switch is the model for every future fruit.
- **Effects are cosmetic and disposable.** A broken effect must never cost a
  frame of tracking or a message on the wire.

---

## 11. Work log

Newest first. Do not delete entries.

### 2026-11-04 — Ace's fire: fist-open, shake, sequential fingertips

**Goal:** make the fire read as fire on the hand rather than an orange mask,
and replace the snap toggle with the intended gestures: close then open to
light, shake to put out, and flames earned one fingertip at a time.

**What changed:**
- `vision/ofp_vision/config.py` — new `FireConfig` fields:
  `ignite_window_ms`, `ignite_cooldown_ms`, the `shake_*` window/span/step/
  reversals/cooldown, `finger_light_s`, fingertip flame size, finger ember
  rate, and separate hand/finger alpha.
- `vision/ofp_vision/gestures.py` — `HandSigns.ignite` and `.shake`;
  `SignTracker` detects a quick fist→open (edge-triggered, held fists do not
  fire) and a rapid side-to-side wrist shake measured in palm lengths. Ignition
  clears the shake window, so pre-ignition motion cannot put the new fire out
  on the next frame. Snap is still reported but no longer toggles the fire.
  Added `FLAME_FINGERS` with the thumb as the capstone.
- `vision/ofp_vision/fire.py` — replaced the snap toggle with `_light` and
  `_extinguish`; added per-finger state and the sequential count gate; added
  teardrop fingertip flames with additive blending, flicker and embers; changed
  `_draw_hand` from an opaque silhouette fill to a hot rim + faint body tint +
  flame streaks; re-lighting resets fingertip levels so it always starts at
  one; the thumb flame needs at least one finger still extended, so a fist
  cannot leave a lone thumb flame; `status()` now reports `fingers` and
  `flames`.
- `vision/ofp_vision/server.py` — fire-mode status hint and comment updated to
  the new gestures.
- `vision/tests/hand_fixture.py` — `moved()` helper for shake tests.
- `vision/tests/test_gestures.py` — tests for fist→open, a held fist, snap not
  igniting, one shake event, and normal movement not shaking.
- `vision/tests/test_fire.py` — rewritten for fist→open ignition, snap not
  lighting, one finger from an open hand, sequential earning, the jump gate,
  shake extinguish, and a draw-smoke test.
- `vision/README.md` — fire section and sign table rewritten.
- `README.md` — status entry and test count updated.

**Decisions:**
- Ignition requires the fist to open inside `ignite_window_ms`; holding a fist
  is not a delayed trigger. A cooldown prevents a fast open/close from firing
  twice.
- Shake uses the wrist's horizontal position in palm lengths, with a minimum
  travel, step size and direction-reversal count. A single sweep or ordinary
  drift cannot put the fire out.
- Fingertip count starts at one, can only grow by one, and persists while
  fingers are lowered. The thumb is the capstone (lights only after all four
  fingers) because an open hand often has the thumb out already; letting the
  thumb drive the count would let the player skip to five.
- Snap is kept as a sign (and its re-arm fix is preserved) but is no longer
  the fire trigger. Fire remains lab-only and is still not on the wire.

**Verification:**
- `pytest -q tests/test_gestures.py tests/test_fire.py` → **23 passed**.
- `pytest -q` → **267 passed** (no camera).

**Open / next:** tune `ignite_window_ms`, the shake thresholds and the finger
poses against a real camera; then put the fruit + fire on the protocol or
scaffold Unity (§12).

**Commit:** `c9cf48e` — *fire: fist-open, shake, sequential fingertips*, plus
small follow-up tweaks.

### 2026-11-04 — One fruit at a time (rubber ↔ fire switch)

**Goal:** stop the lab from running the Gum-Gum stretch and Ace's fire
simultaneously; add a switch between them.

**What changed:**
- `vision/ofp_vision/config.py` — added `FRUITS`, `FRUIT_LABELS`, and
  `FruitConfig.active` (default `"rubber"`); wired into `Config`.
- `vision/ofp_vision/server.py` — `Lab.fruit` state, `fruit_label`,
  `switch_fruit()`, and key **`f`**. `_stretched()` now advances/draws only
  the active fruit. The status panel shows the active fruit and its controls.
- `vision/ofp_vision/fire.py` — added `Fire.reset()`: put every flame out and
  clear embers when the fruit is switched away.
- `vision/ofp_vision/lab.py` — `LabOverlay.compose(show_targets=...)`; the
  Gum-Gum rings and pull lines are hidden in fire mode.
- `vision/ofp_vision/gestures.py` — fixed the snap re-arm bug: a pinch held
  past `snap_window_ms` disarmed but re-armed on the next still-closed frame,
  so releasing a long pinch toggled the fire. Added `_pinched` so it cannot
  re-arm until the pinch opens.
- `vision/tests/test_server.py` — only the active effect runs; switching away
  from fire resets it; `f` switches; fire mode hides the rings.
- `vision/tests/test_fire.py` — **new**: a synthetic snap lights the hand, and
  `reset()` leaves nothing to draw.
- `vision/README.md` — added the one-fruit rule, an Ace's-fire section, the
  `f` key, and `gestures.py`/`fire.py` in the layout.
- `README.md` — fire fruit added to the status list. (It also carries a run
  command snippet added by the user before this session.)

**Decisions:**
- The switch is **lab-only** for now. The wire still streams all channels; the
  active fruit will go on the protocol when Unity exists (see §12).
- The inactive fruit is not advanced at all (not just not drawn), so its
  gestures cannot fire and its state cannot drift.
- Fire is reset on the way out rather than left lit, so returning starts cold.
- `f` was chosen over a mouse button because the lab is a keyboard-driven
  OpenCV window.

**Verification:**
- `pytest -q tests/test_server.py tests/test_fire.py tests/test_gestures.py`
  → 16 passed.
- `pytest -q` → **255 passed** (was 249 passed + 1 failed).
- Manual: not yet run with a real camera after this change.

**Open / next:** put the fruit and fire on the wire, or scaffold Unity (§12).

**Commit:** uncommitted at the time of writing.

### 2026-11-02 — Start work on Ace's fire fruit

**Goal:** begin the second Devil Fruit: Ace's Mera Mera fire.

**What changed (commit `2560b1d`):**
- `fire.py` — new: hand silhouette as the flame body, rising particles,
  ignition flash/embers, extinguish smoke, floating fireball. Frame-clock
  driven.
- `gestures.py` — new: snap, fist, index-only signs; `SignTracker` with
  hysteresis, snap timing, cooldown.
- `config.py` — `FireConfig`.
- `server.py` — fire drawn in the lab; status line; `Lab` integration.
- `skin.py` — `hand_silhouette()`.
- `capture.py`, `tools/probe_camera.py` — phone/network camera support
  (`--url`), bypassing DirectShow.
- `vfx.py` — release sparks (lab feedback for a let-go).
- `fingers.py`, `stretch.py` — silhouette/tube changes.
- Tests for gestures, skin silhouette, fingers, vfx, grab tweaks.
- `vision/README.md` — phone camera docs and effect wording updates.

**Decisions:**
- Fire signs are coarse whole-hand shapes because that is what a webcam reads
  reliably at arm's length.
- The fire is lab-only; it is not added to the wire yet.
- The fire is cosmetic and isolated like the stretch: a failure must not stop
  tracking.

**Verification:** the suite ended at 249 passed + 1 failed. The failure was
`test_a_pinch_held_does_not_snap_when_it_lets_go` — a real re-arm bug, fixed
on 2026-11-04.

**Open / next:** the fire switch and the snap fix (done 2026-11-04), then the
wire.

**Commit:** `2560b1d` — *start work on fire df*.

### 2026-11-01 and earlier — the perception pipeline

**Goal:** build and verify the camera half before any avatar work.

**What changed:**
- MediaPipe face/hand/pose/segmentation tracking on CPU.
- Scale-invariant features, calibration with motion gating, lazy baselines,
  OneEuroFilter, activation/release channels.
- Pinch-grab-pull with both hands, finger grabs, hysteresis, dropouts,
  relabelling, snap distance.
- GPU skin-physics stretch effect, CPU fallback, room memory, lighting.
- Protocol v3 + reference Unity client (never compiled).
- Synthetic mode and the 249-test suite.

**Verification:** 249 tests, no camera. Synthetic pinch-grab verified the
chain end to end.

**Open / next:** real-face recording and `GrabConfig` tuning (still open).

**Commits:** `999e496`, `f155d0c`, `d85d6f8` (approximate split; see
`git log`).

---

## 12. Backlog / next steps

Ordered by what unblocks the most. Update this when priorities change.

1. **Put the fruit on the wire.** Add the active fruit to the protocol (e.g.
   `q.fruit` or a top-level `fruit`), and/or gate `GAMEPLAY_CHANNELS` so Unity
   receives only the active fruit's channels. Bump `PROTOCOL_VERSION` if the
   shape changes. Update `StretchSignalClient.cs`.
2. **Put fire on the wire.** Emit `fire.<hand>.{lit, level, fist, orb, fingers,
   flames}` and ignite/shake events (the `ev` list) so Unity can render the
   fire natively. Add them to `GAMEPLAY_CHANNELS` and the Unity client.
3. **Real-face tuning session.** Record yourself grabbing/pulling each target
   and snapping; replay; tune `GrabConfig` and `FireConfig`. Cut channels that
   do not respond. This is the roadmap's current step and the biggest unknown.
4. **Scaffold the Unity project.** Import the client, drive a test avatar,
   bind the grab channels to a stylised stretchy rig.
5. **Stylised rig.** Limb stretch via bone length; cheek/face via non-uniform
   head scale. Deliberately cartoon.
6. **Gameplay.** Pull to charge, release to fire; Gum-Gum Pistol from a
   finger, Bazooka from an arm. Fire: fist-open to light, shake to put out,
   raise fingers to build the flame, point to aim.
7. **Packaging.** Ship the vision service as a companion app so players do not
   need Python.
8. **Fire/stretch polish.** Visual regression tests, real-camera tuning of the
   ignite/shake/fingertip thresholds, true perspective toward the camera,
   wrinkles, skin detail at real size.

---

## 13. Quick reference

| Thing | Value |
|---|---|
| Python venv | `vision/.venv/Scripts/python.exe` |
| Run the lab | `cd vision; .venv\Scripts\python.exe -m ofp_vision.server` |
| Tests | `cd vision; .venv\Scripts\python.exe -m pytest -q` |
| WebSocket | `ws://127.0.0.1:8765`, protocol v3 |
| Gameplay channels | 35 |
| Current fruit default | `Config().fruit.active == "rubber"` |
| Fruit switch key | `f` |
| Fire activation | fist, then open within `ignite_window_ms` |
| Fire extinguish | rapid wrist shake (`shake_*`) |
| Fire flames | earned one finger at a time; thumb last |
| Calibration file | `vision/baseline.json` |
| Recordings | `vision/records/*.jsonl`, clips in `vision/records/clips/` |
| Reference client | `vision/protocol/Unity/StretchSignalClient.cs` (uncompiled) |
| Source of truth | this file |
