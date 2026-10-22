# Vision service — stretch perception

The camera-driven half of the Devil Fruit project. It watches the player
through the webcam, works out how hard they are pulling on a body part, and
publishes that as a handful of numbers for Unity to turn into an elastic
avatar.

This is the **Signal Lab**: the point of this service right now is not to be
shippable, it is to prove the perception works and to make it tunable.

---

## The main interaction: pinch, pull, let go

```
pinch   thumb tip meets index tip on a target                  -> grab
pull    move the pinched hand away                             -> grab.<hand>.stretch grows
let go  open the pinch (or pull past the snap distance)        -> release event: fire
```

**Targets:** ears, nose, chin and both sides of the jaw, cheeks, mouth
corners and both lips - and any finger of your *other* hand (pinch your left
index with your right hand and pull). Nothing else can be grabbed.

**One hand or both.** Each hand grabs on its own: pull one cheek, or both
cheeks at once like Luffy. Every grab channel names the pinching hand
(`grab.Left.*`, `grab.Right.*`).

**Fingers are grabbed where you pinch them**, anywhere from the knuckle to
the tip, and that spot stays held as the hand moves. The pinch must land
*on* the finger - within 0.15 palm lengths of its centre line, about its own
width. A pinch in the air beside a finger grabs nothing.

Reading how a face deforms through a 2D webcam is fragile; reading what the
*hand* does is not, so the grab is the primary gameplay signal and lives in
[`grab.py`](ofp_vision/grab.py). The face-deformation channels below are
still computed and streamed, but treat them as secondary until real
recordings show they respond.

- The stretch is measured in **interocular distances** on the face and in
  **palm lengths** on a finger, so it means the same at any distance from
  the camera.
- The anchor **follows the face**: moving your head and hand together is not
  a stretch. Only the hand moving relative to the face is.
- The pinch has **hysteresis** (closes below 0.15 palm lengths, opens above
  0.25), so a wobbly pinch does not drop the grab.
- A pinch only grabs **as it closes**. A hand that is already pinched does not
  grab whatever it passes over.
- A tracking dropout under 200ms keeps the grab; a longer one cancels it
  *without* firing.
- MediaPipe sometimes swaps a hand's Left/Right label mid-pull, or gives both
  hands the same one. The labels are steadied first: a grab stays with the
  hand that made it, and two hands never share a label.

All of these are in `GrabConfig` in [`config.py`](ofp_vision/config.py).

In the lab, every face target is drawn as an orange ring; the fingers are
drawn as the hand's skeleton. Press **h** for a clean view without the
landmarks and grab drawings, to judge the stretch as it really looks.

The stretch is **skin physics drawn on the graphics card** - the way games
and AR filters do it:

- when you pinch, a fine grid of ~4,000 points is laid over the skin around
  the pinch, glued to your face (it moves, turns and scales with your head).
  Each hand gets its own; two pinches close together on the face hold the
  same grid;
- every point has mass, springs to its neighbours, and a soft tether to
  where it rests. Your fingers hold the pinched points - firmly at the
  centre of the pinch, gently at its rim - and the rest follows with a
  little **weight**. Let go and it **snaps back quickly and cleanly**, like a
  rubber band: back in 2 frames, still by the 3rd - no slime, no jiggle;
- the eyes are pinned (they never move), the lips are separate (the springs
  across the mouth opening are cut), and **only real skin moves**: a
  segmentation model labels every pixel (face skin, body skin, hair,
  clothes, background), so a face grab moves face skin (and a beard - hair
  below the nose, attached to the skin) and a finger grab moves only that
  finger's own pixels - within the finger's own width of its centre line,
  so never its neighbours, the air beside it, the wall, scalp hair or
  teeth. It runs in the background (~10 times a second) and does not slow
  the frame rate. An ear hidden under hair has no skin showing, so it does
  not stretch;
- what is visible is decided **per pixel** on the graphics card: every drawn
  pixel looks up where it came from in the skin mask and shows only if that
  spot was skin. However far the mesh stretches, the wall beside a finger
  or the room beside a face can never come along with it;
- the mesh is drawn on the GPU, painted with your own camera image and lit
  from its 3D shape (antialiased, ~11-22ms a frame at 720p).

The pieces: [`skinmesh.py`](ofp_vision/skinmesh.py) (the grid and its
springs), [`softbody.py`](ofp_vision/softbody.py) (the physics - XPBD,
compiled with numba), [`gpu.py`](ofp_vision/gpu.py) (drawing, moderngl),
[`stretch.py`](ofp_vision/stretch.py) (ties them together), with what counts
as skin in [`skin.py`](ofp_vision/skin.py). The feel is the constants at the
top of `softbody.py` (snap speed, spread, damping).

Where no OpenGL 3.3 context can be made, the lab falls back to the CPU
effect in [`warp.py`](ofp_vision/warp.py); if either ever fails, it logs it
and shows the plain picture - tracking and streaming carry on.

Still to come: the stretch coming toward the camera in true perspective
(bigger toward your hand), wrinkles where skin bunches, and skin detail
(pores, beard hair) kept at its real size instead of smeared.

---

## The problem this solves

A webcam gives you X and Y. A "pull" is mostly a **Z** movement — grab your
cheek and tug sideways, and your hand comes *toward* the camera while the 2D
landmarks barely move. A naive "distance from cheek to mouth" tracker reads a
few pixels of stretch for a huge physical pull, and reads nothing at all if
you pull straight at the lens.

No camera or depth model fixes that. Depth sensors cost money, are noisy at
face range, and break the "works on any laptop" promise. The fix is
mathematical:

```
feature = |a - b| / scale_reference
```

where `scale_reference` is a rigid distance on the same body part
(interocular distance, palm length, shoulder span). Dividing by it cancels
the player leaning toward or away from the camera, turning their head, and
moving their hand closer to the lens. What survives is the thing we actually
care about: how much the tissue has stretched relative to the body it sits on.

Two more normalisations make it work:

- **Baseline.** Every feature is measured against a calibration snapshot of
  *your* neutral pose. No two faces pull the same way.
- **Asymmetry.** Bilateral features (cheek, ear) also get a `left/right - 1`
  ratio. A one-sided pull is literally a bilateral asymmetry, so this cancels
  scale effects in a single step.

---

## Setup

```powershell
cd vision
uv venv --python 3.15 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m ofp_vision.tools.fetch_models
```

### Windows will probably say no

If the camera fails to open, check **Settings → Privacy & security → Camera →
"Let desktop apps access your camera"**. When that is off, every desktop app
gets `E_ACCESSDENIED` from DirectShow and the error looks like a broken
driver. It is not a broken driver.

The Windows Camera app itself can also be blocked separately, which makes it
look like the hardware is dead. It isn't.

### Pinned dependencies, and why

| Pin | Reason |
|---|---|
| `numpy>=2.5` | numpy 2.2.x ships a MinGW-built cp315 wheel that hard-crashes on import. |
| `opencv-contrib-python<5` | OpenCV 5's rewritten UVC backend could not open this laptop's webcam at all. |
| `mediapipe` | **The GPU delegate does not work on Windows.** It fails at graph init with *"ImageCloneCalculator: GPU processing is disabled in build flags"* for all three models. We run CPU (XNNPACK): **~27ms/frame** for face+hand+pose combined, comfortably inside a 30fps budget. Don't fight it. |

---

## Running the lab

```powershell
.venv\Scripts\python.exe -m ofp_vision.server
```

A window opens sized to your screen: the camera (captured at 1280x720) on
top, and a panel underneath with the status on the left and a live bar per
channel. A bar's length is how far the channel is from neutral; the number
beside it keeps the sign. Green means the channel is past its activation
threshold (the grey tick). Drag the window's corner to resize it.

| Key | Action |
|---|---|
| `c` | Recalibrate (3s neutral capture) |
| `r` | Reset the baseline and recalibrate |
| `h` | Clean view: hide the landmarks and grab drawings |
| `q` | Quit |

On first run it calibrates automatically and saves `baseline.json`. Sit
still, face the camera squarely, and keep your hands up in view if you can.
A channel that was not visible during calibration (usually the hands) learns
its own baseline from the first half-second it is seen, so it works either
way - but a calibration taken with your head turned reads as a permanent
pull. If the bars sit away from the centre line at rest, press `r`.

```powershell
# headless, no window
.venv\Scripts\python.exe -m ofp_vision.server --no-window

# different camera
.venv\Scripts\python.exe -m ofp_vision.server --device "Integrated Camera"

# pipeline/protocol check with no camera at all
.venv\Scripts\python.exe -m ofp_vision.server --synthetic
```

### Reading the bars

The question a bar answers is *"does this number move when I do the thing?"*
If a bar is flat while you pull your cheek, that channel is dead weight and
gets cut from the gameplay list rather than shipped as noise.

---

## Record and replay

This is the single biggest speed-up on the project. You will not re-do
"pull my cheek" fifty times to tune a threshold.

```powershell
# record 30 seconds of yourself doing the thing
.venv\Scripts\python.exe -m ofp_vision.server --record records/pull-cheek.jsonl

# replay it offline and get a table of which channels actually responded
.venv\Scripts\python.exe -m ofp_vision.tools.replay records/pull-cheek.jsonl
```

```
channel                             range    mean     std      p5      p95    peak
hand.Right.index.reach             0.4218  0.1904  0.1482  0.0012  0.3891  0.4218
face.cheekL                       0.2615  0.1002  0.0901  0.0004  0.2210  0.2615
face.earAsym                      0.3000  0.1400  0.1203  0.0000  0.2601  0.3000
```

Channels with a dynamic range under 0.005 are listed separately as dead.
Replay runs the recording through the same [`pipeline.py`](ofp_vision/pipeline.py)
as the live server, so it also lists every grab release and its peak
stretch - that is how to tune `GrabConfig` without re-doing the gesture.

Other flags: `--channel face.cheekL --watch` to see a live bar chart of one
channel, `--csv out.csv` to dump the whole time series.

---

## Channels

Values are **dimensionless** — a fraction of a body-derived reference length,
not pixels or metres. Unity multiplies by a per-channel gain to decide how
many centimetres `0.1` of cheek-pull becomes. Perception and game-feel stay
decoupled: retuning the stretch length never requires touching the vision.

### Grab (the main interaction)

| Channel | Meaning |
|---|---|
One set per pinching hand, `<hand>` being `Left` or `Right`:

| Channel | Meaning |
|---|---|
| `grab.<hand>.active` | 1 while that hand holds a grab |
| `grab.<hand>.stretch` | pull length, in interocular distances (palm lengths for a finger) |
| `grab.<hand>.stretchX/Y` | pull vector, same units. +x = player's right, +y = down |
| `grab.<hand>.dirX/Y` | unit vector of the pull |
| `grab.<hand>.target` | 0 none, 1 cheekL, 2 cheekR, 3 earL, 4 earR, 5 nose, 6 chin, 7 jawL, 8 jawR, 9 mouthL, 10 mouthR, 11 lipTop, 12 lipBottom, 20-24 left thumb..pinky, 25-29 right thumb..pinky |

These are passed through unfiltered and unbaselined: they are already zero
at rest, and `.target` is a category code that must never be smoothed. On
the release frame `.target` still names what was let go.

### Face (`s = interocular distance`)

| Channel | Measures | Notes |
|---|---|---|
| `face.cheekR/L` | mouth corner → cheekbone silhouette point | main cheek-pull signal |
| `face.cheekAsym` | left/right cheek ratio − 1 | scale-free by construction |
| `face.cheekSquintR/L` | blendshape | cheap and surprisingly reliable |
| `face.earR/L` | nose → cheekbone silhouette point | **ear proxy, see below** |
| `face.earAsym` | left/right ear ratio − 1 | |
| `face.earVertR/L` | chin → cheekbone point, vertical | catches upward pulls |
| `face.mouthOpen`, `face.jawOpen` | lip separation, blendshape | |
| `face.smileR/L`, `face.mouthPucker`, `face.mouthFunnel` | blendshapes | |
| `face.pullL/R` | norm of the tragion→chin and tragion→forehead deviations | fires for a tug in **any** direction; suppressed while the jaw is open |
| `face.pullAsym` | `pullL − pullR` | |

### Hands (`s = palm length`)

`hand.{Left|Right}.{thumb|index|middle|ring|pinky}.{reach|ext|straight}`

- `reach` — fingertip to wrist, relative to palm length. Responds to both
  straightening the finger and pulling it sideways.
- `ext` — fingertip to its own knuckle. The cleanest "how stretched is this
  finger" measure.
- `straight` — segment-angle cosine. 1.0 = perfectly straight finger.

Plus `hand.{side}.meanExt` and `hand.{side}.spread`.

### Body (`s = shoulder span`)

`body.armExtend{L,R}`, `body.armReach{L,R}`, `body.shoulderRaise{L,R}`

### About the ear

MediaPipe has **no ear landmarks**. The closest thing is the cheekbone
silhouette point (index 234 / 454), which is where the ear physically
attaches — pulling the ear pushes that point outward, usually upward.

This is the weakest channel in the set and it is labelled as such. If it does
not hold up on a real face, cut it. The gameplay works fine without it.

---

## How a pull becomes an event

```
raw feature → minus baseline → OneEuroFilter → value
                                          → Slope    → velocity
                                          → activation / release
```

**OneEuroFilter, not an EMA.** An EMA has a fixed jitter-vs-lag tradeoff.
OneEuro adapts its cutoff to signal speed: heavy smoothing when your finger
is still (no jitter), light smoothing when it moves (no lag). The whole feel
of the mechanic depends on the stretch tracking your hand tightly.

**Release is the trigger.** A pull is a motion, not a pose, so we watch the
*velocity*. While you pull, the value climbs; the instant you let go, it
drops. The release event fires on that contraction, edge-triggered, and
carries the peak value reached since the pull started — so the game knows how
hard it was yanked.

Release fires when either:

- the value relaxes all the way back to rest, or
- the value contracts by at least `4 × activate_threshold` per second while
  below its peak.

The velocity threshold is expressed as a *ratio* of the activation threshold,
not an absolute number, so it means the same thing for a cheek feature
(range ~0.05) and a finger reach feature (range ~3).

After a snap-back release the channel stays disarmed until it has returned
to rest, so the tail of one pull cannot read as a second one.

Timestamps on the wire are milliseconds; the filters work in seconds (their
cutoffs are in Hz, the release velocity is per second). `ChannelManager`
converts once, at the top of `update()`.

---

## Calibration

The first three seconds are a neutral capture. Frames where the player is
moving are **rejected** — otherwise someone still leaning in and settling
poisons the baseline and every channel reads permanently offset.

Stillness is measured as the largest per-body-part landmark displacement
since the previous frame (75th percentile within each group). Scored per
group rather than averaged across all landmarks, because averaging dilutes
exactly the case we need to catch — a single finger moving while the rest of
the body holds still — down below the noise floor.

The baseline is saved to `baseline.json` and reused next session.

---

## Protocol

A local websocket on `127.0.0.1:8765`, protocol version 3 (version 3 gave
every grab channel its hand). The gameplay channels
(`config.GAMEPLAY_CHANNELS`, 35 of them) at 30Hz.

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

- `v` — value (deviation from baseline, dimensionless, signed)
- `m` — magnitude, `|v|`. A pull is unsigned; drive the rig with this
- `a` — 1 if the channel is past its activation threshold (grab channels: 1 while a grab is held)
- `l` — 1 if the source left the frame (the game should relax, not freeze)
- `ev` — release events: `peak` is the magnitude of the pull that just ended,
  `d` the direction it went (+1/−1). A pinch-grab releases on its hand's
  `grab.<hand>.stretch`
- `q` — per-frame health flags, so the game degrades gracefully

Reference Unity client: [`protocol/Unity/StretchSignalClient.cs`](protocol/Unity/StretchSignalClient.cs).
Its socket runs on a background thread but only queues raw messages; parsing
and every event fire on Unity's main thread in `Update()`, so handlers may
touch GameObjects freely. *It has not been compiled yet - there is no Unity
project. Expect to fix small things when it is first dropped in.*

```powershell
# watch the stream
.venv\Scripts\python.exe -m ofp_vision.tools.probe_client --watch
```

---

## Layout

```
ofp_vision/
  indices.py     landmark index constants      (no dependencies)
  signals.py     landmarks → features          (pure, no mediapipe)
  filters.py     OneEuroFilter + Slope
  channels.py    per-channel filter / activate / release state machine
  grab.py        pinch-grab-pull, the main interaction  (pure, no mediapipe)
  pipeline.py    features → calibration → channels → message, shared by server and replay
  landmarks.py   MediaPipe wrappers            (the only mediapipe import)
  capture.py     camera acquisition, device selected by NAME
  baseline.py    calibration capture + persistence
  protocol.py    wire format
  lab.py         overlay window
  light.py       the room's light, judged from the face (lights stretched skin)
  warp.py        the stretch effect: elastic-skin physics, lighting, rendering (cosmetic)
  skin.py        what counts as skin: face, ears, fingers, mouth, skin colour
  skinmesh.py    the skin mesh: a grid glued to the face or finger, and its springs
  softbody.py    skin physics: mass, springs, tethers (XPBD, numba)
  gpu.py         draws the skin mesh on the graphics card (moderngl)
  stretch.py     the skin-physics stretch effect: mesh + physics + GPU, per frame
  server.py      camera → pipeline → socket + lab window
  synthetic.py   generated frames (incl. a pinch-grab), for CI and protocol checks
  tools/         fetch_models, probe, probe_client, replay
tests/           synthetic-landmark tests, no camera needed
```

`signals.py` imports nothing but the standard library. That is the point: the
stretch math — the risky part of this project — is testable against fixtures
and replayable offline without a camera.

### A trap worth knowing about

This machine has **OBS Virtual Camera** installed, and it enumerates at a
*lower* DirectShow index than the real webcam. Opening it succeeds and
yields a frozen frame, because OBS isn't running. An index-based grabber
will happily record a motionless image forever and every channel will read
zero forever.

`capture.py` therefore selects the device **by name** (`device_name` in
`config.py`, matched case-insensitively as a substring) and falls back to
skipping anything whose name contains "virtual", "obs", "snap" and friends.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

189 tests, no camera required. They lock in the properties that matter:
scale invariance (multiply every landmark by 2 and nothing changes),
responsiveness (move the cheekbone and the ear channel goes up),
symmetry (a symmetric face gives exactly zero asymmetry), the calibration
motion gate, one release per pull, the grab gesture end to end (grab, pull,
head movement, hysteresis, dropouts, snap, both hands at once, fingers held
where pinched and never from the air beside them), and the stretch effect's
rules: only skin moves, eyes and teeth never do, the snap-back, and its
speed.
