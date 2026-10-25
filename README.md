# One Piece — Devil Fruit

Camera-driven elastic body. The player uses a webcam to stretch their own
cheek, ear, finger or hand, and the avatar stretches with them.

```
   webcam ──► vision service ──► websocket ──► Unity ──► avatar
              (python, 30Hz)      (~60 numbers)   (C#)
```

The webcam feed never leaves the player's machine. Only derived numbers do.

---

## Where things are

| Path | What |
|---|---|
| [`vision/`](vision/) | The perception service and the Signal Lab. See its [README](vision/README.md). |
| `vision/protocol/Unity/` | Reference Unity client for the state stream. |
| `Unity project` | *Not created yet.* See the roadmap below. |

---

## Status

**Done — the perception pipeline is built and verified end to end in synthetic mode.**

- MediaPipe face + hand + pose tracking, CPU, ~30-40ms/frame
- **Pinch-grab-pull**, the main interaction: pinch an ear, the nose, the
  jaw, a cheek, the mouth or any finger of the other hand, pull, let go to
  fire - with one hand, or both at once. Only skin stretches in the lab's video; the room never bends. Stretch measured in interocular distances, anchor
  follows the head
- Scale-invariant stretch features (the maths that survives a 2D webcam)
- Calibration with motion gating; channels missing from the calibration
  learn their own baseline
- Activation and release detection, per channel, one release per pull
- Skin physics in the lab, drawn on the graphics card: pinched skin pulls
  out with weight, drag and a rubber snap-back; eyes, teeth, hair and the
  room never move (~11-22ms/frame at 720p). CPU fallback without a GPU
- Websocket protocol v3 + Unity reference client (not yet compiled — no
  Unity project exists)
- 249 tests, no camera needed
- Synthetic mode, including a synthetic pinch-grab, to verify the whole
  chain without a camera

**Not done — needs a real face in front of the lens.**

No real recording with a face in it exists yet. Record yourself grabbing
and pulling, replay it, and tune `GrabConfig` from the numbers. That is the
next session's job.

---

## Roadmap

1. **Signal Lab tuning.** Sit in front of the camera, pinch-grab and pull
   each target, record it, replay it, tune `GrabConfig`. Then check which
   face-deformation channels survive contact with reality and cut the ones
   that don't. *This is the current step.*
2. **Unity project.** Scaffold it, add the vision client, drive a test
   avatar.
3. **Stylized stretchy rig.** Limb stretch via bone length, cheek/face
   stretch via non-uniform head scale. Deliberately cartoon.
4. **Gameplay.** Pull to charge, release to fire. Gum-Gum Pistol from a
   finger, Bazooka from an arm.
5. **Packaging.** Ship the vision service as a companion app so players
   don't need Python.

---

## Decisions worth remembering

**Read the hand, not the face.** A pinch is something MediaPipe sees
reliably; a cheek deforming under a tug, through a 2D webcam, is not. So
the primary gesture is pinch-grab-pull, and the face-deformation channels
are secondary until real recordings prove them.

**The hard part is the camera, not the fruit.** Stretching a rig is a solved
problem. Knowing *that the player pulled their cheek, and by how much, from a
2D webcam* is not. Hence the Signal Lab first, in isolation, before any
avatar work.

**Ratios, not distances.** A pull is mostly a Z movement that a 2D webcam
cannot see. Every measurement is divided by a rigid, body-local reference
length so that leaning in, leaning out, and turning your head all cancel.
Details in the [vision README](vision/README.md).

**The vision service emits dimensionless numbers.** It reports "how much more
than your neutral pose". Unity decides what that is worth in centimetres.
Retuning the *feel* of the stretch never requires touching the perception,
and vice versa.

**Ear is the weak channel.** MediaPipe has no ear landmarks. The ear proxy
uses the cheekbone silhouette point where the ear attaches. It is labelled as
the weakest signal in the set and will be cut if it does not hold up.

**Calibration rejects movement.** A player still leaning in and settling
would otherwise poison the baseline for the whole session.

---

## Environment notes (this machine)

- The webcam was blocked by a Windows privacy setting
  (`ConsentStore\webcam\NonPackaged = Deny`), which makes every desktop app
  get `E_ACCESSDENIED`. It needs **Settings → Privacy & security → Camera →
  "Let desktop apps access your camera"** turned on.
- **OBS Virtual Camera** is installed and enumerates *before* the real
  webcam. Selecting cameras by index silently captures a frozen frame. The
  vision service selects by name for this reason.
- MediaPipe's GPU delegate is broken in the Windows wheels. CPU it is —
  ~27ms/frame is fine.
