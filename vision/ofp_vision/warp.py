"""The lab's stretch effect: skin pinched and pulled like rubber.

Cosmetic only: Unity owns the real avatar. This exists so that tuning the
grab in the lab looks and feels like the game will.

The model
---------
Skin is one elastic sheet, pinned a short way from the pinch. Pinch it and
pull, and:

- the pinched patch (the *core*) travels with the fingers;
- the skin around it is dragged along, less the further out it is, down to
  nothing at the *reach*, where it is pinned. The ring of skin at original
  radius rho moves by h(rho) of the pull, with h falling logarithmically
  from 1 at the core to 0 at the reach - the profile of a stretched
  membrane;
- the lifted skin rides *over* the skin ahead of it, so where several rings
  land on one pixel the most-lifted (largest h) is the one you see.

The result is a tent: wide where it leaves the face, narrowing into a strand
that ends between the fingers. Nothing outside the reach moves; the reach
always stops short of the eyes, so they never move; and only skin is drawn
(see `skin.py`) - the room never bends, hair and background are never
pulled, and the teeth stay where they are.

Inverting it
------------
For rendering we need, per output pixel, which skin lands there. A pixel at
offset q from the anchor lies on ring h when

    |q - h * pull| = rho(h)

rho(h) is exponential in h. Approximated as piecewise linear over a few
segments, the equation on each segment is a quadratic in h with a closed
form solution - so the whole inverse is a handful of vectorised numpy
operations, no mesh, no iteration. The largest valid root is the top layer.
The same geometry gives the tent's exact slope at every pixel (implicit
differentiation of the ring equation), which is what the lighting uses.

Making it look real
-------------------
- Texture: the stretched skin *is* the real skin, spread out.
- Light: the light direction is measured from the player's own face (which
  cheek is brighter, forehead vs chin), and the tent is lit with it from its
  exact slopes: the strand rolls into shade on its far side like a real
  tube, and taut skin gets a soft sheen.
- Shadow: the lifted tent casts a soft shadow on what is under it.
- Blanching: skin under tension goes paler, as real skin does.
- Grain: stretched pixels get camera-like grain, so they are no smoother
  than the rest of the image.
- Mouth: pulling a lip away opens the mouth onto its dark inside.
- Motion: on release the skin snaps back on a damped spring.
- Folds are antialiased; the pinching fingers stay on top of it all.

Coordinates are the raw, unmirrored camera image - the same space as the
landmarks. Warp first, then mirror the preview.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from .grab import NO_TARGET, GrabState, finger_owner, followed_grab
from .indices import (
    FACE_EYE_POINTS,
    FACE_INNER_LIP_LOWER,
    FACE_INNER_LIP_UPPER,
    FACE_OVAL,
    HAND_DIGITS,
    HAND_MIDDLE_MCP,
    HAND_WRIST,
)
from .light import default_light, followed_light
from .signals import Frame, HandObservation, Point
from .skin import (
    FEATHER_PX,
    Masks,
    Window,
    cavity_colour,
    digit_mask,
    face_masks,
    feathered,
    hand_mask,
    interocular,
    patch_around,
)

# --- shape ------------------------------------------------------------------

# The pinched core, as a fraction of interocular distance: how much skin the
# fingers hold. The nose takes its nostrils with it (a smaller core smears
# them into dark streaks); lips and mouth corners are pinched finely.
CORE_PER_FACE = 0.12
CORE_BY_TARGET = {"nose": 0.17, "lipTop": 0.08, "lipBottom": 0.08, "mouthL": 0.08, "mouthR": 0.08}
# How far around the pinch the skin is dragged: this much at rest, growing
# with the pull (a harder pull recruits more skin), up to a limit.
REACH_PER_FACE = 0.30
REACH_GROWTH = 0.20
MAX_REACH_PER_FACE = 0.55
# The reach stops this fraction of the way to the nearest eye point - and,
# for a pinch on one side of the mouth, to the nearest point of the other
# lip: the lips are joined only at the corners, so skin never moves across
# the mouth opening.
EYE_CLEARANCE = 0.85
FAR_LIP: dict[str, tuple[int, ...]] = {
    "lipBottom": FACE_INNER_LIP_UPPER,
    "chin": FACE_INNER_LIP_UPPER,
    "lipTop": FACE_INNER_LIP_LOWER,
    "nose": FACE_INNER_LIP_LOWER,
}
# On a finger: the core in palm lengths; the reach is the finger itself,
# tip to knuckle, so the whole finger stretches and its base stays put.
CORE_PER_PALM = 0.10
# The core is never more than this fraction of the reach.
MAX_CORE_PER_REACH = 0.6
# Pieces of the piecewise-linear ring profile. 6 is visually exact.
SEGMENTS = 6
# Skin converging into the strand: lateral sampling widens by up to this
# much mid-strand, so the strand shows gathered skin.
GATHER = 0.3

# --- light ------------------------------------------------------------------

# Steepness of the tent: height gained per unit of radius lost.
TENT_SLOPE = 0.9
# Steepest slope the lighting will use (a strand's silhouette is vertical).
MAX_SLOPE = 4.0
# Share of the light that is ambient; the rest is directional.
AMBIENT = 0.5
# Taut skin is shinier: a soft specular sheen, scaled by the stretch.
SPECULAR = 0.08
SHININESS = 16.0
# Shadow cast by the lifted skin: strength, how far it falls per unit of
# height, and how soft it is (blur, as a fraction of the tent's height).
SHADOW = 0.22
SHADOW_REACH = 0.35
SHADOW_SOFTNESS = 0.3
# A shadow reaches full strength once the skin is lifted this many pixels;
# a slight rise casts only a faint one.
SHADOW_FULL_LIFT = 20.0
# The shadow is computed at a resolution where its blur is about this wide.
SHADOW_BLUR_PX = 4.0
# Skin under tension goes paler: per unit of area stretch, up to a limit.
BLANCH_PER_STRETCH = 0.05
MAX_BLANCH = 0.25
# Camera-like grain on stretched pixels, relative to the image's own noise.
GRAIN = 1.0
# How fully an opened mouth shows its dark inside rather than what was there.
CAVITY = 0.9
# A jump in h larger than this between neighbouring pixels is a fold edge.
FOLD_JUMP = 0.08
# cv2.remap refuses outputs with 32767 or more rows or columns, so the moved
# pixels are sampled as a block this wide rather than as one tall column.
REMAP_COLUMNS = 512

# --- motion -----------------------------------------------------------------

FRAME_DT = 1.0 / 30.0
# The snap back after a release: a damped spring, integrated in substeps
# (one explicit step per frame is unstable at this stiffness and made the
# snap vanish in a single frame).
SPRING_HZ = 5.0
SPRING_DAMPING = 0.45
SPRING_SUBSTEPS = 8
# The rebound past rest is limited to this fraction of the reach.
MAX_REBOUND = 0.3
# Pulls shorter than this are invisible.
MIN_PULL_PX = 0.5

# --- level of detail ----------------------------------------------------------

# A stretch whose path covers more than this many pixels (a face close to
# the lens) is drawn at a lower resolution and composited back, changed
# pixels only. Cost then stays near a normal-distance face's, and so does
# the detail per face.
LOD_AREA = 45_000
# Never below a third of the camera's resolution.
MAX_LOD_SHRINK = 3


def _length_px(a: Point, b: Point, size: tuple[int, int]) -> float:
    return math.hypot((a[0] - b[0]) * size[0], (a[1] - b[1]) * size[1])


# --- the pinch --------------------------------------------------------------


@dataclass(frozen=True)
class Pinch:
    """One frame's pinch, in pixels of the raw camera image."""

    anchor: tuple[float, float]
    pull: tuple[float, float]
    core: float
    reach: float

    @property
    def length(self) -> float:
        return math.hypot(*self.pull)

    @property
    def peak(self) -> float:
        """Height of the pinched core above the face.

        Skin lifts as far as it is pulled, up to the tent's full height: a
        barely-pulled pinch (or the last frames of a snap back) is a gentle
        rise, not a bump.
        """
        return TENT_SLOPE * min(self.reach - self.core, self.length)

    def window(self, size: tuple[int, int]) -> Window:
        """The part of the frame the tent, its source skin and its shadow touch."""
        ax, ay = self.anchor
        tx, ty = ax + self.pull[0], ay + self.pull[1]
        shadow = SHADOW_REACH * self.peak + 2.0 * SHADOW_SOFTNESS * self.peak
        margin = int(self.reach * (1.0 + GATHER) + shadow) + FEATHER_PX + 2
        box = Window(int(min(ax, tx)), int(min(ay, ty)), int(max(ax, tx)) + 1, int(max(ay, ty)) + 1)
        return box.grown(margin, size)


def ring_radius(h, core: float, reach: float):
    """Original radius of the ring of skin that has moved by h of the pull."""
    return reach * (core / reach) ** h


def top_layer(qx: np.ndarray, qy: np.ndarray, pinch: Pinch) -> np.ndarray:
    """How far (as a fraction of the pull) the topmost skin at each pixel moved.

    `qx`, `qy` are pixel offsets from the anchor. 0 = unmoved skin (or no skin
    of the tent at all), 1 = the pinched core.
    """
    dx, dy = pinch.pull
    length2 = np.float32(dx * dx + dy * dy)
    qd = qx * np.float32(dx) + qy * np.float32(dy)
    qq = qx * qx + qy * qy
    best = np.zeros(qd.shape, dtype=np.float32)
    levels = np.linspace(0.0, 1.0, SEGMENTS + 1)
    radii = ring_radius(levels, pinch.core, pinch.reach)
    for h0, h1, r0, r1 in zip(levels[:-1], levels[1:], radii[:-1], radii[1:]):
        beta = (r1 - r0) / (h1 - h0)
        alpha = r0 - beta * h0
        # |q - h*pull|^2 = (alpha + beta*h)^2  ->  a2*h^2 - 2*b*h + c = 0
        a2 = np.float32(length2 - beta * beta)
        b = qd + np.float32(alpha * beta)
        c = qq - np.float32(alpha * alpha)
        for h in _quadratic_roots(a2, b, c):
            ok = (h >= h0 - 1e-4) & (h <= h1 + 1e-4) & (h > best)
            best = np.where(ok, np.minimum(h, np.float32(1.0)), best)
    in_core = (qx - np.float32(dx)) ** 2 + (qy - np.float32(dy)) ** 2 <= pinch.core**2
    return np.where(in_core, np.float32(1.0), best)


def _quadratic_roots(a2: np.float32, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, ...]:
    """Real roots of a2*h^2 - 2*b*h + c = 0; NaN where there are none."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(a2) < 1e-6:
            return (c / (2.0 * b),)
        disc = b * b - a2 * c
        root = np.sqrt(np.where(disc >= 0, disc, np.nan))
        return (b + root) / a2, (b - root) / a2


def source_offsets(qx: np.ndarray, qy: np.ndarray, h: np.ndarray, pinch: Pinch):
    """Where each pixel's skin came from, as offsets from the anchor."""
    dx, dy = pinch.pull
    yx = qx - h * np.float32(dx)
    yy = qy - h * np.float32(dy)
    length = pinch.length
    if length < 1e-6:
        return yx, yy
    ux, uy = np.float32(dx / length), np.float32(dy / length)
    along = yx * ux + yy * uy
    # Wider sampling across the strand mid-way: skin gathered into it.
    across = (yy * ux - yx * uy) * (1.0 + GATHER * 4.0 * h * (1.0 - h))
    return along * ux - across * uy, along * uy + across * ux


def tent_height(h: np.ndarray, pinch: Pinch) -> np.ndarray:
    """How high the skin at each pixel has been lifted off the face.

    A smoothstep in h: the skin leaves the face tangentially (no crease
    where the tent starts) and is flat on top where the fingers hold it.
    """
    return np.float32(pinch.peak) * h * h * (3.0 - 2.0 * h)


def tent_surface(qx: np.ndarray, qy: np.ndarray, h: np.ndarray, pinch: Pinch):
    """(gx, gy, stretch): the tent's exact slope, and how stretched its skin is.

    From the ring equation phi = |q - h*pull| - rho(h) = 0, the gradient of h
    is e / (e.pull + rho'(h)), e being the unit vector from the ring's centre
    to the pixel. The height's gradient is that times dHeight/dh. It
    steepens without limit where the strand's side turns away from the
    camera - that is its silhouette - so it is capped at MAX_SLOPE.

    The source map is q - h*pull, so its Jacobian determinant is
    rho' / (e.pull + rho'): the area of skin each pixel shows. Its inverse
    is the stretch (below 1 on the leading side, where skin bunches up; the
    gather across the strand compresses it a little more).
    """
    dx, dy = pinch.pull
    yx, yy = qx - h * np.float32(dx), qy - h * np.float32(dy)
    distance = np.sqrt(yx * yx + yy * yy) + np.float32(1e-6)
    ex, ey = yx / distance, yy / distance
    rate = ring_radius(h, pinch.core, pinch.reach) * np.float32(math.log(pinch.core / pinch.reach))
    toward = np.minimum(ex * np.float32(dx) + ey * np.float32(dy) + rate, np.float32(-1e-3))
    rise = np.float32(pinch.peak * 6.0) * h * (1.0 - h)
    slope = np.minimum(-rise / toward, np.float32(MAX_SLOPE))
    gather = 1.0 + np.float32(GATHER * 4.0) * h * (1.0 - h)
    stretch = np.where(h < 1.0, toward / np.minimum(rate, np.float32(-1e-3)) / gather, np.float32(1.0))
    return -slope * ex, -slope * ey, np.clip(stretch, np.float32(1.0), np.float32(6.0))


def pinch_sizes(target: str, frame: Frame, size: tuple[int, int], anchor: np.ndarray, pull_len: float):
    """(core, reach) in pixels for this target, or None if it is out of view."""
    owner = finger_owner(target)
    if owner is not None:
        return _finger_sizes(target, owner, frame, size)
    face = frame.face
    if face is None or len(face) <= max(FACE_OVAL):
        return None
    eyes_apart = interocular(face, size)
    core = CORE_BY_TARGET.get(target, CORE_PER_FACE) * eyes_apart
    reach = min(REACH_PER_FACE * eyes_apart + REACH_GROWTH * pull_len, MAX_REACH_PER_FACE * eyes_apart)
    pinned = FACE_EYE_POINTS + FAR_LIP.get(target, ())
    nearest = min(
        math.hypot(face[i][0] * size[0] - anchor[0], face[i][1] * size[1] - anchor[1]) for i in pinned
    )
    return _fit(core, min(reach, EYE_CLEARANCE * nearest))


def _finger_sizes(target: str, owner: str, frame: Frame, size: tuple[int, int]):
    hand = next((h for h in frame.hands if h.handedness == owner), None)
    if hand is None:
        return None
    lm = hand.landmarks
    digit = HAND_DIGITS[target.split(".")[2]]
    palm = _length_px(lm[HAND_WRIST], lm[HAND_MIDDLE_MCP], size)
    finger = sum(_length_px(lm[a], lm[b], size) for a, b in zip(digit, digit[1:]))
    return _fit(CORE_PER_PALM * palm, finger)


def _fit(core: float, reach: float) -> tuple[float, float]:
    reach = max(reach, 3.0)
    return max(1.0, min(core, MAX_CORE_PER_REACH * reach)), reach


# --- light ------------------------------------------------------------------


def shade(gx: np.ndarray, gy: np.ndarray, light: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(diffuse ratio, specular) for a surface with height gradient (gx, gy).

    Relative to flat skin, which keeps exactly its own colour: (1, 0).
    """
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    lx, ly, lz = (np.float32(v) for v in light)
    n_dot_l = (-gx * lx - gy * ly + lz) / norm
    diffuse = (AMBIENT + (1.0 - AMBIENT) * np.maximum(n_dot_l, 0.0)) / (AMBIENT + (1.0 - AMBIENT) * lz)
    half = np.array([lx, ly, lz + 1.0])
    hx, hy, hz = (np.float32(v) for v in half / np.linalg.norm(half))
    n_dot_h = np.maximum((-gx * hx - gy * hy + hz) / norm, 0.0)
    specular = np.maximum(n_dot_h**SHININESS - hz**SHININESS, 0.0)
    return diffuse, specular


def cast_shadow(lift: np.ndarray, light: np.ndarray) -> np.ndarray:
    """How much the lifted skin darkens each pixel beneath it, 0..SHADOW.

    A soft shadow has no fine detail, so it is worked out at a resolution
    where its blur is a few pixels wide, then scaled back up: a close face's
    big tent would otherwise need a very wide (slow) blur.
    """
    peak = float(lift.max()) if lift.size else 0.0
    if peak < 1.0:
        return np.zeros_like(lift)
    blur = max(1.0, SHADOW_SOFTNESS * peak)
    shrink = max(1, int(blur / SHADOW_BLUR_PX))
    small = lift if shrink == 1 else cv2.resize(
        lift, (max(1, lift.shape[1] // shrink), max(1, lift.shape[0] // shrink)), interpolation=cv2.INTER_AREA
    )
    lz = max(float(light[2]), 0.35)
    shift = (-light[0] / lz * SHADOW_REACH * peak / shrink, -light[1] / lz * SHADOW_REACH * peak / shrink)
    moved = cv2.warpAffine(
        small, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (small.shape[1], small.shape[0])
    )
    soft = cv2.GaussianBlur(moved, (0, 0), blur / shrink)
    if shrink > 1:
        soft = cv2.resize(soft, (lift.shape[1], lift.shape[0]), interpolation=cv2.INTER_LINEAR)
    strength = np.float32(SHADOW * min(1.0, peak / SHADOW_FULL_LIFT))
    return strength * np.clip((soft - lift) / np.float32(0.5 * peak), 0.0, 1.0)


def grain_level(bgr: np.ndarray, centre: tuple[float, float], radius: float) -> float:
    """The camera's noise level around a point (robust std of the fine detail)."""
    patch = patch_around(bgr, int(centre[0]), int(centre[1]), int(max(8, radius)))
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    return float(1.4826 * np.median(np.abs(fine)))


# --- rendering --------------------------------------------------------------


@dataclass
class Look:
    """Per-frame inputs to the look that do not come from the geometry."""

    light: np.ndarray
    grain_sigma: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


@dataclass(frozen=True)
class Geometry:
    """The skin that moved, as flat arrays over just those pixels.

    `rows`, `cols` are window-local; `map_x`, `map_y` where each pixel's skin
    came from (frame coordinates); `h` how far it moved; `gx`, `gy` the
    tent's slope there; `stretch` how stretched it is.
    """

    rows: np.ndarray
    cols: np.ndarray
    h: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    stretch: np.ndarray

    def where(self, keep: np.ndarray) -> "Geometry":
        return Geometry(*(getattr(self, name)[keep] for name in self.__dataclass_fields__))


def _reachable(pinch: Pinch, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Window-local pixels the tent can cover: within the reach of the pull's path.

    Every ring stays inside that capsule, so nothing outside it can move.
    """
    mask = np.zeros(window.shape, dtype=np.uint8)
    ax, ay = pinch.anchor[0] - window.x0, pinch.anchor[1] - window.y0
    start, end = (int(ax), int(ay)), (int(ax + pinch.pull[0]), int(ay + pinch.pull[1]))
    radius = int(math.ceil(pinch.reach)) + 1
    cv2.line(mask, start, end, 255, 2 * radius + 1)
    cv2.circle(mask, start, radius, 255, -1)
    cv2.circle(mask, end, radius, 255, -1)
    return np.nonzero(mask)


def solve(pinch: Pinch, window: Window) -> Geometry:
    rows, cols = _reachable(pinch, window)
    qx = (cols + window.x0).astype(np.float32) - np.float32(pinch.anchor[0])
    qy = (rows + window.y0).astype(np.float32) - np.float32(pinch.anchor[1])
    h = top_layer(qx, qy, pinch)
    moved = h > 0.0
    rows, cols, qx, qy, h = rows[moved], cols[moved], qx[moved], qy[moved], h[moved]
    sx, sy = source_offsets(qx, qy, h, pinch)
    gx, gy, stretch = tent_surface(qx, qy, h, pinch)
    return Geometry(
        rows, cols, h, sx + np.float32(pinch.anchor[0]), sy + np.float32(pinch.anchor[1]), gx, gy, stretch
    )


def _read(image: np.ndarray, geometry: Geometry, window: Window | None, cubic: bool = False) -> np.ndarray:
    """`image` sampled where each moved pixel's skin came from.

    With a window, `image` is window-local. The N sample points go to remap
    as a block REMAP_COLUMNS wide (padded), not as one N-row column: remap
    refuses outputs 32767 or more rows tall, and a close face moves more
    pixels than that.
    """
    count = geometry.map_x.shape[0]
    offset_x = np.float32(window.x0) if window is not None else np.float32(0.0)
    offset_y = np.float32(window.y0) if window is not None else np.float32(0.0)
    rows = -(-count // REMAP_COLUMNS)
    pad = rows * REMAP_COLUMNS - count
    map_x = np.pad(geometry.map_x - offset_x, (0, pad)).reshape(rows, REMAP_COLUMNS)
    map_y = np.pad(geometry.map_y - offset_y, (0, pad)).reshape(rows, REMAP_COLUMNS)
    sampled = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE if window is None else cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(rows * REMAP_COLUMNS, *sampled.shape[2:])[:count]


def render(bgr: np.ndarray, pinch: Pinch, masks: Masks, look: Look) -> np.ndarray:
    """A copy of `bgr` with the pinch drawn in."""
    window = masks.window
    original = window.crop(bgr)
    out = original.copy()
    geometry = solve(pinch, window)
    if geometry.h.size == 0:
        return bgr

    # Only skin moves: a pixel shows moved content only if what it samples
    # is skin. (Where skin pulled inward leaves a gap, the gap keeps its own
    # pixel - never the room dragged in after it.) The one exception is the
    # mouth: where a lip pulled away leaves skin uncovered, the open mouth's
    # dark inside shows.
    skin_from = _read(masks.skin, geometry, window)
    alpha = skin_from
    mouth_from = None
    if masks.mouth is not None:
        mouth_from = _read(masks.mouth, geometry, window)
        here = masks.skin[geometry.rows, geometry.cols] * (1.0 - masks.mouth[geometry.rows, geometry.cols])
        alpha = np.maximum(alpha, mouth_from * here)
    keep = alpha > (1.0 / 255.0)
    if not keep.any():
        return bgr
    geometry, alpha, skin_from = geometry.where(keep), alpha[keep], skin_from[keep]
    mouth_from = mouth_from[keep] if mouth_from is not None else None

    diffuse, specular = shade(geometry.gx, geometry.gy, look.light)
    lifted = _lit(_skin_colours(bgr, geometry, masks, skin_from), geometry, diffuse, specular, look)
    if mouth_from is not None:
        # An opened mouth shows its shadowed inside, not the teeth smeared open.
        cavity = np.float32(masks.cavity) * diffuse[:, np.newaxis]
        lifted += (cavity - lifted) * (np.float32(CAVITY) * mouth_from)[:, np.newaxis]
    before = original[geometry.rows, geometry.cols].astype(np.float32)
    after = before + (lifted - before) * alpha[:, np.newaxis]
    out[geometry.rows, geometry.cols] = np.clip(after + 0.5, 0, 255).astype(np.uint8)

    _darken_under(out, geometry, pinch, alpha, look.light)
    _keep_on_top(out, original, masks.front)
    out = _antialias_folds(out, geometry, window)
    result = bgr.copy()
    result[window.y0 : window.y1, window.x0 : window.x1] = out
    return result


def _darken_under(out: np.ndarray, geometry: Geometry, pinch: Pinch, alpha: np.ndarray, light) -> None:
    """The shadow the lifted skin casts on what is beneath it."""
    lift = np.zeros(out.shape[:2], dtype=np.float32)
    lift[geometry.rows, geometry.cols] = tent_height(geometry.h, pinch) * (alpha > 0.5)
    shadow = cast_shadow(lift, light)
    rows, cols = np.nonzero(shadow > 1e-3)
    darker = out[rows, cols].astype(np.float32) * (1.0 - shadow[rows, cols])[:, np.newaxis]
    out[rows, cols] = np.clip(darker + 0.5, 0, 255).astype(np.uint8)


def _keep_on_top(out: np.ndarray, original: np.ndarray, front: np.ndarray | None) -> None:
    """Put the hands that are not being stretched back over everything."""
    if front is None:
        return
    rows, cols = np.nonzero(front > 1e-3)
    weight = front[rows, cols][:, np.newaxis]
    mixed = out[rows, cols] * (1.0 - weight) + original[rows, cols] * weight
    out[rows, cols] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def _skin_colours(bgr: np.ndarray, geometry: Geometry, masks: Masks, skin_from: np.ndarray) -> np.ndarray:
    """The colour of the skin each moved pixel shows. (N, 3) float.

    Sharp (bicubic) inside the skin. At its soft edge the colour is
    premultiplied: the skin-only image (colour times the unsoftened mask) is
    softened with the very kernel that softened the coverage, then divided
    by that coverage. Every pixel the soft edge covers - including the part
    of the feather that lies over the room - gets the colour of the nearest
    skin, never the room's. (A target on the face's outline is half room.)
    """
    colours = _read(bgr, geometry, None, cubic=True).astype(np.float32)
    edge = (skin_from > 1e-3) & (skin_from < 0.98)
    if edge.any():
        at_edge = geometry.where(edge)
        # Only the band the edge samples from needs softening.
        margin = FEATHER_PX + 2
        window = masks.window
        band = Window(
            int(at_edge.map_x.min()) - margin, int(at_edge.map_y.min()) - margin,
            int(at_edge.map_x.max()) + margin + 1, int(at_edge.map_y.max()) + margin + 1,
        )
        band = Window(max(band.x0, window.x0), max(band.y0, window.y0), min(band.x1, window.x1), min(band.y1, window.y1))
        local = Window(band.x0 - window.x0, band.y0 - window.y0, band.x1 - window.x0, band.y1 - window.y0)
        skin_only = band.crop(bgr).astype(np.float32) * local.crop(masks.skin_solid)[..., np.newaxis]
        softened = cv2.GaussianBlur(skin_only, (FEATHER_PX, FEATHER_PX), 0)
        colours[edge] = _read(softened, at_edge, band) / skin_from[edge][:, np.newaxis]
    return colours


def _lit(colours: np.ndarray, geometry: Geometry, diffuse, specular, look: Look) -> np.ndarray:
    """The moved skin, lit: shading, sheen, blanching and grain. (N, 3) float."""
    stretch = geometry.stretch
    taut = np.clip((stretch - 1.0) * 0.5, 0.0, 1.0)
    lifted = colours * diffuse[:, np.newaxis]
    lifted += (np.float32(255.0 * SPECULAR) * specular * taut)[:, np.newaxis]

    # Blanching: towards a paler, less saturated version of the same skin.
    amount = np.clip((stretch - 1.0) * np.float32(BLANCH_PER_STRETCH), 0.0, MAX_BLANCH)[:, np.newaxis]
    luma = lifted @ np.float32([0.114, 0.587, 0.299])
    pale = 0.75 * lifted + 0.25 * luma[:, np.newaxis] + 22.0
    lifted += (pale - lifted) * amount

    if look.grain_sigma > 0:
        grain = look.rng.standard_normal(stretch.shape, dtype=np.float32)
        grain *= np.float32(GRAIN * look.grain_sigma) * np.clip(np.sqrt(stretch) - 1.0, 0.0, 1.0)
        lifted += grain[:, np.newaxis]
    return lifted


def _antialias_folds(out: np.ndarray, geometry: Geometry, window: Window) -> np.ndarray:
    """Soften the hard edge where lifted skin folds over the skin beneath."""
    h = np.zeros(window.shape, dtype=np.float32)
    h[geometry.rows, geometry.cols] = geometry.h
    edge = np.zeros(h.shape, dtype=np.uint8)
    edge[:, 1:] |= (np.abs(np.diff(h, axis=1)) > FOLD_JUMP).astype(np.uint8)
    edge[1:, :] |= (np.abs(np.diff(h, axis=0)) > FOLD_JUMP).astype(np.uint8)
    if not edge.any():
        return out
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8)).astype(bool)
    soft = cv2.blur(out, (3, 3))
    out[edge] = soft[edge]
    return out


# --- motion -----------------------------------------------------------------


@dataclass
class Spring:
    """The pull, held by the fingers or snapping back on its own."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    released_along: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def hold(self, pull: np.ndarray) -> None:
        """The fingers have it: follow them exactly, remembering the speed."""
        self.velocity = (pull - self.position) / FRAME_DT
        self.position = pull
        length = float(np.hypot(*pull))
        self.released_along = pull / length if length > 1e-6 else np.zeros(2)

    def let_go(self, reach: float) -> None:
        """One frame of the snap back: a damped spring towards rest."""
        omega = 2.0 * math.pi * SPRING_HZ
        step = FRAME_DT / SPRING_SUBSTEPS
        limit = MAX_REBOUND * reach
        for _ in range(SPRING_SUBSTEPS):
            accel = -omega * omega * self.position - 2.0 * SPRING_DAMPING * omega * self.velocity
            self.velocity = self.velocity + accel * step
            self.position = self.position + self.velocity * step
            # Past rest the skin is pushing into the face: keep that rebound
            # small, and stop it pushing further.
            rebound = -float(self.position @ self.released_along)
            if rebound > limit:
                self.position = self.position + self.released_along * (rebound - limit)
                inward = float(self.velocity @ self.released_along)
                if inward < 0.0:
                    self.velocity = self.velocity - self.released_along * inward
        if np.hypot(*self.position) < MIN_PULL_PX and np.hypot(*self.velocity) < 30.0:
            self.position = np.zeros(2)
            self.velocity = np.zeros(2)


def _composite(full: np.ndarray, before: np.ndarray, after: np.ndarray, shrink: int) -> np.ndarray:
    """Bring a stretch drawn at 1/shrink resolution back into the full frame.

    Only the pixels the stretch changed are replaced (with a soft edge), so
    the rest of the picture keeps its full sharpness.
    """
    changed = (np.abs(after.astype(np.int16) - before).max(axis=2) > 0).astype(np.uint8)
    rows, cols = np.nonzero(changed)
    if len(rows) == 0:
        return full
    y0, y1 = max(0, rows.min() - 2), min(changed.shape[0], rows.max() + 3)
    x0, x1 = max(0, cols.min() - 2), min(changed.shape[1], cols.max() + 3)
    region = (shrink * (x1 - x0), shrink * (y1 - y0))
    upscaled = cv2.resize(after[y0:y1, x0:x1], region, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    soft = cv2.dilate(changed[y0:y1, x0:x1] * 255, np.ones((3, 3), np.uint8))
    weight = cv2.resize(soft, region, interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., np.newaxis] / 255.0
    result = full.copy()
    target = result[shrink * y0 : shrink * y1, shrink * x0 : shrink * x1]
    target[:] = np.clip(target * (1.0 - weight) + upscaled * weight + 0.5, 0, 255).astype(np.uint8)
    return result


# --- the warper ---------------------------------------------------------------


class FaceWarper:
    """Turns grab states into the drawn stretch, frame by frame."""

    def __init__(self, seed: int | None = None) -> None:
        self._anchor = np.zeros(2)
        self._spring = Spring()
        self._sizes = (1.0, 3.0)
        # What the last grab held. Kept through the release, so the part
        # that springs back is the part that was let go.
        self._target = NO_TARGET
        # The latest landmarks seen. Kept through a tracking dropout, so the
        # effect does not flash off for a frame.
        self._face: list[Point] | None = None
        self._grabbed_hand: HandObservation | None = None
        self._other_hands: list[HandObservation] = []
        self._light = default_light()
        self._rng = np.random.default_rng(seed)
        # The hand whose grab is drawn: this effect draws one at a time.
        self._hand: str | None = None

    @property
    def pull_px(self) -> np.ndarray:
        return self._spring.position.copy()

    def update(self, grabs: Mapping[str, GrabState], frame: Frame, size: tuple[int, int]) -> None:
        """Take this frame's grabs, one per hand. `size` is (width, height)."""
        self._hand, grab = followed_grab(grabs, self._hand)
        if grab.target != NO_TARGET:
            self._target = grab.target
        if frame.face is not None and len(frame.face) > max(FACE_OVAL):
            self._face = frame.face
        owner = finger_owner(self._target)
        grabbed = next((h for h in frame.hands if h.handedness == owner), None)
        if grabbed is not None:
            self._grabbed_hand = grabbed
        self._other_hands = [h for h in frame.hands if h.handedness != owner]

        if grab.active and grab.anchor is not None:
            self._anchor = np.array([grab.anchor[0] * size[0], grab.anchor[1] * size[1]])
            self._spring.hold(np.array([grab.pull[0] * size[0], grab.pull[1] * size[1]]))
        else:
            self._spring.let_go(self._sizes[1])

        sizes = pinch_sizes(self._target, frame, size, self._anchor, float(np.hypot(*self._spring.position)))
        if sizes is not None:
            self._sizes = sizes

    def _masks(self, bgr: np.ndarray, window: Window) -> Masks | None:
        size = (bgr.shape[1], bgr.shape[0])
        front = feathered(hand_mask(self._other_hands, size, window)) if self._other_hands else None
        if finger_owner(self._target) is not None:
            if self._grabbed_hand is None:
                return None
            skin = digit_mask(self._grabbed_hand, self._target.split(".")[2], size, window)
            return Masks.of(window, skin, front=front)
        if self._face is None:
            return None
        skin, mouth = face_masks(bgr, self._face, window)
        cavity = cavity_colour(bgr, mouth, window)
        return Masks.of(window, skin, mouth=feathered(mouth), front=front, cavity=cavity)

    def warp(self, bgr: np.ndarray) -> np.ndarray:
        """Return a copy of `bgr` with the stretch drawn (or `bgr` if none)."""
        pull = self._spring.position
        if np.hypot(*pull) < MIN_PULL_PX:
            self._light = followed_light(self._light, bgr, self._face)
            return bgr
        core, reach = self._sizes
        path = math.pi * reach * reach + 2.0 * reach * float(np.hypot(*pull))
        shrink = min(MAX_LOD_SHRINK, max(1, math.ceil(math.sqrt(path / LOD_AREA))))
        if shrink == 1:
            return self._draw(bgr, 1.0)
        small = cv2.resize(bgr, (bgr.shape[1] // shrink, bgr.shape[0] // shrink), interpolation=cv2.INTER_AREA)
        return _composite(bgr, small, self._draw(small, 1.0 / shrink), shrink)

    def _draw(self, bgr: np.ndarray, scale: float) -> np.ndarray:
        """The stretch drawn into `bgr`, an image `scale` times the camera's."""
        size = (bgr.shape[1], bgr.shape[0])
        core, reach = self._sizes
        pull = self._spring.position * scale
        anchor = self._anchor * scale
        pinch = Pinch((float(anchor[0]), float(anchor[1])), (float(pull[0]), float(pull[1])), core * scale, reach * scale)
        window = pinch.window(size)
        if window.empty:
            return bgr
        masks = self._masks(bgr, window)
        if masks is None:
            return bgr
        look = Look(self._light, grain_level(bgr, pinch.anchor, pinch.reach), self._rng)
        return render(bgr, pinch, masks, look)
