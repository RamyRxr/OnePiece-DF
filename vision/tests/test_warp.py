"""Tests for the lab's stretch effect: skin pinched and pulled like rubber.

The rules the effect has to keep, whatever it looks like:

- only skin moves: the room, the teeth, hair never do
- nothing far from the pinch moves; the eyes never do
- the lifted skin rides over the skin ahead of it; the tent narrows from a
  wide base to the pinch
- the pinched skin is what the fingers hold at the tip
- the pinching fingers stay on top
- an opened mouth shows a dark inside, not stretched teeth
- a release snaps back past rest a little, then settles
- it fits in a live frame at 720p

"Not moved" allows for shading (the tent's shadow darkens what is under it)
but not for different content: a moved pixel of the fixture's gradient
background changes its colour, a shaded one only gets darker.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from face_fixture import EYE_L, H, MOUTH, MOUTH_HALF, RADIUS, TEETH, W, image, landmarks, norm, px

from ofp_vision.grab import GrabState, face_target_points
from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import Frame, HandObservation
from ofp_vision.light import estimate_light
from ofp_vision.warp import MAX_REBOUND, FaceWarper, Pinch, tent_surface, top_layer

FACE = landmarks()
TARGETS = face_target_points(FACE)


def fastest(action, runs: int = 7) -> float:
    """Seconds the quickest of `runs` calls took. The fastest run is what
    the code costs; slower ones measure whatever else the machine was doing."""
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def pulling(target: str, dx: float, dy: float, hand: str = "Right") -> GrabState:
    """A grab on `target`, pulled `dx`, `dy` frame heights."""
    return GrabState(
        active=True, hand=hand, target=target, anchor=TARGETS[target][:2] if target in TARGETS else None,
        pull=(dx * H / W, dy),
    )


def warp(grab: GrabState, frame_image=None, hands=(), face=FACE, frames: int = 2):
    warper = FaceWarper(seed=1)
    frame = Frame(t_ms=0.0, face=face, hands=list(hands))
    img = image() if frame_image is None else frame_image
    out = img
    for _ in range(frames):
        warper.update({"Right": grab}, frame, (W, H))
        out = warper.warp(img)
    return img, out


def assert_not_moved(out: np.ndarray, img: np.ndarray, point, radius: int = 4) -> None:
    x, y = px(point)
    before = img[y - radius : y + radius, x - radius : x + radius].astype(float) + 1.0
    after = out[y - radius : y + radius, x - radius : x + radius].astype(float) + 1.0
    ratio = after / before
    assert np.all(ratio <= 1.02), "never brighter"
    assert np.all(ratio.max(axis=2) - ratio.min(axis=2) < 0.06), "same content, at most shaded"


# --- what moves and what does not --------------------------------------------


def test_no_grab_leaves_the_image_untouched():
    img, out = warp(GrabState())
    assert out is img


def test_the_room_never_moves_even_right_beside_the_face():
    img, out = warp(pulling("cheekL", 0.3, 0.0))
    # Just above and below the face's edge, level with the pull.
    for dy in (-RADIUS - 0.03, RADIUS + 0.03):
        assert_not_moved(out, img, norm(0.12, dy))


def test_nothing_far_from_the_pinch_moves():
    img, out = warp(pulling("cheekL", 0.3, 0.0))
    for point in (norm(-0.1, 0.04), norm(0.0, -0.12), norm(-0.12, 0.12)):
        assert_not_moved(out, img, point)


def test_the_eyes_never_move_even_pinched_right_beside_them():
    pinch_point = norm(EYE_L[0], EYE_L[1] + 0.05)
    grab = GrabState(active=True, hand="Right", target="cheekL", anchor=pinch_point[:2], pull=(0.0, 0.25))
    img, out = warp(grab)
    assert_not_moved(out, img, norm(*EYE_L), radius=6)


def test_the_pinched_skin_is_at_the_fingertips():
    img, out = warp(pulling("cheekL", 0.35, 0.0))
    ax, ay = px(TARGETS["cheekL"])
    tx, ty = px((TARGETS["cheekL"][0] + 0.35 * H / W, TARGETS["cheekL"][1]))
    assert np.abs(out[ty, tx].astype(int) - img[ay, ax].astype(int)).max() <= 8


def test_skin_is_pulled_out_over_the_room():
    """Beyond the face's edge, along the pull, the background is covered by skin."""
    img, out = warp(pulling("cheekL", 0.35, 0.0))
    x, y = px(norm(RADIUS + 0.08, 0.04))
    b, g, r = out[y, x].astype(int)
    assert r > 150 and r > b, "skin-coloured (red-dominant), not the blue-green room"


def test_the_input_image_is_not_modified():
    img = image()
    before = img.copy()
    warp(pulling("cheekL", 0.3, 0.0), frame_image=img)
    assert np.array_equal(img, before)


# --- the mouth -----------------------------------------------------------------


def test_the_teeth_do_not_stretch():
    img, out = warp(pulling("lipBottom", 0.0, 0.15))
    upper_teeth = norm(MOUTH[0], MOUTH[1] - MOUTH_HALF[1] * 0.5)
    assert_not_moved(out, img, upper_teeth, radius=2)


def test_an_opened_mouth_shows_a_dark_inside_not_teeth():
    img, out = warp(pulling("lipBottom", 0.0, 0.15))
    # Where the lower lip was, the mouth is now open.
    x, y = px(norm(0.0, 0.122))
    luma = out[y, x] @ np.array([0.114, 0.587, 0.299])
    assert luma < 0.6 * (np.array(TEETH) @ np.array([0.114, 0.587, 0.299]))


# --- the shape -----------------------------------------------------------------


def straight_pinch(length: float = 300.0) -> Pinch:
    return Pinch(anchor=(0.0, 0.0), pull=(length, 0.0), core=12.0, reach=60.0)


def test_the_lifted_skin_rides_over_the_skin_ahead():
    """Past the original patch, two layers cover a pixel: the skin carried
    ahead of the pinch (moved by ~0.18 of the pull) and, over it, the skin
    stretched from behind (~0.43). The one on top is what shows."""
    h = top_layer(np.float32([100.0]), np.float32([0.0]), straight_pinch())
    assert h[0] == pytest.approx(0.435, abs=0.02)


def test_the_tent_narrows_from_its_base_to_the_pinch():
    pinch = straight_pinch()
    ys = np.arange(-80, 81, dtype=np.float32)

    def width_at(x: float) -> int:
        h = top_layer(np.full_like(ys, x), ys, pinch)
        return int((h > 0).sum())

    assert width_at(40.0) > width_at(150.0) > width_at(260.0) > 0


def test_nothing_outside_the_reach_of_the_pull_moves():
    h = top_layer(np.float32([0.0, 150.0]), np.float32([70.0, 75.0]), straight_pinch())
    assert np.all(h == 0.0)


def test_skin_behind_the_pinch_is_stretched_and_the_core_is_not():
    pinch = straight_pinch()
    qx, qy = np.float32([-30.0, 300.0]), np.float32([0.0, 0.0])
    h = top_layer(qx, qy, pinch)
    _, _, stretch = tent_surface(qx, qy, h, pinch)
    assert stretch[0] > 1.5
    assert stretch[1] == pytest.approx(1.0)


# --- hands ---------------------------------------------------------------------


def hand(at: tuple[float, float], side: str = "Right", scale: float = 1.0) -> HandObservation:
    """A hand pointing down, thumb and index tips at `at` (normalised).

    At scale 1 the palm is ~65px; at 2.4, ~156px (a hand close to the lens).
    """
    x, y = at
    k = scale
    points = [(x, y - 0.06 * k, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y - 0.18 * k, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y - 0.09 * k, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        mcp, pip, dip, tip = HAND_DIGITS[digit]
        fx = x + (offset - 1) * 0.012 * k
        points[mcp], points[pip], points[dip] = (fx, y - 0.09 * k, 0.0), (fx, y - 0.06 * k, 0.0), (fx, y - 0.03 * k, 0.0)
        points[tip] = (fx, y, 0.0)
    points[HAND_DIGITS["thumb"][3]] = (x, y, 0.0)
    points[HAND_DIGITS["index"][3]] = (x, y, 0.0)
    return HandObservation(handedness=side, landmarks=points)


def test_the_pinching_fingers_stay_on_top():
    pinch_at = (TARGETS["cheekL"][0] + 0.3 * H / W, TARGETS["cheekL"][1])
    img, out = warp(pulling("cheekL", 0.3, 0.0), hands=[hand(pinch_at)])
    finger = (pinch_at[0], pinch_at[1] - 0.04)
    x, y = px(finger)
    assert np.array_equal(out[y - 1 : y + 2, x - 1 : x + 2], img[y - 1 : y + 2, x - 1 : x + 2])


def test_a_finger_grab_stretches_the_finger_and_not_the_face():
    left = hand((0.85, 0.6), side="Left")
    tip = left.landmarks[HAND_DIGITS["index"][3]]
    grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=tip[:2], pull=(0.0, 0.08))
    img, out = warp(grab, hands=[left])
    assert not np.array_equal(out, img), "the finger stretches"
    assert_not_moved(out, img, norm(0.0, 0.0), radius=10)


# --- light ---------------------------------------------------------------------


def test_the_light_direction_is_read_from_the_face():
    img = image()
    lit_right = img.copy()
    lit_right[:, W // 2 :] = np.clip(lit_right[:, W // 2 :].astype(int) + 50, 0, 255).astype(np.uint8)
    assert estimate_light(lit_right, FACE)[0] > estimate_light(img, FACE)[0]
    lit_above = img.copy()
    lit_above[: H // 2] = np.clip(lit_above[: H // 2].astype(int) + 50, 0, 255).astype(np.uint8)
    assert estimate_light(lit_above, FACE)[1] < estimate_light(img, FACE)[1]


# --- motion --------------------------------------------------------------------


def test_a_release_snaps_back_past_rest_then_settles():
    warper = FaceWarper(seed=1)
    frame = Frame(t_ms=0.0, face=FACE)
    grab = pulling("cheekL", 0.3, 0.0)
    for _ in range(3):
        warper.update({"Right": grab}, frame, (W, H))
    reach = warper._sizes[1]
    along = []
    for _ in range(40):
        warper.update({}, frame, (W, H))
        along.append(float(warper.pull_px[0]))
    assert min(along) < 0.0, "it overshoots rest, like rubber"
    assert min(along) >= -MAX_REBOUND * reach - 1e-6, "but only a little"
    assert along[-1] == 0.0, "and settles"


# --- speed ---------------------------------------------------------------------


def test_the_effect_fits_a_live_frame_at_720p():
    """Inference already takes ~45ms; the effect must not double that."""
    warper = FaceWarper(seed=1)
    frame = Frame(t_ms=0.0, face=FACE)
    img = image()
    grab = pulling("cheekL", 0.35, 0.0)
    warper.update({"Right": grab}, frame, (W, H))
    warper.warp(img)

    def one_frame():
        warper.update({"Right": grab}, frame, (W, H))
        warper.warp(img)

    assert fastest(one_frame) < 0.030


def test_a_barely_pulled_pinch_is_a_gentle_rise_not_a_bump():
    """The tent lifts as far as the skin is pulled: at a 2px pull the skin
    barely rises, so it casts no visible shadow and shows no bump.

    Compared as local brightness (blurred): a bump or shadow changes that;
    the fixture's pixel-sized skin texture shifting by 2px does not.
    """
    img, out = warp(pulling("cheekL", 2.0 / H, 0.0))
    x, y = px(TARGETS["cheekL"])

    def brightness(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame[y - 40 : y + 40, x - 40 : x + 40], cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5.0)

    # A full-height bump on a 2px pull changed local brightness by ~53; the
    # honest shading of a 2px rise is ~3.
    assert np.abs(brightness(out) - brightness(img)).max() < 8.0



# --- regressions from review ---------------------------------------------------


def test_more_moved_pixels_than_remap_takes_in_one_column_do_not_crash():
    """OpenCV's remap refuses an output taller than 32767 rows; the moved
    pixels used to be fed to it as one (N, 1) column. A close face or hand
    moves more pixels than that, and the whole service died."""
    from ofp_vision.skin import Masks
    from ofp_vision.warp import Look, render, solve

    pinch = Pinch(anchor=(400.0, 360.0), pull=(500.0, 0.0), core=30.0, reach=150.0)
    window = pinch.window((W, H))
    masks = Masks.of(window, np.full(window.shape, 255, np.uint8))
    assert solve(pinch, window).h.size > 32767
    out = render(image(), pinch, masks, Look(np.array([0.0, 0.0, 1.0]), 0.0))
    assert out.shape == (H, W, 3)


def test_a_close_hand_finger_grab_does_not_crash():
    left = hand((0.75, 0.75), side="Left", scale=2.4)
    tip = left.landmarks[HAND_DIGITS["index"][3]]
    grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=tip[:2], pull=(0.0, 0.1))
    img, out = warp(grab, hands=[left])
    assert not np.array_equal(out, img)


def test_a_close_face_pulled_far_does_not_crash():
    face, img = landmarks(scale=1.8), image(scale=1.8)
    targets = face_target_points(face)
    grab = GrabState(active=True, hand="Right", target="cheekL", anchor=targets["cheekL"][:2], pull=(0.6 * H / W, 0.0))
    _, out = warp(grab, frame_image=img, face=face)
    assert not np.array_equal(out, img)


def room_like(frame: np.ndarray) -> np.ndarray:
    """The fixture's room is blue-green; skin is red-dominant, teeth neutral."""
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    return (b > r + 20) & (g > r + 20)


@pytest.mark.parametrize(
    "target, dx, dy",
    [("chin", 0.0, -0.1), ("earL", -0.33, 0.0), ("jawL", -0.1, -0.1), ("cheekR", 0.15, 0.0)],
)
def test_an_inward_pull_never_paints_the_room_onto_the_face(target, dx, dy):
    """Pull skin inward and the gap it leaves must not fill with the room:
    only pixels whose source is skin may show moved content."""
    img, out = warp(pulling(target, dx, dy))
    assert (room_like(out) & ~room_like(img)).sum() == 0


@pytest.mark.parametrize("target", ["nose", "lipTop"])
def test_a_long_pull_on_a_small_feature_does_not_crash(target):
    """A window wholly inside the face has no edge band to colour-check."""
    img, out = warp(pulling(target, 0.0, 0.16 if target == "nose" else -0.16))
    assert out.shape == img.shape


def test_a_face_partly_out_of_frame_does_not_crash_or_bend_the_room():
    """The forehead is above the frame. Skin may be pulled out over the room
    (that is the stretch); the room itself must not move. Moved room shows
    as room-coloured pixels whose content changed - not just darkened by
    the shadow - away from the strand's antialiased outline."""
    face, img = landmarks(centre=(0.5, 0.08)), image(centre=(0.5, 0.08))
    targets = face_target_points(face)
    grab = GrabState(active=True, hand="Right", target="cheekL", anchor=targets["cheekL"][:2], pull=(0.3 * H / W, 0.0))
    _, out = warp(grab, frame_image=img, face=face)
    covered = room_like(img) & ~room_like(out)
    still_room = room_like(img) & room_like(out)
    ratio = (out.astype(float) + 1.0) / (img.astype(float) + 1.0)
    changed_content = still_room & (ratio.max(axis=2) - ratio.min(axis=2) > 0.06)
    outline = cv2.dilate(covered.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert covered.sum() > 1000, "the cheek is pulled out over the room"
    assert (changed_content & ~outline).sum() == 0


def test_the_snap_back_takes_a_few_frames_not_one():
    """Real rubber snaps fast but visibly: most of the pull is still there a
    frame after letting go, and nearly none two frames later."""
    warper = FaceWarper(seed=1)
    frame = Frame(t_ms=0.0, face=FACE)
    grab = pulling("cheekL", 0.3, 0.0)
    for _ in range(3):
        warper.update({"Right": grab}, frame, (W, H))
    length = float(warper.pull_px[0])
    along = []
    for _ in range(3):
        warper.update({}, frame, (W, H))
        along.append(float(warper.pull_px[0]))
    assert along[0] > 0.4 * length
    assert abs(along[1]) < 0.25 * length


def test_the_light_is_not_measured_off_frame():
    face = landmarks(centre=(0.5, -0.05))
    assert estimate_light(image(centre=(0.5, -0.05)), face) is None



def test_a_face_close_to_the_lens_is_drawn_at_a_lower_level_of_detail_in_time():
    """A face filling a third of the frame moves ~100k pixels on a long pull.
    It is drawn at half resolution and composited back: still in budget."""
    face, img = landmarks(scale=2.2), image(scale=2.2)
    targets = face_target_points(face)
    grab = GrabState(active=True, hand="Right", target="cheekL", anchor=targets["cheekL"][:2], pull=(0.55 * H / W, 0.0))
    warper = FaceWarper(seed=1)
    frame = Frame(t_ms=0.0, face=face)
    warper.update({"Right": grab}, frame, (W, H))
    out = warper.warp(img)
    assert not np.array_equal(out, img)

    def one_frame():
        warper.update({"Right": grab}, frame, (W, H))
        warper.warp(img)

    # ~17-31ms measured (the higher with the lab running alongside); the
    # budget catches a real slowdown, not noise.
    assert fastest(one_frame) < 0.045


def test_the_low_detail_drawing_matches_the_full_one():
    """Same stretch, drawn both ways: the difference is only softness."""
    import ofp_vision.warp as warp_module

    grab = pulling("cheekL", 0.3, 0.0)
    _, full = warp(grab)
    saved = warp_module.LOD_AREA, warp_module.MAX_LOD_SHRINK
    warp_module.LOD_AREA, warp_module.MAX_LOD_SHRINK = 1.0, 2  # force half resolution
    try:
        img, low = warp(grab)
    finally:
        warp_module.LOD_AREA, warp_module.MAX_LOD_SHRINK = saved
    changed = np.abs(full.astype(int) - img.astype(int)).max(axis=2) > 0
    assert changed.sum() > 1000
    assert np.abs(low.astype(int) - full.astype(int))[changed].mean() < 12.0
    unchanged = ~cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert np.array_equal(low[unchanged], img[unchanged]), "the rest of the frame stays full-resolution"
