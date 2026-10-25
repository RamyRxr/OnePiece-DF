"""Tests for the skin-physics stretch effect (GPU), end to end.

Same rules as the CPU effect: only skin moves; eyes, teeth and the room never
do; a release snaps back quickly and cleanly, then the mesh is dropped. Each
hand stretches on its own - one hand, or both at once.
Skipped where no OpenGL 3.3 context can be made.
"""

from __future__ import annotations

import numpy as np
import pytest
from face_fixture import CENTRE, EYE_L, H, W, image, landmarks, norm, px

moderngl = pytest.importorskip("moderngl")
try:
    moderngl.create_standalone_context().release()
except Exception:  # pragma: no cover - depends on the machine
    pytest.skip("no OpenGL context on this machine", allow_module_level=True)

from ofp_vision.grab import GrabState, face_target_points  # noqa: E402
from ofp_vision.signals import Frame  # noqa: E402
from ofp_vision.stretch import SkinStretcher  # noqa: E402

FACE = landmarks()
TARGETS = face_target_points(FACE)
FRAME = Frame(t_ms=0.0, face=FACE)


@pytest.fixture(scope="module")
def stretcher():
    return SkinStretcher()


def pulling(target: str, dx: float, dy: float, hand: str = "Right", anchor=None) -> GrabState:
    anchor = TARGETS[target][:2] if anchor is None else anchor
    return GrabState(active=True, hand=hand, target=target, anchor=anchor, pull=(dx * H / W, dy))


def eased(grab: GrabState, k: float) -> GrabState:
    return GrabState(active=True, hand=grab.hand, target=grab.target, anchor=grab.anchor,
                     pull=(grab.pull[0] * k, grab.pull[1] * k))


def hold(stretcher: SkinStretcher, *grabs: GrabState, frames: int = 25):
    """Each grab's hand pinches and pulls out over a quarter second, then holds."""
    img = image()
    out = img
    for i in range(frames):
        k = min(1.0, (i + 1) / 8)
        stretcher.update({grab.hand: eased(grab, k) for grab in grabs}, FRAME, (W, H))
        out = stretcher.warp(img)
    return img, out


def let_go(stretcher: SkinStretcher, frames: int = 90) -> None:
    for _ in range(frames):
        stretcher.update({}, FRAME, (W, H))
        stretcher.warp(image())


def room_like(frame: np.ndarray) -> np.ndarray:
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    return (b > r + 20) & (g > r + 20)


def test_nothing_is_drawn_without_a_grab(stretcher):
    img = image()
    stretcher.update({}, FRAME, (W, H))
    assert stretcher.warp(img) is img


def test_a_pull_stretches_the_skin_out_over_the_room(stretcher):
    img, out = hold(stretcher, pulling("cheekL", 0.3, 0.0))
    let_go(stretcher)
    x, y = px(norm(0.24, 0.04))
    b, g, r = out[y, x].astype(int)
    assert r > 150 and r > b, "skin-coloured where the room was"


def test_the_eyes_and_the_room_beside_the_face_never_move(stretcher):
    img, out = hold(stretcher, pulling("cheekL", 0.3, 0.0))
    let_go(stretcher)
    x, y = px(norm(*EYE_L))
    # Allowing 1 level: the GPU's 8-bit round trip rounds.
    assert np.abs(out[y - 4 : y + 4, x - 8 : x + 8].astype(int) - img[y - 4 : y + 4, x - 8 : x + 8]).max() <= 1
    assert (room_like(img) & ~room_like(out) & (np.arange(H)[:, None] < px(norm(0, -0.21))[1])).sum() == 0


def test_a_cheek_pull_leaves_the_far_side_of_the_face_alone(stretcher):
    """No seam where the mesh ends: just beyond its edge, across the face from
    the pull, nothing changes (it once showed there as a line of wall)."""
    img, out = hold(stretcher, pulling("cheekL", 0.3, 0.0))
    (session,) = stretcher._sessions
    edge = int(session.rest()[:, 0].min())
    let_go(stretcher)
    (_, y0), (_, y1) = px(norm(0.0, -0.02)), px(norm(0.0, 0.03))
    band = slice(edge - 15, edge - 1)
    assert np.abs(out[y0:y1, band].astype(int) - img[y0:y1, band]).max() <= 2


def teeth_columns(frame: np.ndarray) -> np.ndarray:
    """The columns where the mouth shows (white teeth), along its middle row."""
    row = frame[px(norm(0.0, 0.09))[1]].astype(int)
    return np.nonzero(row.min(axis=1) > 200)[0]


def test_pulling_a_mouth_corner_stretches_the_mouth_itself(stretcher):
    """Luffy's mouth: pulled at its corner, the mouth widens toward the pull -
    not a strip of skin pulled out beside a mouth that stays as it was."""
    img, out = hold(stretcher, pulling("mouthL", 0.15, 0.0))
    let_go(stretcher)
    assert teeth_columns(out).max() > teeth_columns(img).max() + 40


def test_let_go_it_snaps_back_quickly_and_is_dropped(stretcher):
    hold(stretcher, pulling("cheekL", 0.3, 0.0))
    (session,) = stretcher._sessions
    centre = session.holds["Right"].core[0]
    rest = session.rest()[centre, :2]
    track = []
    for _ in range(90):
        stretcher.update({}, FRAME, (W, H))
        stretcher.warp(image())
        if not stretcher._sessions:
            break
        track.append(session.body.positions[centre, 0] - rest[0])
    assert len(track) <= 10, "back and still within a third of a second"
    assert not stretcher._sessions, "and once still, the mesh is dropped"


# --- both hands at once --------------------------------------------------------


def skin_where_room_was(out: np.ndarray, point) -> bool:
    x, y = px(point)
    b, g, r = out[y, x].astype(int)
    return r > 150 and r > b


def test_both_hands_stretch_two_parts_at_once(stretcher):
    """Luffy's pose: both cheeks pulled out, one by each hand."""
    _, out = hold(stretcher, pulling("cheekL", 0.3, 0.0, "Right"), pulling("cheekR", -0.3, 0.0, "Left"))
    let_go(stretcher)
    assert skin_where_room_was(out, norm(0.24, 0.04)), "the left cheek is out"
    assert skin_where_room_was(out, norm(-0.24, 0.04)), "and so is the right"


def test_one_hand_letting_go_leaves_the_other_stretched(stretcher):
    hold(stretcher, pulling("cheekL", 0.3, 0.0, "Right"), pulling("cheekR", -0.3, 0.0, "Left"))
    img = image()
    out = img
    for _ in range(30):
        stretcher.update({"Left": pulling("cheekR", -0.3, 0.0, "Left")}, FRAME, (W, H))
        out = stretcher.warp(img)
    let_go(stretcher)
    assert skin_where_room_was(out, norm(-0.24, 0.04)), "the right cheek is still out"
    assert not skin_where_room_was(out, norm(0.24, 0.04)), "the left one went back"


def test_two_pinches_close_together_share_one_piece_of_skin(stretcher):
    """Both hands on one cheek pull the same skin, not two copies of it."""
    near = (TARGETS["cheekL"][0], TARGETS["cheekL"][1] + 0.06)
    hold(stretcher, pulling("cheekL", 0.2, 0.0, "Right"), pulling("cheekL", 0.2, 0.1, "Left", anchor=near))
    holders = [set(session.holds) for session in stretcher._sessions]
    let_go(stretcher)
    assert holders == [{"Right", "Left"}], "one mesh, held by both hands"



# --- the screenshot: a finger pulled across a wall -----------------------------


def hand_on_the_wall():
    """A left hand on the textured wall, its fingers labelled body skin (and
    nothing else) by a synthetic segmentation, as the real model does."""
    import cv2

    from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
    from ofp_vision.signals import HandObservation
    from ofp_vision.skin import SEGMENT_BODY_SKIN, SEGMENT_FACE_SKIN

    x, y = 0.8, 0.75
    points = [(x, y, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y + 0.2, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        fx = x + (offset - 2) * 0.04
        for j, joint in enumerate(HAND_DIGITS[digit]):
            points[joint] = (fx, y - 0.07 * j, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y, 0.0)
    hand = HandObservation(handedness="Left", landmarks=points)

    img = image()
    labels = np.zeros((H // 2, W // 2), dtype=np.uint8)
    cv2.circle(labels, (W // 4, H // 4), int(0.2 * H / 2), SEGMENT_FACE_SKIN, -1)
    for digit in HAND_DIGITS.values():
        a = (int(points[digit[0]][0] * W), int(points[digit[0]][1] * H))
        b = (int(points[digit[3]][0] * W), int(points[digit[3]][1] * H))
        cv2.line(img, a, b, (120, 160, 215), 16)
        cv2.line(labels, (a[0] // 2, a[1] // 2), (b[0] // 2, b[1] // 2), SEGMENT_BODY_SKIN, 8)
    return hand, img, labels


def test_a_finger_pulled_across_the_wall_never_carries_the_wall(stretcher):
    """The screenshot: the index finger pulled far up and away. Its stretch
    may cover the wall (that is the finger, in front of it) but must never
    show the wall moved - no blue sheet coming along with the finger."""
    hand, img, labels = hand_on_the_wall()
    tip = hand.landmarks[8]
    frame = Frame(t_ms=0.0, face=FACE, hands=[hand], segmentation=labels)
    out = img
    for i in range(25):
        k = min(1.0, (i + 1) / 8)
        grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=tip[:2],
                         pull=(-0.15 * k, -0.3 * k))
        stretcher.update({"Right": grab}, frame, (W, H))
        out = stretcher.warp(img)
    let_go(stretcher)

    changed = np.abs(out.astype(int) - img.astype(int)).max(axis=2) > 12
    assert changed.sum() > 500, "the finger did stretch"
    # Where the finger now covers the wall, its colour is mixed in - fully,
    # or partly along its soft edge - and skin is far redder than this wall:
    # red rises. Moved wall is different: the colour changes with no skin
    # added. That must never happen.
    no_skin_added = out[..., 2].astype(int) <= img[..., 2].astype(int) + 5
    # Where the finger itself was is refilled with what is behind it (the
    # next test) - give or take a pixel, as the segmentation is never exact.
    import cv2

    finger = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    finger = cv2.dilate(finger.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    moved_wall = changed & room_like(img) & no_skin_added & ~finger
    assert moved_wall.sum() == 0, "no piece of the wall came along with the finger"


def test_where_a_finger_was_pulled_away_from_shows_the_wall(stretcher):
    """No ghost: the top of the finger, pulled off sideways, is no longer
    also drawn where it was - the wall behind it shows there."""
    hand, img, labels = hand_on_the_wall()
    tip = hand.landmarks[8]
    frame = Frame(t_ms=0.0, face=FACE, hands=[hand], segmentation=labels)
    out = img
    for i in range(25):
        k = min(1.0, (i + 1) / 8)
        grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=tip[:2],
                         pull=(0.12 * k, -0.05 * k))
        stretcher.update({"Right": grab}, frame, (W, H))
        out = stretcher.warp(img)
    let_go(stretcher)

    x, y = int(tip[0] * W), int(tip[1] * H) + 20
    assert not room_like(img)[y, x], "the finger was here"
    assert room_like(out)[y - 2 : y + 3, x - 2 : x + 3].all(), "and now the wall is"


def test_moving_the_head_after_a_pull_leaves_no_mask_behind(stretcher):
    """Let go and keep moving your head: the stretch snaps back and is gone,
    and your face is just the camera's face again - not a copy of it sliding
    along behind, like a mask."""
    hold(stretcher, pulling("cheekL", 0.3, 0.0))
    for i in range(1, 13):
        centre = (CENTRE[0] + 0.004 * i, CENTRE[1])  # ~5px a frame
        stretcher.update({}, Frame(t_ms=0.0, face=landmarks(centre=centre)), (W, H))
        stretcher.warp(image(centre=centre))
    assert not stretcher._sessions


def test_a_finger_pinched_halfway_takes_its_tip_along_whole(stretcher):
    """Rubber, not a noodle: past the pinch the finger travels whole and as
    wide as it was (turned to follow the pull); between the knuckle and the
    pinch, it stretches."""
    hand, img, labels = hand_on_the_wall()
    tip = hand.landmarks[8]
    frame = Frame(t_ms=0.0, face=FACE, hands=[hand], segmentation=labels)
    out = img
    for i in range(25):
        k = min(1.0, (i + 1) / 8)
        grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=(tip[0], tip[1] + 0.1),
                         pull=(0.15 * k, -0.12 * k))
        stretcher.update({"Right": grab}, frame, (W, H))
        out = stretcher.warp(img)
    (session,) = stretcher._sessions
    points = session.body.positions[:, :2].copy()
    pivot = points[session.holds["Right"].core[0]]
    let_go(stretcher)

    # Across the held part, halfway from the pinch to the tip: the finger.
    middle = (pivot + points[-3]) / 2
    along = (points[-3] - pivot) / np.linalg.norm(points[-3] - pivot)
    across = np.array([-along[1], along[0]])
    line = np.round(middle + np.arange(-25, 26)[:, None] * across).astype(int)
    colours = out[line[:, 1], line[:, 0]].astype(int)
    skin = (colours[:, 2] > 150) & (colours[:, 2] > colours[:, 0])
    assert skin.sum() >= 13, "the top of the finger, moved whole: as wide as the finger (16px)"


def test_a_hand_lost_for_a_moment_is_not_learned_as_background(stretcher):
    """Pinching, one hand covers the other, and the tracker loses it for a few
    frames. Its finger must not be learned as what is behind it - or the
    ghost comes back the moment it is pulled away."""
    hand, img, labels = hand_on_the_wall()
    tip = hand.landmarks[8]
    lost = Frame(t_ms=0.0, face=FACE, hands=[], segmentation=labels)
    seen = Frame(t_ms=0.0, face=FACE, hands=[hand], segmentation=labels)
    out = img
    for i in range(25):
        k = min(1.0, (i + 1) / 8)
        grab = GrabState(active=True, hand="Right", target="finger.Left.index", anchor=tip[:2],
                         pull=(0.12 * k, -0.05 * k))
        stretcher.update({"Right": grab}, lost if 3 <= i < 9 else seen, (W, H))
        out = stretcher.warp(img)
    let_go(stretcher)

    x, y = int(tip[0] * W), int(tip[1] * H) + 20
    assert room_like(out)[y - 2 : y + 3, x - 2 : x + 3].all(), "the wall, not the finger learned while it was lost"
