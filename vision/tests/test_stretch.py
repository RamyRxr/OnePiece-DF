"""Tests for the skin-physics stretch effect (GPU), end to end.

Same rules as the CPU effect: only skin moves; eyes, teeth and the room never
do; a release snaps back quickly and cleanly, then the mesh is dropped. Each
hand stretches on its own - one hand, or both at once.
Skipped where no OpenGL 3.3 context can be made.
"""

from __future__ import annotations

import numpy as np
import pytest
from face_fixture import EYE_L, H, W, image, landmarks, norm, px

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


def test_the_teeth_never_stretch(stretcher):
    img, out = hold(stretcher, pulling("lipBottom", 0.0, 0.15))
    let_go(stretcher)
    x, y = px(norm(0.0, 0.08))
    assert np.abs(out[y - 2 : y + 2, x - 2 : x + 2].astype(int) - img[y - 2 : y + 2, x - 2 : x + 2]).max() <= 2


def test_let_go_it_snaps_back_quickly_and_is_dropped(stretcher):
    hold(stretcher, pulling("cheekL", 0.3, 0.0))
    (session,) = stretcher._sessions
    centre = session.holds["Right"].core[0]
    rest = session.frame.to_image(session.patch.local[[centre]])[0]
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
    moved_wall = changed & room_like(img) & no_skin_added
    assert moved_wall.sum() == 0, "no piece of the wall came along with the finger"
