"""The skin-physics stretch effect, drawn on the graphics card.

Ties the pieces together, frame by frame:

    grab starts  a skin mesh is laid over the pinched area (`skinmesh.py`)
                 and given physics (`softbody.py`): pinned at the eyes and
                 at its border, cut along the mouth opening (the lips are
                 only joined at the corners), and made only of real skin -
                 never hair, background or teeth (`skin.py`)
    each frame   the mesh's resting places follow the head; the fingers
                 hold the pinched points; physics moves the rest, with
                 weight, drag and wobble. It is drawn on the graphics card,
                 painted with the camera image and lit (`gpu.py`), then
                 composited onto the frame, with the hands put back on top
    let go       physics snaps it back - past rest, a wobble, still - and
                 then the mesh is dropped

Each hand grabs on its own, so two stretches can run at once. Each body part
- the face, or one finger - has its own mesh; both hands pinching the same
part of the face hold the same mesh. Meshes are drawn one over the other,
the face first.

Same interface as `warp.FaceWarper` (update, then warp), which stays as the
fallback where no graphics card can be used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .grab import GrabState, finger_owner
from .gpu import SkinRenderer, vertex_normals
from .indices import FACE_EYE_POINTS
from .light import default_light, followed_light
from .signals import Frame, HandObservation, Point
from .skin import (
    FEATHER_PX,
    Window,
    digit_mask,
    face_masks,
    feathered,
    hand_mask,
    segmented_digit_mask,
    segmented_face_masks,
)
from .skinmesh import BodyFrame, SkinPatch, build_patch, face_frame, finger_frame
from .softbody import SoftBody
from .warp import CORE_BY_TARGET, CORE_PER_FACE

# The mesh over a face: this far around the pinch, points this far apart
# (both in eye-distances): ~3,700 points, ~5px apart at webcam distance.
FACE_PATCH_HALF = 0.9
FACE_SPACING = 0.03
# Over a finger: in palm lengths; the finger itself is the skin.
FINGER_PATCH_HALF = 1.0
FINGER_SPACING = 0.03
FINGER_CORE = 0.1
# Points within this of the eye outline are pinned: the eyes never move.
EYE_PIN = 0.1
# The mesh's outer rows are pinned too: the skin beyond holds it.
BORDER_ROWS = 2
# A point is skin (and moves) above this much skin in the mask.
SKIN_MIN = 0.05
# The visible edge starts this far inside the mask's soft edge (which is
# 0.5 on the skin's outline), where the camera's pixels are pure skin. On
# the outline itself they are part wall, hair or background - stretched,
# that shows as a fringe of the room along the strand.
VISIBLE_FROM, VISIBLE_TO = 0.75, 0.98
# The fingers hold the inner part of the pinch firmly (this fraction of
# its radius); beyond, the grip fades to a gentle pull at its rim.
GRIP_FIRM = 0.5
# Pulled skin comes off the face toward the camera, this much per pixel of
# pull (until the real depth of the hand is measured).
LIFT_PER_PULL = 0.3


# The mesh of the face is shared: a second pinch on it joins the mesh that
# is already moving, if it lands this far inside its edge (a fraction of the
# mesh's half size). Nearer the edge, where it is pinned, it gets its own.
JOIN_INSIDE = 0.7
FACE_PART = "face"


@dataclass(frozen=True)
class Hold:
    """One hand's pinch on a mesh: the points it holds, and how firmly."""

    target: str
    core: np.ndarray
    grip: np.ndarray


@dataclass
class Session:
    """One body part's mesh and physics - the face, or one finger - from the
    first pinch on it until it has settled. Each hand on it has a `Hold`."""

    part: str
    patch: SkinPatch
    body: SoftBody
    # What is visible, per pixel: the skin mask over the window the mesh
    # was laid on, and where each point sat in it.
    skin: np.ndarray
    skin_uv: np.ndarray
    is_skin: np.ndarray
    frame: BodyFrame
    holds: dict[str, Hold] = field(default_factory=dict)


def part_of(target: str) -> str:
    """The body part a target is on: the face, or that finger."""
    return target if finger_owner(target) is not None else FACE_PART


def _sample(mask: np.ndarray, points: np.ndarray, window: Window) -> np.ndarray:
    """A window-local mask read at frame-pixel points (nearest; 0 outside)."""
    cols = np.round(points[:, 0]).astype(int) - window.x0
    rows = np.round(points[:, 1]).astype(int) - window.y0
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    values = np.zeros(len(points), dtype=np.float32)
    values[inside] = mask[rows[inside], cols[inside]]
    return values


def _window_around(points: np.ndarray, size: tuple[int, int], margin: int) -> Window:
    low, high = points.min(axis=0), points.max(axis=0)
    return Window(int(low[0]), int(low[1]), int(high[0]) + 1, int(high[1]) + 1).grown(margin, size)


class SkinStretcher:
    """Turns each hand's grab into the drawn stretch, with skin physics."""

    def __init__(self, renderer: SkinRenderer | None = None) -> None:
        self._renderer = renderer or SkinRenderer()
        self._sessions: list[Session] = []
        self._grabs: dict[str, GrabState] = {}
        self._face: list[Point] | None = None
        self._hands: list[HandObservation] = []
        self._segmentation: np.ndarray | None = None
        self._size = (0, 0)
        self._light = default_light()
        _warm_up()

    def update(self, grabs: Mapping[str, GrabState], frame: Frame, size: tuple[int, int]) -> None:
        """Take this frame's grabs (one per hand) and landmarks. `size` is
        (width, height)."""
        self._grabs = dict(grabs)
        self._size = size
        if frame.face is not None:
            self._face = frame.face
        self._hands = list(frame.hands)
        if frame.segmentation is not None:
            self._segmentation = frame.segmentation

    def warp(self, bgr: np.ndarray) -> np.ndarray:
        """Return a copy of `bgr` with the stretches drawn (or `bgr` if none)."""
        self._follow_grabs(bgr)
        if not self._sessions:
            # Follow the room's light between pinches (a pinching hand over
            # the cheek would read as a shadow).
            self._light = followed_light(self._light, bgr, self._face)
            return bgr
        out, alive = bgr, []
        # The face first, then fingers: a finger pulled over the face is in front.
        for session in sorted(self._sessions, key=lambda s: s.part != FACE_PART):
            rest = self._step(session)
            if session.holds or not session.body.settled(rest):
                alive.append(session)
                out = self._draw(out, session, rest)
        self._sessions = alive
        return out

    # --- which hand holds what ----------------------------------------------

    def _follow_grabs(self, bgr: np.ndarray) -> None:
        """Let go of what each hand released; take hold of what it pinched."""
        for session in self._sessions:
            session.holds = {
                hand: hold for hand, hold in session.holds.items() if _still_holding(self._grabs.get(hand), hold)
            }
        holding = {hand for session in self._sessions for hand in session.holds}
        for hand, grab in self._grabs.items():
            if grab.active and grab.anchor is not None and hand not in holding:
                self._take_hold(bgr, hand, grab)

    def _take_hold(self, bgr: np.ndarray, hand: str, grab: GrabState) -> None:
        """Join a mesh already on that body part, or lay a new one."""
        part = part_of(grab.target)
        for session in self._sessions:
            hold = self._hold(session, grab) if session.part == part else None
            if hold is not None:
                session.holds[hand] = hold
                return
        session = self._start(bgr, grab)
        hold = self._hold(session, grab) if session is not None else None
        if session is not None and hold is not None:
            session.holds[hand] = hold
            self._sessions.append(session)

    def _hold(self, session: Session, grab: GrabState) -> Hold | None:
        """The points of `session`'s mesh a pinch at the grab's anchor holds:
        firm in the middle of the pinch, gentle at its rim. None if the pinch
        is not well inside the mesh, or holds no skin of it."""
        anchor = session.frame.to_local(np.array([self._px(grab.anchor)]))[0]
        low, high = session.patch.local.min(axis=0), session.patch.local.max(axis=0)
        centre, half = (low + high) / 2, (high - low) / 2
        if np.any(np.abs(anchor - centre) > JOIN_INSIDE * half):
            return None
        finger = finger_owner(grab.target) is not None
        core_size = FINGER_CORE if finger else CORE_BY_TARGET.get(grab.target, CORE_PER_FACE)
        reach = np.linalg.norm(session.patch.local - anchor, axis=1)
        free = session.is_skin & ~session.body.pinned
        for other in session.holds.values():
            free[other.core] = False
        core = np.nonzero((reach <= core_size) & free)[0]
        if len(core) == 0:
            return None
        grip = np.clip(1.0 - (reach[core] / core_size - GRIP_FIRM) / (1.0 - GRIP_FIRM), 0.05, 1.0)
        return Hold(grab.target, core, grip)

    def _px(self, point) -> tuple[float, float]:
        return (point[0] * self._size[0], point[1] * self._size[1])

    # --- a body part's mesh -------------------------------------------------

    def _body_frame(self, part: str) -> BodyFrame | None:
        owner = finger_owner(part)
        if owner is None:
            return face_frame(self._face, self._size) if self._face is not None else None
        hand = next((h for h in self._hands if h.handedness == owner), None)
        return finger_frame(hand, part.split(".")[2], self._size) if hand is not None else None

    def _start(self, bgr: np.ndarray, grab: GrabState) -> Session | None:
        """Lay a mesh over the body part around the pinch."""
        part = part_of(grab.target)
        frame = self._body_frame(part)
        if frame is None:
            return None
        finger = part != FACE_PART
        anchor = frame.to_local(np.array([self._px(grab.anchor)]))[0]
        patch = build_patch(
            anchor, FINGER_PATCH_HALF if finger else FACE_PATCH_HALF, FINGER_SPACING if finger else FACE_SPACING
        )
        rest = frame.to_image(patch.local)
        window = _window_around(rest, self._size, FEATHER_PX + 2)
        skin, mouth = self._masks(bgr, part, window)
        soft = feathered(skin)
        ramp = np.clip((soft - VISIBLE_FROM) / (VISIBLE_TO - VISIBLE_FROM), 0.0, 1.0)
        visible = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
        skin_uv = (rest - [window.x0, window.y0]) / [window.shape[1], window.shape[0]]
        # Every point moves - the ones that are not skin just stay invisible:
        # the GPU shows a pixel only if where it came from was skin. Only the
        # mouth opening cuts springs.
        edges = patch.edges
        if mouth is not None:
            edges = edges[~_crosses(mouth, rest, edges, window)]
        pinned = _border(patch)
        if part == FACE_PART:
            pinned = pinned | self._near_eyes(rest, frame)
        rest3 = np.zeros((len(rest), 3))
        rest3[:, :2] = rest
        body = SoftBody(rest3, edges, pinned)
        is_skin = _sample(soft, rest, window) > SKIN_MIN
        return Session(part, patch, body, visible, skin_uv, is_skin, frame)

    def _step(self, session: Session) -> np.ndarray:
        """One frame of a mesh's physics: its resting places follow the body
        part, each hand holds its points. Returns the resting places."""
        session.frame = self._body_frame(session.part) or session.frame
        rest = np.zeros((len(session.patch.local), 3))
        rest[:, :2] = session.frame.to_image(session.patch.local)
        holds = list(session.holds.items())
        if not holds:
            session.body.step(rest, np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
            return rest
        grabbed = np.concatenate([hold.core for _, hold in holds])
        targets = np.concatenate([rest[hold.core] + self._pull(self._grabs[hand]) for hand, hold in holds])
        grip = np.concatenate([hold.grip for _, hold in holds])
        session.body.step(rest, grabbed, targets, grip)
        return rest

    def _pull(self, grab: GrabState) -> np.ndarray:
        """A grab's pull in pixels - and off the skin, toward the camera."""
        pull = np.array([grab.pull[0] * self._size[0], grab.pull[1] * self._size[1], 0.0])
        pull[2] = LIFT_PER_PULL * float(np.hypot(pull[0], pull[1]))
        return pull

    def _masks(self, bgr: np.ndarray, part: str, window: Window):
        """(skin, mouth) over `window`: from the segmentation model's labels
        when it has run - real edges - else from shapes and skin colour."""
        owner = finger_owner(part)
        labels = self._segmentation
        if owner is None:
            if labels is not None:
                return segmented_face_masks(labels, self._face, self._size, window)
            return face_masks(bgr, self._face, window)
        hand = next(h for h in self._hands if h.handedness == owner)
        digit = part.split(".")[2]
        if labels is not None:
            return segmented_digit_mask(labels, hand, digit, self._size, window), None
        return digit_mask(hand, digit, self._size, window), None

    def _near_eyes(self, rest: np.ndarray, frame: BodyFrame) -> np.ndarray:
        if self._face is None:
            return np.zeros(len(rest), dtype=bool)
        eyes = np.array([[self._face[i][0] * self._size[0], self._face[i][1] * self._size[1]] for i in FACE_EYE_POINTS])
        nearest = np.min(np.linalg.norm(rest[:, None, :] - eyes[None, :, :], axis=2), axis=1)
        return nearest < EYE_PIN * frame.scale

    # --- drawing ------------------------------------------------------------

    def _draw(self, bgr: np.ndarray, session: Session, rest: np.ndarray) -> np.ndarray:
        positions = session.body.positions
        # All of the mesh is drawn; the GPU discards whatever did not come
        # from skin, pixel by pixel.
        drawn = session.patch.triangles
        order = np.argsort(positions[drawn].mean(axis=1)[:, 2], kind="stable")
        box = _window_around(positions[:, :2], self._size, 2)
        if box.empty:
            return bgr
        normals = vertex_normals(positions, session.patch.triangles)
        # Pinned skin (the eyes) has not moved: it keeps exactly its own light.
        normals[session.body.pinned] = (0.0, 0.0, 1.0)
        uv = rest[:, :2] / np.array(self._size, dtype=np.float64)
        self._renderer.set_skin(session.skin)
        colour, cover = self._renderer.draw(
            bgr, positions, uv, session.skin_uv, normals, drawn[order], self._light, box
        )
        region = box.crop(bgr).astype(np.float32)
        out = region * (1.0 - cover[..., None]) + colour
        owner = finger_owner(session.part)
        others = [h for h in self._hands if h.handedness != owner]
        if others:
            on_top = feathered(hand_mask(others, self._size, box))[..., None]
            out = out * (1.0 - on_top) + region * on_top
        result = bgr.copy()
        result[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return result


def _border(patch: SkinPatch) -> np.ndarray:
    rows, cols = patch.shape
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:BORDER_ROWS, :] = grid[-BORDER_ROWS:, :] = True
    grid[:, :BORDER_ROWS] = grid[:, -BORDER_ROWS:] = True
    return grid.ravel()


def _still_holding(grab: GrabState | None, hold: Hold) -> bool:
    return grab is not None and grab.active and grab.target == hold.target


def _crosses(mouth: np.ndarray, rest: np.ndarray, edges: np.ndarray, window: Window) -> np.ndarray:
    """Springs that cross the mouth opening (any of three points along them)."""
    crossing = np.zeros(len(edges), dtype=bool)
    a, b = rest[edges[:, 0]], rest[edges[:, 1]]
    for t in (0.25, 0.5, 0.75):
        crossing |= _sample(mouth, a + (b - a) * t, window) > 0
    return crossing


def _warm_up() -> None:
    """Compile the physics now, not on the first pinch (a visible hitch)."""
    patch = build_patch(np.zeros(2), 3.0, 1.0)
    rest = np.zeros((len(patch.local), 3))
    rest[:, :2] = patch.local
    body = SoftBody(rest, patch.edges, np.zeros(len(rest), dtype=bool))
    body.step(rest, np.array([0], dtype=np.int64), rest[[0]] + 1.0)
