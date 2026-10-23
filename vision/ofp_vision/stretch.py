"""The skin-physics stretch effect, drawn on the graphics card.

Ties the pieces together, frame by frame:

    grab starts  on the face, a skin mesh is laid over the pinched area
                 (`skinmesh.py`) and given physics (`softbody.py`): pinned
                 at the eyes and at its border, cut along the mouth opening
                 (the lips are only joined at the corners), and made only of
                 real skin - never hair, background or teeth (`skin.py`).
                 A finger becomes a rubber tube instead (`fingers.py`)
    each frame   the mesh moves with the head (or the tube with the hand)
                 exactly; the fingers hold the pinched points; physics moves
                 the rest, with weight, drag and wobble. It is drawn on the graphics card,
                 painted with the camera image and lit (`gpu.py`), then
                 composited onto the frame, with the hands put back on top
    let go       physics snaps it back - past rest, a wobble, still - and
                 then the mesh is dropped

Where a part was and no longer is, what is behind it shows - the room,
remembered from earlier frames (`room.py`) - so the real part does not stay
in the picture next to its stretched self.

Each hand grabs on its own, so two stretches can run at once. Each body part
- the face, or one finger - has its own mesh; both hands pinching the same
part of the face hold the same mesh. They are drawn one over the other, the
face first.

Same interface as `warp.FaceWarper` (update, then warp), which stays as the
fallback where no graphics card can be used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from .grab import GrabState, finger_owner
from .fingers import TRIANGLES as TUBE_TRIANGLES
from .fingers import FingerSession, finger_usable
from .gpu import SkinRenderer, vertex_normals
from .indices import FACE_EYE_POINTS, FACE_OVAL
from .light import default_light, followed_light
from .room import RoomMemory, filled
from .signals import Frame, HandObservation, Point
from .skin import (
    FEATHER_PX,
    Window,
    face_masks,
    feathered,
    hand_mask,
    segmented_face_masks,
)
from .skinmesh import BodyFrame, Hold, SkinPatch, build_patch, face_frame
from .softbody import SoftBody
from .warp import CORE_BY_TARGET, CORE_PER_FACE

# The mesh over a face: this far around the pinch, points this far apart
# (both in eye-distances): ~3,700 points, ~5px apart at webcam distance.
FACE_PATCH_HALF = 0.9
FACE_SPACING = 0.03
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
# The room is never learned this close to a hand (pixels): the segmentation
# is a few frames late, and a moving hand is where it is most wrong.
HAND_MARGIN = 25
# How far past its visible skin a part's soft edge reaches (pixels, odd).
EDGE_KERNEL = np.ones((FEATHER_PX, FEATHER_PX), dtype=np.uint8)
# A spot a part has left is filled from this far around it (pixels).
FILL_CONTEXT = 32


# The mesh of the face is shared: a second pinch on it joins the mesh that
# is already moving, if it lands this far inside its edge (a fraction of the
# mesh's half size). Nearer the edge, where it is pinned, it gets its own.
JOIN_INSIDE = 0.7
FACE_PART = "face"


@dataclass
class Session:
    """The face's mesh and physics, from the first pinch on it until it has
    settled. Each hand on it has a `Hold`. (A finger is a `FingerSession`.)"""

    part: str
    patch: SkinPatch
    body: SoftBody
    # What is visible, per pixel: the skin mask over the window the mesh
    # was laid on, and where each point sat in it.
    skin: np.ndarray
    skin_uv: np.ndarray
    is_skin: np.ndarray
    frame: BodyFrame
    # The window `skin` covers, and the body frame when it was laid.
    window: Window
    start: BodyFrame
    # The part of `skin` that can leave a gap behind it (`_can_leave`).
    leaves: np.ndarray
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


@dataclass(frozen=True)
class Layer:
    """A stretch drawn over `box`: premultiplied colour, its coverage, and
    where the part's skin that can leave a gap was (0..1). `extent`, if
    known, is the part's whole outline: no gap reaches past it."""

    box: Window
    colour: np.ndarray
    cover: np.ndarray
    was: np.ndarray
    extent: np.ndarray | None = None


def _visible(skin: np.ndarray) -> np.ndarray:
    """What of a skin mask (uint8) is drawn: from a little inside its soft
    edge, where the camera's pixels are pure skin."""
    ramp = np.clip((feathered(skin) - VISIBLE_FROM) / (VISIBLE_TO - VISIBLE_FROM), 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)


def _window_around(points: np.ndarray, size: tuple[int, int], margin: int) -> Window:
    low, high = points.min(axis=0), points.max(axis=0)
    return Window(int(low[0]), int(low[1]), int(high[0]) + 1, int(high[1]) + 1).grown(margin, size)


class SkinStretcher:
    """Turns each hand's grab into the drawn stretch, with skin physics."""

    def __init__(self, renderer: SkinRenderer | None = None) -> None:
        self._renderer = renderer or SkinRenderer()
        self._sessions: list[Session | FingerSession] = []
        self._grabs: dict[str, GrabState] = {}
        self._face: list[Point] | None = None
        self._hands: list[HandObservation] = []
        self._segmentation: np.ndarray | None = None
        self._size = (0, 0)
        self._light = default_light()
        self._room = RoomMemory()
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
        self._learn_room(bgr)
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

    def _learn_room(self, bgr: np.ndarray) -> None:
        """Remember the room behind the player, from each new segmentation -
        never near the hands."""
        if not self._room.is_new(self._segmentation):
            return
        exclude = None
        if self._hands:
            whole = Window(0, 0, self._size[0], self._size[1])
            reach = 2 * HAND_MARGIN + 1
            exclude = cv2.dilate(hand_mask(self._hands, self._size, whole), np.ones((reach, reach), np.uint8))
        self._room.update(bgr, self._segmentation, exclude)

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

    def _hold(self, session: Session | FingerSession, grab: GrabState) -> Hold | None:
        if isinstance(session, FingerSession):
            return session.hold(grab.target, np.array(self._px(grab.anchor)))
        return self._face_hold(session, grab)

    def _face_hold(self, session: Session, grab: GrabState) -> Hold | None:
        """The points of the face's mesh a pinch at the grab's anchor holds:
        firm in the middle of the pinch, gentle at its rim. None if the pinch
        is not well inside the mesh, or holds no skin of it."""
        anchor = session.frame.to_local(np.array([self._px(grab.anchor)]))[0]
        low, high = session.patch.local.min(axis=0), session.patch.local.max(axis=0)
        centre, half = (low + high) / 2, (high - low) / 2
        if np.any(np.abs(anchor - centre) > JOIN_INSIDE * half):
            return None
        core_size = CORE_BY_TARGET.get(grab.target, CORE_PER_FACE)
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

    def _hand(self, side: str | None) -> HandObservation | None:
        return next((h for h in self._hands if h.handedness == side), None)

    # --- starting a stretch -------------------------------------------------

    def _start(self, bgr: np.ndarray, grab: GrabState) -> Session | FingerSession | None:
        """Lay a mesh over the face around the pinch - or a tube along the
        pinched finger."""
        part = part_of(grab.target)
        if part == FACE_PART:
            return self._start_face(bgr, grab)
        hand = self._hand(finger_owner(part))
        if hand is None or not finger_usable(hand, part, self._size):
            return None
        return FingerSession.start(part, hand, self._size)

    def _face_frame(self) -> BodyFrame | None:
        return face_frame(self._face, self._size) if self._face is not None else None

    def _start_face(self, bgr: np.ndarray, grab: GrabState) -> Session | None:
        frame = self._face_frame()
        if frame is None:
            return None
        anchor = frame.to_local(np.array([self._px(grab.anchor)]))[0]
        patch = build_patch(anchor, FACE_PATCH_HALF, FACE_SPACING)
        rest = frame.to_image(patch.local)
        window = _window_around(rest, self._size, FEATHER_PX + 2)
        skin, mouth = self._face_masks(bgr, window)
        soft = feathered(skin)
        visible = _visible(skin)
        skin_uv = (rest - [window.x0, window.y0]) / [window.shape[1], window.shape[0]]
        # Every point moves - the ones that are not skin just stay invisible:
        # the GPU shows a pixel only if where it came from was skin. Only the
        # mouth opening cuts springs.
        edges = patch.edges
        if mouth is not None:
            edges = edges[~_crosses(mouth, rest, edges, window)]
        pinned = _border(patch) | self._near_eyes(rest, frame)
        rest3 = np.zeros((len(rest), 3))
        rest3[:, :2] = rest
        body = SoftBody(rest3, edges, pinned)
        is_skin = _sample(soft, rest, window) > SKIN_MIN
        leaves = self._can_leave(patch, frame, window, visible)
        return Session(FACE_PART, patch, body, visible, skin_uv, is_skin, frame, window, frame, leaves)

    def _can_leave(self, patch: SkinPatch, frame: BodyFrame, window: Window, visible: np.ndarray) -> np.ndarray:
        """Which of the face's visible skin can leave a gap behind it, over
        `window`: within the mesh's moving rows (its pinned rim never moves),
        and only outside the face itself - an ear, say - where the room is
        behind it. Inside, a face's skin is one sheet: it never opens."""
        inner = patch.local[~_border(patch)]
        low, high = inner.min(axis=0), inner.max(axis=0)
        corners = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
        origin = np.array([window.x0, window.y0])
        inside = np.zeros(window.shape, dtype=np.uint8)
        cv2.fillConvexPoly(inside, np.round(frame.to_image(corners) - origin).astype(np.int32), 1)
        if self._face is not None:
            oval = np.array([self._px(self._face[i]) for i in FACE_OVAL]) - origin
            cv2.fillConvexPoly(inside, cv2.convexHull(np.round(oval).astype(np.int32)), 0)
        return visible * inside

    # --- each frame ---------------------------------------------------------

    def _step(self, session: Session | FingerSession) -> np.ndarray:
        """One frame of a stretch's physics: its resting places follow the
        body part, each hand holds its points. Returns the resting places."""
        if isinstance(session, FingerSession):
            rest = session.follow(self._hand(session.owner), self._size)
            session.step(rest, {hand: self._pull(self._grabs[hand]) for hand in session.holds})
            return rest
        return self._step_face(session)

    def _step_face(self, session: Session) -> np.ndarray:
        before = session.frame.to_image(session.patch.local)
        session.frame = self._face_frame() or session.frame
        rest = np.zeros((len(session.patch.local), 3))
        rest[:, :2] = session.frame.to_image(session.patch.local)
        # The skin moves with the head (or hand) exactly; physics only adds
        # the stretch. Left to catch up, it slid along behind, like a mask.
        moved = np.zeros_like(rest)
        moved[:, :2] = rest[:, :2] - before
        session.body.carry(moved)
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

    def _face_masks(self, bgr: np.ndarray, window: Window):
        """(skin, mouth) over `window`: from the segmentation model's labels
        when it has run - real edges - else from shapes and skin colour."""
        if self._segmentation is not None:
            return segmented_face_masks(self._segmentation, self._face, self._size, window)
        return face_masks(bgr, self._face, window)

    def _near_eyes(self, rest: np.ndarray, frame: BodyFrame) -> np.ndarray:
        if self._face is None:
            return np.zeros(len(rest), dtype=bool)
        eyes = np.array([[self._face[i][0] * self._size[0], self._face[i][1] * self._size[1]] for i in FACE_EYE_POINTS])
        nearest = np.min(np.linalg.norm(rest[:, None, :] - eyes[None, :, :], axis=2), axis=1)
        return nearest < EYE_PIN * frame.scale

    # --- drawing ------------------------------------------------------------

    def _draw(self, bgr: np.ndarray, session: Session | FingerSession, rest: np.ndarray) -> np.ndarray:
        if isinstance(session, FingerSession):
            layer = self._finger_layer(bgr, session, rest)
        else:
            layer = self._face_layer(bgr, session, rest)
        return bgr if layer is None else self._composite(bgr, layer, finger_owner(session.part))

    def _face_layer(self, bgr: np.ndarray, session: Session, rest: np.ndarray) -> Layer | None:
        positions = session.body.positions
        # All of the mesh is drawn; the GPU discards whatever did not come
        # from skin, pixel by pixel.
        drawn = session.patch.triangles
        order = np.argsort(positions[drawn].mean(axis=1)[:, 2], kind="stable")
        box = _window_around(positions[:, :2], self._size, 2)
        if box.empty:
            return None
        normals = vertex_normals(positions, session.patch.triangles)
        # Pinned skin (the eyes) has not moved: it keeps exactly its own light.
        normals[session.body.pinned] = (0.0, 0.0, 1.0)
        uv = rest[:, :2] / np.array(self._size, dtype=np.float64)
        self._renderer.set_skin(session.skin)
        colour, cover = self._renderer.draw(
            bgr, positions, uv, session.skin_uv, normals, drawn[order], self._light, box
        )
        return Layer(box, colour, cover, _resting_skin(session, box))

    def _finger_layer(self, bgr: np.ndarray, session: FingerSession, rest: np.ndarray) -> Layer | None:
        """The finger's tube, painted with the finger's own pixels where it
        rests - never the other hand's, which may still be over it."""
        drawn, source = session.tube(rest)
        box = _window_around(np.vstack([drawn[:, :2], source]), self._size, FEATHER_PX + 2)
        if box.empty:
            return None
        skin = session.skin(rest, self._segmentation, self._size, box, hand=self._hand(session.owner))
        others = [h for h in self._hands if h.handedness != session.owner]
        if others:
            skin[hand_mask(others, self._size, box) > 0] = 0
        visible = _visible(skin)
        self._renderer.set_skin(visible)
        order = np.argsort(drawn[TUBE_TRIANGLES][:, :, 2].mean(axis=1), kind="stable")
        flat = np.tile((0.0, 0.0, 1.0), (len(drawn), 1))
        skin_uv = (source - [box.x0, box.y0]) / [box.shape[1], box.shape[0]]
        uv = source / np.array(self._size, dtype=np.float64)
        colour, cover = self._renderer.draw(
            bgr, drawn, uv, skin_uv, flat, TUBE_TRIANGLES[order], self._light, box
        )
        moving = session.skin(rest, None, self._size, box, moving_only=True) > 0
        # Its outline, a pixel wider: the segmentation's edge is never exact,
        # and a rim of finger left behind reads as a ghost outline.
        extent = cv2.dilate(((skin > 64) & moving).astype(np.float32), np.ones((3, 3), np.uint8))
        return Layer(box, colour, cover, visible * moving, extent)

    def _composite(self, bgr: np.ndarray, layer: Layer, owner: str | None) -> np.ndarray:
        """The stretch over the frame: what it left behind filled in, and
        the other hands put back on top."""
        box, cover = layer.box, layer.cover
        region = box.crop(bgr).astype(np.float32)
        others = [h for h in self._hands if h.handedness != owner]
        on_top = feathered(hand_mask(others, self._size, box)) if others else np.zeros(box.shape, np.float32)
        # `was` is the part's visible skin, which starts inside its soft edge:
        # widened by that edge, the edge leaves with it and is not sampled
        # for the fill. No further: the room beside it stays exactly as it is.
        left = cv2.dilate(np.clip(layer.was - cover, 0.0, 1.0), EDGE_KERNEL)
        if layer.extent is not None:
            left = left * layer.extent
        body = cv2.dilate(np.maximum(layer.was, on_top), EDGE_KERNEL)
        under = self._behind_left(region, left, body, box)
        out = under * (1.0 - cover[..., None]) + layer.colour
        if others:
            out = out * (1.0 - on_top[..., None]) + region * on_top[..., None]
        result = bgr.copy()
        result[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return result

    def _behind_left(self, region: np.ndarray, left: np.ndarray, body: np.ndarray, box: Window) -> np.ndarray:
        """`region`, except where the part has `left` its place: there, what
        is behind it - the remembered room where it is known, else a fill
        from the colours around the `body` (the part and the hands)."""
        x, y, w, h = cv2.boundingRect((left > 0.01).astype(np.uint8))
        if w == 0 or h == 0:
            return region
        spot = Window(x, y, x + w, y + h).grown(FILL_CONTEXT, (region.shape[1], region.shape[0]))
        room, known = self._room.fill(Window(box.x0 + spot.x0, box.y0 + spot.y0, box.x0 + spot.x1, box.y0 + spot.y1))
        under = spot.crop(region)
        behind = filled(under, spot.crop(body))
        behind = behind + (room - behind) * known[..., None]
        out = region.copy()
        out[spot.y0 : spot.y1, spot.x0 : spot.x1] = under + (behind - under) * spot.crop(left)[..., None]
        return out


def _resting_skin(session: Session, box: Window) -> np.ndarray:
    """Where the part's skin that can leave a gap rests now, over `box`: its
    mask, moved, turned and scaled along with the body part."""
    to_box = (
        _shift(-box.x0, -box.y0)
        @ _affine(session.frame)
        @ np.linalg.inv(_affine(session.start))
        @ _shift(session.window.x0, session.window.y0)
    )
    size = (box.x1 - box.x0, box.y1 - box.y0)
    return cv2.warpAffine(session.leaves, to_box[:2], size, flags=cv2.INTER_LINEAR, borderValue=0.0)


def _affine(frame: BodyFrame) -> np.ndarray:
    """A body frame's local -> image map, as a 3x3 matrix."""
    s = frame.scale
    return np.array(
        [
            [s * frame.x_axis[0], s * frame.y_axis[0], frame.origin[0]],
            [s * frame.x_axis[1], s * frame.y_axis[1], frame.origin[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


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
