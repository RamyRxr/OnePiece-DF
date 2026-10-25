"""The skin-physics stretch effect, drawn on the graphics card.

Ties the pieces together, frame by frame:

    grab starts  on the face, a skin mesh is laid over the pinched area and
                 glued to the face's own landmarks (`faceskin.py`); on a
                 finger, a rubber chain along it, which warps the picture of
                 the finger (`fingers.py`). Both get physics (`softbody.py`)
    each frame   the mesh goes with the face (or the tube with the hand)
                 exactly; the fingers hold the pinched points; physics moves
                 the rest, with weight, drag and wobble. What of the face
                 shows is worked out from this very frame, so nothing of the
                 moment of the pinch lingers once you move. It is drawn on
                 the graphics card, painted with the camera image and lit
                 (`gpu.py`), then composited onto the frame, with the hands
                 put back on top
    let go       physics snaps it back - past rest, a wobble, still - and
                 then the mesh is dropped

Where a part was and no longer is, what is behind it shows - remembered from
earlier frames (`room.py`) - so the real part does not stay in the picture
next to its stretched self.

Each hand grabs on its own, so two stretches can run at once. Each body part
- the face, or one finger - has its own mesh; both hands pinching the same
part of the face hold the same mesh. They are drawn one over the other, the
face first.

Same interface as `warp.FaceWarper` (update, then warp), which stays as the
fallback where no graphics card can be used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from .faceskin import FACE_PART, FaceSession, face_visible
from .fingers import FingerSession, finger_usable
from .gpu import SkinRenderer, vertex_normals
from .grab import GrabState, finger_owner
from .light import default_light, followed_light
from .room import ROOM_SCALE, RoomMemory, filled, grain
from .signals import Frame, HandObservation, Point
from .skin import FEATHER_PX, Window, feathered, hand_mask
from .skinmesh import Hold, build_patch
from .softbody import SoftBody

# Pulled skin comes off the face toward the camera, this much per pixel of
# pull (until the real depth of the hand is measured).
LIFT_PER_PULL = 0.3
# What is behind the hands is never learned this close to one (pixels): a
# hand moves fast, and its landmarks trail it by a frame or two.
HAND_MARGIN = 25
# A hand the tracker loses is still there for this many frames, as far as
# learning what is behind it goes: pinching, one hand often hides the other.
HAND_MEMORY = 15
# How far past its visible skin a part's soft edge reaches (pixels, odd).
EDGE_KERNEL = np.ones((FEATHER_PX, FEATHER_PX), dtype=np.uint8)
# A spot a part has left is filled from this far around it (pixels).
FILL_CONTEXT = 32
# The camera's grain for filled spots, cut from a tile of it: fresh noise
# for a whole spot each frame costs more than the rest of the fill.
GRAIN_TILE = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
# A skin texture of one pixel, all skin: a warp shows every pixel it moves.
OPAQUE = np.ones((1, 1), dtype=np.float32)

Stretch = FaceSession | FingerSession


def part_of(target: str) -> str:
    """The body part a target is on: the face, or that finger."""
    return target if finger_owner(target) is not None else FACE_PART


@dataclass(frozen=True, eq=False)
class Layer:
    """A stretch drawn over `box`: premultiplied colour, its coverage, and
    where the part's skin that can leave a gap was (0..1). `extent`, if
    known, is the part's whole outline: no gap reaches past it."""

    box: Window
    colour: np.ndarray
    cover: np.ndarray
    was: np.ndarray
    extent: np.ndarray | None = None


def _window_around(points: np.ndarray, size: tuple[int, int], margin: int) -> Window:
    low, high = points.min(axis=0), points.max(axis=0)
    return Window(int(low[0]), int(low[1]), int(high[0]) + 1, int(high[1]) + 1).grown(margin, size)


class SkinStretcher:
    """Turns each hand's grab into the drawn stretch, with skin physics."""

    def __init__(self, renderer: SkinRenderer | None = None) -> None:
        self._renderer = renderer or SkinRenderer()
        self._sessions: list[Stretch] = []
        self._grabs: dict[str, GrabState] = {}
        self._face: list[Point] | None = None
        self._hands: list[HandObservation] = []
        # Each hand last seen, and how many frames ago.
        self._recent: dict[str, tuple[HandObservation, int]] = {}
        self._segmentation: np.ndarray | None = None
        self._size = (0, 0)
        self._light = default_light()
        self._room = RoomMemory()
        self._rng = np.random.default_rng()
        _warm_up()

    def update(self, grabs: Mapping[str, GrabState], frame: Frame, size: tuple[int, int]) -> None:
        """Take this frame's grabs (one per hand) and landmarks. `size` is
        (width, height)."""
        self._grabs = dict(grabs)
        self._size = size
        if frame.face is not None:
            self._face = frame.face
        self._hands = list(frame.hands)
        older = {side: (hand, ago + 1) for side, (hand, ago) in self._recent.items() if ago + 1 <= HAND_MEMORY}
        self._recent = {**older, **{hand.handedness: (hand, 0) for hand in frame.hands}}
        if frame.segmentation is not None:
            self._segmentation = frame.segmentation

    def warp(self, bgr: np.ndarray) -> np.ndarray:
        """Return a copy of `bgr` with the stretches drawn (or `bgr` if none)."""
        self._learn_room(bgr)
        self._follow_grabs()
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
        """Remember the picture behind the hands: all of it but the hands -
        those seen lately too (one the tracker loses for a moment is still
        there) - and anything being stretched, which is still in the camera's
        picture where it rests. Drawn at the memory's own, smaller size."""
        small = (max(1, round(self._size[0] * ROOM_SCALE)), max(1, round(self._size[1] * ROOM_SCALE)))
        hands = [hand for hand, _ in self._recent.values()]
        exclude = hand_mask(hands, small, Window(0, 0, small[0], small[1])) if hands else None
        if exclude is None:
            exclude = np.zeros((small[1], small[0]), dtype=np.uint8)
        for session in self._sessions:
            session.mark_resting(exclude, ROOM_SCALE)
        reach = 2 * round(HAND_MARGIN * ROOM_SCALE) + 1
        self._room.update(bgr, cv2.dilate(exclude, np.ones((reach, reach), np.uint8)))

    # --- which hand holds what ----------------------------------------------

    def _follow_grabs(self) -> None:
        """Let go of what each hand released; take hold of what it pinched."""
        for session in self._sessions:
            session.holds = {
                hand: hold for hand, hold in session.holds.items() if _still_holding(self._grabs.get(hand), hold)
            }
        holding = {hand for session in self._sessions for hand in session.holds}
        for hand, grab in self._grabs.items():
            if grab.active and grab.anchor is not None and hand not in holding:
                self._take_hold(hand, grab)

    def _take_hold(self, hand: str, grab: GrabState) -> None:
        """Join a mesh already on that body part, or lay a new one."""
        part = part_of(grab.target)
        anchor = np.array(self._px(grab.anchor))
        for session in self._sessions:
            hold = session.hold(grab.target, anchor) if session.part == part else None
            if hold is not None:
                session.holds[hand] = hold
                return
        session = self._start(part, anchor)
        hold = session.hold(grab.target, anchor) if session is not None else None
        if session is not None and hold is not None:
            session.holds[hand] = hold
            self._sessions.append(session)

    def _start(self, part: str, anchor: np.ndarray) -> Stretch | None:
        """Lay a mesh over the face around the pinch - or a tube along the
        pinched finger."""
        if part == FACE_PART:
            return FaceSession.start(self._face, self._size, anchor) if self._face is not None else None
        hand = self._hand(finger_owner(part))
        if hand is None or not finger_usable(hand, part, self._size):
            return None
        return FingerSession.start(part, hand, self._size)

    def _px(self, point) -> tuple[float, float]:
        return (point[0] * self._size[0], point[1] * self._size[1])

    def _hand(self, side: str | None) -> HandObservation | None:
        return next((h for h in self._hands if h.handedness == side), None)

    # --- each frame ---------------------------------------------------------

    def _step(self, session: Stretch) -> np.ndarray:
        """One frame of a stretch's physics: its resting places follow the
        body part, each hand holds its points. Returns the resting places."""
        if isinstance(session, FingerSession):
            rest = session.follow(self._hand(session.owner), self._size)
        else:
            rest = session.follow(self._face, self._size)
        session.step(rest, {hand: self._pull(self._grabs[hand]) for hand in session.holds})
        return rest

    def _pull(self, grab: GrabState) -> np.ndarray:
        """A grab's pull in pixels - and off the skin, toward the camera."""
        pull = np.array([grab.pull[0] * self._size[0], grab.pull[1] * self._size[1], 0.0])
        pull[2] = LIFT_PER_PULL * float(np.hypot(pull[0], pull[1]))
        return pull

    # --- drawing ------------------------------------------------------------

    def _draw(self, bgr: np.ndarray, session: Stretch, rest: np.ndarray) -> np.ndarray:
        if isinstance(session, FingerSession):
            layer = self._finger_layer(bgr, session, rest)
        else:
            layer = self._face_layer(bgr, session, rest)
        return bgr if layer is None else self._composite(bgr, layer, finger_owner(session.part))

    def _face_layer(self, bgr: np.ndarray, session: FaceSession, rest: np.ndarray) -> Layer | None:
        """The face's mesh, painted with the face as the camera sees it now -
        and what of it shows worked out from this frame, too."""
        positions = session.body.positions
        box = _window_around(np.vstack([positions[:, :2], rest[:, :2]]), self._size, FEATHER_PX + 2)
        if box.empty or self._face is None:
            return None
        visible = face_visible(self._face, self._segmentation, bgr, self._size, box)
        # All of the mesh is drawn; the GPU discards whatever did not come
        # from skin, pixel by pixel.
        drawn = session.patch.triangles
        order = np.argsort(positions[drawn].mean(axis=1)[:, 2], kind="stable")
        normals = vertex_normals(positions, drawn)
        # Pinned skin (the eyes) has not moved: it keeps exactly its own light.
        normals[session.body.pinned] = (0.0, 0.0, 1.0)
        uv = rest[:, :2] / np.array(self._size, dtype=np.float64)
        skin_uv = (rest[:, :2] - [box.x0, box.y0]) / [box.shape[1], box.shape[0]]
        self._renderer.set_skin(visible)
        colour, cover = self._renderer.draw(bgr, positions, uv, skin_uv, normals, drawn[order], self._light, box)
        return Layer(box, colour, cover, session.leaves(self._face, self._size, box, visible))

    def _finger_layer(self, bgr: np.ndarray, session: FingerSession, rest: np.ndarray) -> Layer | None:
        """The picture along the finger, warped with it: the finger's own
        pixels stretch, and a thin skirt around it blends the stretch into
        the picture. Nothing is erased; nothing new is drawn."""
        warp = session.warp(rest, self._hand(session.owner), self._size)
        if warp is None:
            return None
        box = _window_around(np.vstack([warp.positions[:, :2], warp.source]), self._size, 2)
        if box.empty:
            return None
        order = np.argsort(warp.positions[warp.triangles, 2].mean(axis=1), kind="stable")
        flat = np.tile((0.0, 0.0, 1.0), (len(warp.positions), 1))
        uv = warp.source / np.array(self._size, dtype=np.float64)
        self._renderer.set_skin(OPAQUE)
        colour, cover = self._renderer.draw(
            bgr, warp.positions, uv, np.full_like(uv, 0.5), flat, warp.triangles[order], self._light, box
        )
        return Layer(box, colour, cover, np.zeros(box.shape, dtype=np.float32))

    def _composite(self, bgr: np.ndarray, layer: Layer, owner: str | None) -> np.ndarray:
        """The stretch over the frame: what it left behind filled in, and
        the other hands put back on top."""
        box, cover = layer.box, layer.cover
        region = box.crop(bgr).astype(np.float32)
        others = [h for h in self._hands if h.handedness != owner]
        mine = [h for h in self._hands if h.handedness == owner]
        others_mask = hand_mask(others, self._size, box) if others else np.zeros(box.shape, np.uint8)
        on_top = feathered(others_mask) if others else np.zeros(box.shape, np.float32)
        # `was` is the part's visible skin, which starts inside its soft edge:
        # widened by that edge, the edge leaves with it and is not sampled
        # for the fill. No further: the room beside it stays exactly as it is.
        left = cv2.dilate(np.clip(layer.was - cover, 0.0, 1.0), EDGE_KERNEL)
        # A soft edge: the filled spot blends into what is around it - but
        # never past the part's own outline, into the room beside it.
        left = cv2.GaussianBlur(left, (5, 5), 0)
        if layer.extent is not None:
            left = left * layer.extent
        # The fill never samples a hand - not even the one the part is on: its
        # other fingers' skin would tint it.
        hands = np.maximum(others_mask, hand_mask(mine, self._size, box)) if mine else others_mask
        body = cv2.dilate(np.maximum(layer.was, hands.astype(np.float32) / 255.0), EDGE_KERNEL)
        under = self._behind_left(region, left, body, box)
        out = under * (1.0 - cover[..., None]) + layer.colour
        if others:
            out = out * (1.0 - on_top[..., None]) + region * on_top[..., None]
        result = bgr.copy()
        result[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return result

    def _grain(self, shape: tuple[int, int]) -> np.ndarray:
        """Unit grain over `shape`, cut from the tile at a random place."""
        size = GRAIN_TILE.shape[0]
        y, x = self._rng.integers(0, size, 2)
        return GRAIN_TILE[np.ix_((y + np.arange(shape[0])) % size, (x + np.arange(shape[1])) % size)]

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
        hole = spot.crop(body)
        behind = filled(under, hole)
        behind = behind + (room - behind) * known[..., None]
        # The camera's grain: a smooth fill reads as a flat cut-out.
        level = grain(under, hole < 0.05)
        behind = behind + level * self._grain(behind.shape[:2])[..., None]
        out = region.copy()
        out[spot.y0 : spot.y1, spot.x0 : spot.x1] = under + (behind - under) * spot.crop(left)[..., None]
        return out


def _still_holding(grab: GrabState | None, hold: Hold) -> bool:
    return grab is not None and grab.active and grab.target == hold.target


def _warm_up() -> None:
    """Compile the physics now, not on the first pinch (a visible hitch)."""
    patch = build_patch(np.zeros(2), 3.0, 1.0)
    rest = np.zeros((len(patch.local), 3))
    rest[:, :2] = patch.local
    body = SoftBody(rest, patch.edges, np.zeros(len(rest), dtype=bool))
    body.step(rest, np.array([0], dtype=np.int64), rest[[0]] + 1.0)
