"""Pinch-grab-pull: the main interaction.

Measuring how a face deforms under a tug, through a 2D webcam, is fragile.
Measuring what the player's *hand* is doing is not: MediaPipe tracks hands
well, and a pinch is unambiguous. So the gesture is

    pinch   thumb tip meets index tip, right on a target
    pull    move the pinched hand away; the stretch is the distance moved
    release open the pinch (or pull so far it snaps) -> the "fire" event

Targets are the parts that stretch: ears, nose, jaw, cheeks, mouth - and any
finger of the *other* hand, grabbed where the pinch lands on it (knuckle to
tip), only if it lands on the finger itself and not in the air beside it.

Each hand grabs on its own (`TwoHandGrabber`): one hand, or both at once.

This is the single source of truth for the grab. The lab's video warp reads
the state from here instead of running a pinch detector of its own.

Units
-----
Every distance is measured in pixel-proportional units (x is scaled by the
frame's aspect ratio, so a circle stays a circle) and divided by a body
length: pinch gaps by palm length, face targets by interocular distance,
finger targets by the grabbed hand's palm length. The stretch therefore means
the same thing at any distance from the camera.

The anchor follows the body
---------------------------
The grabbed point is re-read from the landmarks every frame, plus the offset
between the pinch and the target at the moment of the grab. Moving the head
(or the grabbed hand) along with the pinching hand is not a stretch; only
the pinch moving *relative to* the grabbed part is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .config import GRAB_HANDS, GrabConfig
from .filters import OneEuroFilter
from .indices import (
    FACE_CHEEK_L,
    FACE_CHEEK_R,
    FACE_CHIN,
    FACE_EAR_LEVEL_L,
    FACE_EAR_LEVEL_R,
    FACE_EYE_OUTER_L,
    FACE_EYE_OUTER_R,
    FACE_JAW_L,
    FACE_JAW_R,
    FACE_LIP_BOTTOM_OUTER,
    FACE_LIP_TOP_OUTER,
    FACE_MOUTH_L,
    FACE_MOUTH_R,
    FACE_NOSE_TIP,
    HAND_DIGIT_ORDER,
    HAND_DIGITS,
    HAND_FINGER_JOINTS,
    HAND_MIDDLE_MCP,
    HAND_WRIST,
)
from .signals import Frame, HandObservation, Point

HAND_THUMB_TIP = HAND_DIGITS["thumb"][3]
HAND_INDEX_TIP = HAND_DIGITS["index"][3]

Point2 = tuple[float, float]

# Face targets read straight off a landmark: name -> landmark.
FACE_LANDMARK_TARGETS: dict[str, int] = {
    "cheekL": FACE_CHEEK_L,
    "cheekR": FACE_CHEEK_R,
    "nose": FACE_NOSE_TIP,
    "chin": FACE_CHIN,
    "jawL": FACE_JAW_L,
    "jawR": FACE_JAW_R,
    "mouthL": FACE_MOUTH_L,
    "mouthR": FACE_MOUTH_R,
    "lipTop": FACE_LIP_TOP_OUTER,
    "lipBottom": FACE_LIP_BOTTOM_OUTER,
}

# MediaPipe has no ear landmarks. The ear sits just outside the face mesh,
# beyond the silhouette point level with its middle: this far again along
# the line from the nose tip out to that point. (Checked against a real
# portrait: it lands on the ear.)
EAR_OUTSET = 0.18
EAR_SILHOUETTE: dict[str, int] = {"earL": FACE_EAR_LEVEL_L, "earR": FACE_EAR_LEVEL_R}

FINGER_PREFIX = "finger."

# Codes on the wire (`grab.<hand>.target`). 0 means nothing is grabbed.
TARGET_CODES: dict[str, int] = {
    "cheekL": 1,
    "cheekR": 2,
    "earL": 3,
    "earR": 4,
    "nose": 5,
    "chin": 6,
    "jawL": 7,
    "jawR": 8,
    "mouthL": 9,
    "mouthR": 10,
    "lipTop": 11,
    "lipBottom": 12,
    # 20-24 the left hand's thumb..pinky, 25-29 the right hand's.
    **{
        f"{FINGER_PREFIX}{side}.{digit}": base + offset
        for side, base in (("Left", 20), ("Right", 25))
        for offset, digit in enumerate(HAND_DIGIT_ORDER)
    },
}

NO_TARGET = "none"


def ear_points(face: list[Point]) -> dict[str, Point2]:
    """Where the ears are: outside the mesh, beyond the silhouette points."""
    nose = face[FACE_NOSE_TIP]
    points = {}
    for name, index in EAR_SILHOUETTE.items():
        edge = face[index]
        points[name] = (
            edge[0] + EAR_OUTSET * (edge[0] - nose[0]),
            edge[1] + EAR_OUTSET * (edge[1] - nose[1]),
        )
    return points


def face_target_points(face: list[Point]) -> dict[str, Point2]:
    """Every grabbable point on the face, by target name."""
    points = {name: (face[i][0], face[i][1]) for name, i in FACE_LANDMARK_TARGETS.items()}
    return {**points, **ear_points(face)}


def pinch_point(hand: HandObservation) -> Point2:
    """Where a hand pinches: midway between its thumb and index tips."""
    thumb, index = hand.landmarks[HAND_THUMB_TIP], hand.landmarks[HAND_INDEX_TIP]
    return ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2)


def followed_grab(grabs: Mapping[str, GrabState], hand: str | None) -> tuple[str | None, GrabState]:
    """For an effect that draws one grab: keep to `hand` while it holds (or
    is letting go); else switch to whichever hand holds one now."""
    current = grabs.get(hand) if hand is not None else None
    if current is not None and (current.active or current.just_released):
        return hand, current
    for other, grab in grabs.items():
        if grab.active:
            return other, grab
    return hand, current or GrabState()


def finger_owner(target: str) -> str | None:
    """The hand a finger target belongs to ("Left"/"Right"), else None."""
    if not target.startswith(FINGER_PREFIX):
        return None
    return target.split(".")[1]


@dataclass(frozen=True)
class Target:
    """A grabbable point this frame, the length its stretch is measured in,
    and how close (in that length) a pinch must be to grab it. For a finger,
    `place` is where along it: 0 at the knuckle, one more per joint, to the tip."""

    name: str
    point: Point2
    scale: float
    radius: float
    place: float = 0.0


@dataclass(frozen=True)
class GrabState:
    """One frame of grab state. Immutable: every update returns a new one.

    `anchor` and `pull` are in raw (unmirrored) normalised image coordinates,
    for drawing. The `stretch*` and `dir*` fields are what goes on the wire.
    """

    active: bool = False
    hand: str = ""
    target: str = NO_TARGET
    anchor: Point2 | None = None
    pull: Point2 = (0.0, 0.0)
    stretch: float = 0.0
    stretch_x: float = 0.0
    stretch_y: float = 0.0
    dir_x: float = 0.0
    dir_y: float = 0.0
    peak: float = 0.0
    just_grabbed: bool = False
    # The grab ended with an unpinch or a snap - the game should fire.
    # A grab cancelled by lost tracking does not set this.
    just_released: bool = False
    release_peak: float = 0.0

    def features(self, prefix: str = "grab") -> dict[str, float]:
        """This grab's channels (`prefix.active`, ...), for the channel manager.

        On the release frame `.target` still names what was let go, so the
        game knows which body part to snap back.
        """
        return {
            f"{prefix}.active": 1.0 if self.active else 0.0,
            f"{prefix}.stretch": self.stretch,
            f"{prefix}.stretchX": self.stretch_x,
            f"{prefix}.stretchY": self.stretch_y,
            f"{prefix}.dirX": self.dir_x,
            f"{prefix}.dirY": self.dir_y,
            f"{prefix}.target": float(TARGET_CODES.get(self.target, 0)),
        }


@dataclass
class _HandPinch:
    """Pinch bookkeeping for one hand, kept across frames for hysteresis."""

    closed: bool = False
    closed_at_ms: float = 0.0


@dataclass
class GrabDetector:
    """Turns frames into a stream of `GrabState`s."""

    config: GrabConfig = field(default_factory=GrabConfig)
    # The lab mirrors the preview. With mirror on, +x on the wire is the
    # player's right, which is what a mirrored avatar wants.
    mirror: bool = True
    # Frame width / height. Normalised landmarks squash x by this much.
    aspect: float = 4 / 3
    # Only this hand's pinch can grab (one detector per hand). None: any.
    only: str | None = None

    def __post_init__(self) -> None:
        self.state = GrabState()
        self._pinches: dict[str, _HandPinch] = {}
        self._offset: Point2 = (0.0, 0.0)
        self._place = 0.0
        self._last_pinch: Point2 = (0.0, 0.0)
        self._last_seen_ms = 0.0
        self._filter_x = self._new_filter()
        self._filter_y = self._new_filter()

    # --- geometry -----------------------------------------------------------

    def _dist(self, a: Point | Point2, b: Point | Point2) -> float:
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])

    @property
    def held_pinch(self) -> Point2 | None:
        """Where the pinch holding the grab was last seen; None if no grab."""
        return self._last_pinch if self.state.active else None

    def _face_scale(self, face: list[Point] | None) -> float | None:
        if face is None or len(face) <= max(FACE_EYE_OUTER_L, FACE_EYE_OUTER_R):
            return None
        scale = self._dist(face[FACE_EYE_OUTER_R], face[FACE_EYE_OUTER_L])
        return scale if scale > 1e-6 else None

    def _palm(self, hand: HandObservation) -> float:
        return self._dist(hand.landmarks[HAND_WRIST], hand.landmarks[HAND_MIDDLE_MCP])

    def _new_filter(self) -> OneEuroFilter:
        return OneEuroFilter(
            min_cutoff=self.config.smooth_min_cutoff, beta=self.config.smooth_beta
        )

    # --- targets ------------------------------------------------------------

    def _face_targets(self, face: list[Point] | None) -> list[Target]:
        scale = self._face_scale(face)
        if scale is None or face is None:
            return []
        radius = self.config.grab_radius
        return [Target(name, point, scale, radius) for name, point in face_target_points(face).items()]

    def _finger_targets(self, hands: list[HandObservation], exclude: str, near: Point2) -> list[Target]:
        """Every finger of every hand but `exclude` (the one pinching), each at
        its point nearest to `near` - a finger is grabbed where the pinch
        lands on it."""
        targets = []
        for hand in hands:
            palm = self._palm(hand)
            if hand.handedness == exclude or palm < 1e-6:
                continue
            for digit in HAND_DIGIT_ORDER:
                place = self._nearest_place(hand, digit, near)
                name = f"{FINGER_PREFIX}{hand.handedness}.{digit}"
                point = self._at_place(hand, digit, place)
                targets.append(Target(name, point, palm, self.config.finger_grab_radius, place))
        return targets

    def _at_place(self, hand: HandObservation, digit: str, place: float) -> Point2:
        """The point `place` along a finger: 0 at the knuckle, 1 a joint
        further, and so on to the tip."""
        joints = HAND_FINGER_JOINTS[digit]
        segment = min(int(place), len(joints) - 2)
        t = place - segment
        a, b = hand.landmarks[joints[segment]], hand.landmarks[joints[segment + 1]]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def _nearest_place(self, hand: HandObservation, digit: str, near: Point2) -> float:
        """Where along a finger (0 at the knuckle) is nearest to `near`."""
        joints = HAND_FINGER_JOINTS[digit]
        best, best_place = math.inf, 0.0
        for segment in range(len(joints) - 1):
            a, b = hand.landmarks[joints[segment]], hand.landmarks[joints[segment + 1]]
            ax, ay = a[0] * self.aspect, a[1]
            bx, by = b[0] * self.aspect, b[1]
            px, py = near[0] * self.aspect, near[1]
            length2 = (bx - ax) ** 2 + (by - ay) ** 2
            along = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
            t = 0.0 if length2 < 1e-12 else min(1.0, max(0.0, along / length2))
            distance = math.hypot(px - (ax + (bx - ax) * t), py - (ay + (by - ay) * t))
            if distance < best:
                best, best_place = distance, segment + t
        return best_place

    def _targets(self, frame: Frame, pinching_hand: str, near: Point2) -> list[Target]:
        return self._face_targets(frame.face) + self._finger_targets(frame.hands, pinching_hand, near)

    def _locate(self, frame: Frame, name: str, pinching_hand: str) -> Target | None:
        """Where the grabbed target is this frame, if it is still in view. A
        finger keeps the same place along it that was grabbed."""
        owner = finger_owner(name)
        if owner is None:
            return next((t for t in self._face_targets(frame.face) if t.name == name), None)
        hand = next((h for h in frame.hands if h.handedness == owner and owner != pinching_hand), None)
        if hand is None or self._palm(hand) < 1e-6:
            return None
        digit = name.split(".")[2]
        return Target(name, self._at_place(hand, digit, self._place), self._palm(hand),
                      self.config.finger_grab_radius, self._place)

    # --- pinch --------------------------------------------------------------

    def _update_pinches(self, hands: list[HandObservation], t_ms: float) -> dict[str, Point2]:
        """Pinch points of the hands that are pinched right now."""
        points: dict[str, Point2] = {}
        for hand in hands:
            lm = hand.landmarks
            palm = self._palm(hand)
            if palm < 1e-6:
                continue
            gap = self._dist(lm[HAND_THUMB_TIP], lm[HAND_INDEX_TIP]) / palm
            previous = self._pinches.get(hand.handedness, _HandPinch())
            limit = self.config.pinch_off if previous.closed else self.config.pinch_on
            closed = gap < limit
            closed_at = previous.closed_at_ms if previous.closed else t_ms
            self._pinches = {**self._pinches, hand.handedness: _HandPinch(closed, closed_at)}
            if closed:
                points[hand.handedness] = pinch_point(hand)
        return points

    # --- per frame ----------------------------------------------------------

    def update(self, frame: Frame, t_ms: float) -> GrabState:
        pinches = self._update_pinches(frame.hands, t_ms)
        if self.only is not None:
            pinches = {hand: point for hand, point in pinches.items() if hand == self.only}
        if self.state.active:
            self.state = self._hold(frame, pinches, t_ms)
        else:
            self.state = self._try_grab(frame, pinches, t_ms)
        return self.state

    def _try_grab(self, frame: Frame, pinches: dict[str, Point2], t_ms: float) -> GrabState:
        best: tuple[float, str, Target, Point2] | None = None
        for hand, point in pinches.items():
            if t_ms - self._pinches[hand].closed_at_ms > self.config.grab_window_ms:
                continue
            for target in self._targets(frame, hand, point):
                distance = self._dist(point, target.point) / target.scale
                if distance < target.radius and (best is None or distance < best[0]):
                    best = (distance, hand, target, point)
        if best is None:
            return GrabState()

        _, hand, target, point = best
        self._offset = (point[0] - target.point[0], point[1] - target.point[1])
        self._place = target.place
        self._last_seen_ms = t_ms
        self._last_pinch = point
        self._filter_x = self._new_filter()
        self._filter_y = self._new_filter()
        return GrabState(active=True, hand=hand, target=target.name, anchor=point, just_grabbed=True)

    def _pinching_hand(self, frame: Frame, pinches: dict[str, Point2]) -> str | None:
        """Which hand label carries the grab this frame, if any is in view.

        MediaPipe occasionally relabels a hand mid-gesture. A pinch closed
        where the grab's pinch just was is the same pinch, whatever its label.
        """
        if any(h.handedness == self.state.hand for h in frame.hands):
            return self.state.hand
        if self.only is not None:
            # One detector per hand: `TwoHandGrabber` has already given the
            # relabelled hand back its label.
            return None
        owner = finger_owner(self.state.target)
        for hand, point in pinches.items():
            if hand == owner:
                continue
            if self._dist(point, self._last_pinch) < self.config.relabel_distance:
                return hand
        return None

    def _hold(self, frame: Frame, pinches: dict[str, Point2], t_ms: float) -> GrabState:
        state = self.state
        hand = self._pinching_hand(frame, pinches)
        target = self._locate(frame, state.target, hand) if hand is not None else None
        if target is None:
            # Tracking dropout. Ride out a short one; cancel a long one.
            if t_ms - self._last_seen_ms > self.config.lost_grace_ms:
                return GrabState()
            return replace(state, just_grabbed=False)

        point = pinches.get(hand)
        if point is None:
            return self._release(state)
        self._last_seen_ms = t_ms
        self._last_pinch = point

        anchor = (target.point[0] + self._offset[0], target.point[1] + self._offset[1])
        # Filtered in body units, the units the beta is tuned for.
        t_s = t_ms / 1000.0
        sx = self._filter_x((point[0] - anchor[0]) * self.aspect / target.scale, t_s)
        sy = self._filter_y((point[1] - anchor[1]) / target.scale, t_s)
        pull = (sx * target.scale / self.aspect, sy * target.scale)
        stretch = math.hypot(sx, sy)
        peak = max(state.peak, stretch)
        if stretch >= self.config.snap_distance:
            return self._release(replace(state, peak=peak))

        if self.mirror:
            sx = -sx
        moving = stretch > 1e-9
        return replace(
            state,
            hand=hand,
            anchor=anchor,
            pull=pull,
            stretch=stretch,
            stretch_x=sx,
            stretch_y=sy,
            dir_x=sx / stretch if moving else 0.0,
            dir_y=sy / stretch if moving else 0.0,
            peak=peak,
            just_grabbed=False,
        )

    @staticmethod
    def _release(state: GrabState) -> GrabState:
        """End the grab and fire. Keeps the target for the release frame."""
        return GrabState(
            hand=state.hand,
            target=state.target,
            just_released=True,
            release_peak=state.peak,
        )


@dataclass
class TwoHandGrabber:
    """One grab per pinching hand: either hand, or both at once.

    It also steadies the hand labels, which MediaPipe gets wrong now and
    then: it swaps a hand's label mid-gesture, or calls both hands the same.
    A hand holding a grab keeps its label while its pinch stays where it was,
    and two hands never share one (`relabel`).
    """

    config: GrabConfig = field(default_factory=GrabConfig)
    mirror: bool = True
    aspect: float = 4 / 3

    def __post_init__(self) -> None:
        self._detectors = {
            hand: GrabDetector(self.config, mirror=self.mirror, aspect=self.aspect, only=hand)
            for hand in GRAB_HANDS
        }
        self.states: dict[str, GrabState] = {hand: GrabState() for hand in GRAB_HANDS}
        # Where each label's wrist was last seen: of two hands given the
        # same label, the one nearer keeps it.
        self._wrists: dict[str, Point2] = {}

    def set_aspect(self, aspect: float) -> None:
        """The camera's real frame shape (width / height)."""
        self.aspect = aspect
        for detector in self._detectors.values():
            detector.aspect = aspect

    def update(self, frame: Frame, t_ms: float) -> dict[str, GrabState]:
        frame = self.relabel(frame)
        self.states = {hand: detector.update(frame, t_ms) for hand, detector in self._detectors.items()}
        self._wrists = {**self._wrists, **{h.handedness: (h.wrist[0], h.wrist[1]) for h in frame.hands}}
        return self.states

    def features(self) -> dict[str, float]:
        """Every hand's grab channels (`grab.<hand>.*`)."""
        out: dict[str, float] = {}
        for hand, state in self.states.items():
            out.update(state.features(f"grab.{hand}"))
        return out

    # --- steady labels ------------------------------------------------------

    def relabel(self, frame: Frame) -> Frame:
        """`frame`, its hands labelled steadily (see the class docstring)."""
        if not frame.hands:
            return frame
        labels = self._label_the_rest(frame.hands, self._label_holders(frame.hands))
        hands = [
            hand if labels[i] == hand.handedness else replace(hand, handedness=labels[i])
            for i, hand in enumerate(frame.hands)
        ]
        return replace(frame, hands=hands)

    def _label_holders(self, hands: list[HandObservation]) -> dict[int, str]:
        """The hands holding a grab, by index. Each is the hand pinching where
        that grab's pinch just was (nearest first) - or, if it moved further
        than that in a frame, the one hand still carrying its label."""
        held = {label: d.held_pinch for label, d in self._detectors.items() if d.held_pinch is not None}
        pairs = sorted(
            (self._dist(pinch_point(hand), pinch), label, i)
            for label, pinch in held.items()
            for i, hand in enumerate(hands)
        )
        labels: dict[int, str] = {}
        for distance, label, i in pairs:
            if distance < self.config.relabel_distance and i not in labels and label not in labels.values():
                labels[i] = label
        for label in held:
            named = [i for i, hand in enumerate(hands) if hand.handedness == label and i not in labels]
            if label not in labels.values() and len(named) == 1:
                labels[named[0]] = label
        return labels

    def _label_the_rest(self, hands: list[HandObservation], labels: dict[int, str]) -> dict[int, str]:
        """Every other hand keeps its own label if that is free. Of two given
        the same one, the hand nearer where that label's wrist was keeps it;
        the other gets the label left over."""
        out = dict(labels)
        rest = sorted((i for i in range(len(hands)) if i not in out), key=lambda i: self._wrist_distance(hands[i]))
        for i in rest:
            label = hands[i].handedness
            spare = [other for other in GRAB_HANDS if other not in out.values()]
            out[i] = label if label not in out.values() or not spare else spare[0]
        return out

    def _wrist_distance(self, hand: HandObservation) -> float:
        last = self._wrists.get(hand.handedness)
        return math.inf if last is None else self._dist(hand.wrist, last)

    def _dist(self, a: Point | Point2, b: Point | Point2) -> float:
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])
