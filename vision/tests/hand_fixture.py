"""A synthetic hand for the sign and fire tests.

21 landmarks, palm 0.2 frame heights long. A finger that is `out` points
straight up (long and straight); one that is `in` folds back on itself into
the palm (short and sharply bent). That is all `gestures.py` measures, so the
same hand can be posed into a snap, a fist or a point.
"""

from __future__ import annotations

from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import HandObservation

ASPECT = 4 / 3

WRIST = (0.50, 0.85)
KNUCKLES = {
    "thumb": (0.40, 0.72),
    "index": (0.44, 0.66),
    "middle": (0.50, 0.65),
    "ring": (0.56, 0.66),
    "pinky": (0.62, 0.69),
}
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def make_hand(
    out: dict[str, bool] | None = None,
    tips: dict[str, tuple[float, float]] | None = None,
    side: str = "Right",
) -> HandObservation:
    """A hand with the given digits out (default: all of them).

    `tips` overrides individual fingertips afterwards, which is how the snap
    tests bring the thumb onto the middle finger and pull it away again.
    """
    out = out if out is not None else {digit: True for digit in DIGITS}
    points: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * 21
    points[HAND_WRIST] = (*WRIST, 0.0)
    points[HAND_MIDDLE_MCP] = (*KNUCKLES["middle"], 0.0)

    for digit, knuckle in KNUCKLES.items():
        if out.get(digit, True):
            lean = (knuckle[0] - 0.5) * 0.5
            joints = [
                knuckle,
                (knuckle[0] + lean * 0.4, knuckle[1] - 0.07),
                (knuckle[0] + lean * 0.8, knuckle[1] - 0.12),
                (knuckle[0] + lean, knuckle[1] - 0.17),
            ]
        else:
            # Folded: out to the palm, then back up toward the knuckle.
            joints = [
                knuckle,
                (knuckle[0] + 0.010, knuckle[1] + 0.060),
                (knuckle[0] + 0.020, knuckle[1] + 0.030),
                (knuckle[0] + 0.030, knuckle[1] + 0.010),
            ]
        for joint, point in zip(HAND_DIGITS[digit], joints):
            points[joint] = (*point, 0.0)

    for digit, point in (tips or {}).items():
        points[HAND_DIGITS[digit][3]] = (*point, 0.0)

    return HandObservation(handedness=side, landmarks=points)


def moved(hand: HandObservation, dx: float = 0.0, dy: float = 0.0) -> HandObservation:
    """The same pose shifted in the frame, for shake and tracking tests."""
    return HandObservation(
        handedness=hand.handedness,
        landmarks=[(x + dx, y + dy, z) for x, y, z in hand.landmarks],
    )


def fingertip(hand: HandObservation, digit: str = "middle") -> tuple[float, float]:
    point = hand.landmarks[HAND_DIGITS[digit][3]]
    return (point[0], point[1])


def snap_closed(hand: HandObservation) -> HandObservation:
    """The same hand with the thumb tip pressed onto the middle fingertip."""
    tips = {"thumb": fingertip(hand, "middle")}
    return HandObservation(handedness=hand.handedness, landmarks=_with_tips(hand, tips))


def _with_tips(hand: HandObservation, tips: dict[str, tuple[float, float]]) -> list:
    points = list(hand.landmarks)
    for digit, point in tips.items():
        points[HAND_DIGITS[digit][3]] = (*point, 0.0)
    return points
