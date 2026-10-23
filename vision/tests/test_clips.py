"""Tests for clips: short recordings of the real camera, for tuning the stretch.

- a clip plays back what was recorded: the pictures (to JPEG quality), each
  frame's tracking, and the skin segmentation exactly as the lab had it, on
  the frames that had it
- it stops itself at its length limit: a forgotten recording cannot fill
  the disk
- rendering a clip runs the grab and the stretch on it, frame by frame, and
  writes the stretched video and stills of the grabs to look at
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from face_fixture import H, W, image, landmarks
from test_grab import OPEN, PINCHED, make_hand

from ofp_vision.clips import ClipRecorder, read_clip
from ofp_vision.grab import face_target_points
from ofp_vision.signals import Frame


def recorded(count: int) -> list[tuple[np.ndarray, Frame]]:
    """Frames with tracking; the segmentation arrives on frame 1 and is
    replaced on frame 3, as the lab's (which runs every third frame) is."""
    first = np.zeros((H // 2, W // 2), dtype=np.uint8)
    first[10:50, 10:50] = 3
    second = first.copy()
    second[60:90] = 2
    out = []
    for i in range(count):
        labels = None if i == 0 else (first if i < 3 else second)
        hand = make_hand((0.5, 0.5), OPEN, "Left")
        out.append((image(), Frame(t_ms=i * 33.3, face=landmarks(), hands=[hand], segmentation=labels)))
    return out


def write(folder, frames, max_seconds: float = 30.0) -> ClipRecorder:
    recorder = ClipRecorder(folder, (W, H), fps=30.0, max_seconds=max_seconds)
    for bgr, frame in frames:
        recorder.write(bgr, frame)
    recorder.close()
    return recorder


def test_a_clip_plays_back_what_was_recorded(tmp_path):
    frames = recorded(5)
    write(tmp_path / "clip", frames)
    played = list(read_clip(tmp_path / "clip"))
    assert len(played) == len(frames)
    for (bgr, frame), (bgr_back, frame_back) in zip(frames, played):
        assert np.abs(bgr_back.astype(int) - bgr).mean() < 3, "the picture, to JPEG quality"
        assert frame_back.t_ms == pytest.approx(frame.t_ms, abs=0.01)
        assert frame_back.face[1] == pytest.approx(frame.face[1], abs=1e-5)
        assert frame_back.hands[0].handedness == "Left"
        if frame.segmentation is None:
            assert frame_back.segmentation is None
        else:
            assert np.array_equal(frame_back.segmentation, frame.segmentation)


def test_a_clip_stops_itself_at_its_length_limit(tmp_path):
    bgr, frame = recorded(1)[0]
    recorder = ClipRecorder(tmp_path / "clip", (W, H), fps=30.0, max_seconds=0.2)
    for _ in range(10):
        recorder.write(bgr, frame)
    assert recorder.full
    recorder.close()
    assert len(list(read_clip(tmp_path / "clip"))) == 6


def test_the_lab_key_starts_a_clip_and_stops_it(tmp_path):
    from ofp_vision.clips import ClipSwitch

    switch = ClipSwitch(tmp_path, fps=30.0)
    assert switch.status() is None
    switch.toggle((W, H))
    for bgr, frame in recorded(3):
        switch.write(bgr, frame)
    assert "REC" in switch.status()
    switch.toggle((W, H))
    assert switch.status() is None
    (folder,) = tmp_path.iterdir()
    assert len(list(read_clip(folder))) == 3


def test_a_full_clip_saves_itself_and_stops(tmp_path):
    from ofp_vision.clips import ClipSwitch

    switch = ClipSwitch(tmp_path, fps=30.0, max_seconds=0.1)
    switch.toggle((W, H))
    for bgr, frame in recorded(5):
        switch.write(bgr, frame)
    assert switch.status() is None, "stopped by itself"
    (folder,) = tmp_path.iterdir()
    assert len(list(read_clip(folder))) == 3


def test_rendering_a_clip_stretches_the_grabbed_skin(tmp_path):
    """A right hand arrives open at the left cheek, pinches and pulls out."""
    from ofp_vision.tools.render_clip import render

    cheek = face_target_points(landmarks())["cheekL"]
    frames = []
    for i in range(20):
        gap = OPEN if i < 2 else PINCHED
        at = (cheek[0] + 0.01 * max(0, i - 2), cheek[1])
        frames.append((image(), Frame(t_ms=i * 33.3, face=landmarks(), hands=[make_hand(at, gap, "Right")])))
    write(tmp_path / "clip", frames)

    report = render(tmp_path / "clip", every=4)
    assert report.frames == 20
    assert report.held >= 15, "the pinch was held from frame 2 on"
    assert report.stills >= 3
    video = cv2.VideoCapture(str(report.video))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    video.release()
    still = cv2.imread(str(report.still_paths[-1]))
    left, right = still[:, : still.shape[1] // 2], still[:, still.shape[1] // 2 :]
    assert np.abs(left.astype(int) - right).max() > 40, "the stretched side differs from the camera"


def test_frames_that_shared_a_segmentation_share_it_on_playback(tmp_path):
    """Live, a segmentation stays one array until the next arrives (every
    third frame), and the room is learned once per segmentation. Played back
    it must be the same, or the room learns three times as often."""
    write(tmp_path / "clip", recorded(5))
    played = [frame.segmentation for _, frame in read_clip(tmp_path / "clip")]
    assert played[1] is played[2]
    assert played[3] is played[4]
    assert played[2] is not played[3]


def test_a_clip_that_cannot_be_saved_does_not_stop_the_lab(tmp_path, monkeypatch):
    from ofp_vision.clips import ClipSwitch

    switch = ClipSwitch(tmp_path, fps=30.0)
    switch.toggle((W, H))

    def disk_full():
        raise OSError("disk full")

    monkeypatch.setattr(switch._recorder, "close", disk_full)
    switch.stop()
    assert switch.status() is None
