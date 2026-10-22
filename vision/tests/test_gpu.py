"""Tests for drawing the skin mesh on the graphics card.

Skipped where no OpenGL 3.3 context can be made (a machine without a GPU
driver): the lab then falls back to the CPU effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofp_vision.skinmesh import build_patch

moderngl = pytest.importorskip("moderngl")
try:
    moderngl.create_standalone_context().release()
except Exception:  # pragma: no cover - depends on the machine
    pytest.skip("no OpenGL context on this machine", allow_module_level=True)

from ofp_vision.gpu import SkinRenderer, vertex_normals  # noqa: E402
from ofp_vision.skin import Window  # noqa: E402

W, H = 640, 360
LIGHT = np.array([0.0, -0.35, 1.0]) / np.linalg.norm([0.0, -0.35, 1.0])
BOX = Window(200, 60, 440, 300)


def textured() -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[..., 0] = (xs * 3) % 256
    image[..., 1] = (ys * 5) % 256
    image[..., 2] = (xs + ys) % 256
    return image


def flat_mesh(spacing: float = 8.0):
    patch = build_patch(np.array([320.0, 180.0]), 80.0, spacing)
    rest = np.zeros((len(patch.local), 3))
    rest[:, :2] = patch.local
    return patch, rest


@pytest.fixture(scope="module")
def renderer():
    gpu = SkinRenderer()
    yield gpu
    gpu.release()


def draw(renderer, image, positions, rest, patch, skin=None):
    """Draw with the skin mask covering the whole frame (or `skin`)."""
    renderer.set_skin(np.ones((H, W), np.float32) if skin is None else skin)
    uv = rest[:, :2] / np.array([W, H])
    normals = vertex_normals(positions, patch.triangles)
    return renderer.draw(image, positions, uv, uv, normals, patch.triangles, LIGHT, BOX)


def test_skin_at_rest_is_drawn_exactly_as_the_camera_saw_it(renderer):
    """Undisturbed, the mesh must be invisible: pixel for pixel the image."""
    image = textured()
    patch, rest = flat_mesh()
    colour, coverage = draw(renderer, image, rest, rest, patch)
    inside = (slice(70, 170), slice(70, 170))  # well inside the mesh, in box coordinates
    assert coverage[inside].min() > 0.99
    assert np.abs(colour[inside] - BOX.crop(image)[inside].astype(np.float32)).max() <= 1.0


def test_moved_skin_carries_its_own_picture(renderer):
    image = textured()
    patch, rest = flat_mesh()
    moved = rest.copy()
    moved[:, 0] += 20.0  # the whole patch slides right by 20px
    colour, _ = draw(renderer, image, moved, rest, patch)
    y, x = 120, 100
    assert np.abs(colour[y, x + 20] - BOX.crop(image)[y, x].astype(np.float32)).max() <= 2.0


def test_skin_tilted_away_from_the_light_is_darker(renderer):
    image = np.full((H, W, 3), 160, np.uint8)
    patch, rest = flat_mesh()
    tilted = rest.copy()
    # Upper part nearer the camera: the surface faces downward.
    tilted[:, 2] = -(rest[:, 1] - 180.0) * 0.8
    colour, _ = draw(renderer, image, tilted, rest, patch)
    flat_colour, _ = draw(renderer, image, rest, rest, patch)
    # Light comes from above; this surface faces away from it.
    assert colour[120, 120].mean() < flat_colour[120, 120].mean() - 5


def test_the_edge_of_the_skin_is_soft(renderer):
    image = textured()
    patch, rest = flat_mesh()
    xs = np.arange(W)[None, :].repeat(H, axis=0)
    skin = np.clip((300.0 - xs) / 20.0, 0.0, 1.0).astype(np.float32)  # fades out from x 280 to 300
    _, coverage = draw(renderer, image, rest, rest, patch, skin)
    values = coverage[120, 60:120]  # x 260..320
    assert values.min() < 0.05 and values.max() > 0.95
    assert np.any((values > 0.2) & (values < 0.8)), "a soft ramp, not a hard step"


def test_stretching_across_the_skins_edge_never_shows_what_was_beyond_it(renderer):
    """The bug in the screenshot: a finger's edge triangles, half skin and
    half wall, stretched into wide sheets that showed the wall.

    Coarse triangles (40px) straddle the edge of the skin (x 320); the mesh
    is stretched to 3x its width around x 300. Visibility is looked up per
    pixel where it came from, so beyond the stretched edge (from x ~ 360)
    nothing shows - however big the triangles got.
    """
    image = textured()
    patch, rest = flat_mesh(spacing=40.0)
    skin = np.zeros((H, W), np.float32)
    skin[:, :320] = 1.0
    stretched = rest.copy()
    stretched[:, 0] = 300.0 + (rest[:, 0] - 300.0) * 3.0
    _, coverage = draw(renderer, image, stretched, rest, patch, skin)
    columns = np.arange(BOX.x0, BOX.x1)
    beyond = columns > 300 + (320 - 300) * 3 + 2  # past where the edge was stretched to
    within = columns < 300 + (320 - 300) * 3 - 4
    assert coverage[:, beyond].max() < 0.02
    assert coverage[100:140, within].min() > 0.9


def test_flat_skin_faces_the_camera():
    patch, rest = flat_mesh()
    normals = vertex_normals(rest, patch.triangles)
    assert normals == pytest.approx(np.tile([0.0, 0.0, 1.0], (len(rest), 1)))
