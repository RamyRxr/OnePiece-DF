"""Drawing the skin mesh on the graphics card.

The skin mesh (`skinmesh.py`, moved by `softbody.py`) is drawn as triangles,
painted with the camera image: each point carries the image coordinate of
where its skin rests, so wherever the physics moves it, it takes its own
piece of the picture along - the way a game engine paints a 3D model.

What is visible is decided per pixel, not per point. The skin mask (from
the segmentation model: skin vs everything else) is a texture too; every
drawn pixel looks up where it came from and shows only if that spot was
skin. However far a triangle on the skin's edge is stretched, the room or
the wall beside a finger can never come along: the edge stays exactly
where the skin's real outline was.

- Undisturbed skin comes out pixel for pixel as the camera saw it, so the
  effect is invisible until something moves.
- The surface is lit relative to flat skin: a patch tilted toward the light
  gets brighter, one tilted away darker, the way stretched skin really
  catches the room's light.
- Triangles are drawn back to front (the most lifted last), so lifted skin
  covers the skin beneath it; edges blend softly (premultiplied alpha).
- Drawn with multisampling (several samples per pixel along triangle
  edges): a stretched strand's outline is smooth, not a pixel staircase.

It renders off-screen (no window of its own) and reads back only the part of
the frame the skin covers. The lab composites that onto the camera frame.
"""

from __future__ import annotations

import moderngl
import numpy as np

from .skin import Window

VERTEX_SHADER = """
#version 330
in vec3 in_position;   // x, y in image pixels; lift toward the camera
in vec2 in_uv;         // where this skin rests in the camera image, 0..1
in vec2 in_skin_uv;    // where it rested in the skin mask, 0..1
in vec3 in_normal;
uniform vec2 size;
out vec2 v_uv;
out vec2 v_skin_uv;
out vec3 v_normal;
void main() {
    // Image row 0 at the bottom of clip space: the framebuffer then reads
    // back top row first, in the image's own order.
    gl_Position = vec4(in_position.xy / size * 2.0 - 1.0, 0.0, 1.0);
    v_uv = in_uv;
    v_skin_uv = in_skin_uv;
    v_normal = in_normal;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D camera;
uniform sampler2D skin;  // 1 where the source pixel is skin, soft at edges
uniform vec3 light;      // unit, x right, y down, z toward the camera
uniform float ambient;
in vec2 v_uv;
in vec2 v_skin_uv;
in vec3 v_normal;
out vec4 colour;
void main() {
    float visible = texture(skin, v_skin_uv).r;
    if (visible <= 0.0) discard;
    vec3 base = texture(camera, v_uv).rgb;
    vec3 n = normalize(v_normal);
    float facing = ambient + (1.0 - ambient) * max(dot(n, light), 0.0);
    float flat_facing = ambient + (1.0 - ambient) * light.z;
    vec3 lit = base * (facing / flat_facing);
    colour = vec4(lit * visible, visible);   // premultiplied
}
"""

# Share of the light that is ambient; the rest comes from the light's direction.
AMBIENT = 0.5
# Samples per pixel along triangle edges (multisample antialiasing).
SAMPLES = 8


def vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unit normals of the (x, y, lift) surface at each point; flat = (0, 0, 1)."""
    a, b, c = (positions[triangles[:, k]] for k in range(3))
    faces = np.cross(b - a, c - a)
    normals = np.zeros_like(positions, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, triangles[:, k], faces)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(length > 1e-12, normals / np.maximum(length, 1e-12), [0.0, 0.0, 1.0])
    # Keep them facing the camera side.
    return np.where(normals[:, 2:3] < 0.0, -normals, normals)


class SkinRenderer:
    """An off-screen OpenGL context that draws skin meshes."""

    def __init__(self) -> None:
        self._ctx = moderngl.create_standalone_context()
        self._program = self._ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self._size: tuple[int, int] | None = None
        self._texture = None
        self._target = None
        self._resolved = None
        self._skin = None
        self._samples = min(SAMPLES, self._ctx.max_samples)

    def release(self) -> None:
        self._ctx.release()

    def set_skin(self, mask: np.ndarray) -> None:
        """The skin mask (2D, 0..1) the next draws read visibility from."""
        if self._skin is not None:
            self._skin.release()
        pixels = np.ascontiguousarray(np.clip(mask, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
        self._skin = self._ctx.texture((pixels.shape[1], pixels.shape[0]), 1, pixels.tobytes(), alignment=1)
        self._skin.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._skin.repeat_x = self._skin.repeat_y = False

    def _vertex_array(self, positions, uv, skin_uv, normals, triangles):
        """The mesh in GPU buffers: (vertex array, its buffers to release)."""
        vertices = np.concatenate(
            [positions.astype(np.float32), uv.astype(np.float32), skin_uv.astype(np.float32),
             normals.astype(np.float32)],
            axis=1,
        )
        vbo = self._ctx.buffer(np.ascontiguousarray(vertices).tobytes())
        ibo = self._ctx.buffer(np.ascontiguousarray(triangles, dtype=np.int32).tobytes())
        vao = self._ctx.vertex_array(
            self._program, [(vbo, "3f 2f 2f 3f", "in_position", "in_uv", "in_skin_uv", "in_normal")], ibo
        )
        return vao, (vbo, ibo)

    def _resize(self, size: tuple[int, int]) -> None:
        if self._size == size:
            return
        for resource in (self._texture, self._target, self._resolved):
            if resource is not None:
                resource.release()
        self._texture = self._ctx.texture(size, 3, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.repeat_x = self._texture.repeat_y = False
        # Drawn multisampled, then resolved to one sample per pixel to read.
        samples = self._ctx.renderbuffer(size, components=4, samples=self._samples)
        self._target = self._ctx.framebuffer(color_attachments=[samples])
        self._resolved = self._ctx.simple_framebuffer(size, components=4)
        self._size = size

    def draw(
        self,
        bgr: np.ndarray,
        positions: np.ndarray,
        uv: np.ndarray,
        skin_uv: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        light: np.ndarray,
        box: Window,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw the mesh; return (premultiplied colour, coverage) over `box`.

        `uv` is where each point's skin rests in `bgr` now; `skin_uv` where
        it rested in the skin mask (see `set_skin`).
        """
        if self._skin is None:
            raise RuntimeError("set_skin() before draw()")
        size = (bgr.shape[1], bgr.shape[0])
        self._resize(size)
        self._texture.write(np.ascontiguousarray(bgr, dtype=np.uint8).tobytes(), alignment=1)
        self._texture.use(0)
        self._skin.use(1)
        program = self._program
        program["size"].value = size
        program["light"].value = tuple(float(v) for v in light)
        program["ambient"].value = AMBIENT
        program["camera"].value = 0
        program["skin"].value = 1

        vao, buffers = self._vertex_array(positions, uv, skin_uv, normals, triangles)
        try:
            self._target.use()
            self._target.clear(0.0, 0.0, 0.0, 0.0)
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            vao.render(moderngl.TRIANGLES)
            self._ctx.copy_framebuffer(self._resolved, self._target)
            data = self._resolved.read(viewport=(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0), components=4)
        finally:
            vao.release()
            for buffer in buffers:
                buffer.release()
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(box.y1 - box.y0, box.x1 - box.x0, 4)
        return pixels[..., :3].astype(np.float32), pixels[..., 3].astype(np.float32) / 255.0
