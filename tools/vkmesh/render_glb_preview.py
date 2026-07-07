#!/usr/bin/env python3
import argparse
import math
import sys

import moderngl
import numpy as np
import trimesh
from PIL import Image


VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    v_normal = normalize(in_normal);
    v_uv = in_uv;
    gl_Position = vec4(in_pos.xy, -in_pos.z * 0.5, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex0;
in vec3 v_normal;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec4 base = texture(tex0, v_uv);
    vec3 n = normalize(v_normal);
    float lit = 0.45 + 0.35 * max(dot(n, normalize(vec3(0.4, 0.7, 0.6))), 0.0)
                    + 0.20 * max(dot(n, normalize(vec3(-0.5, 0.3, 0.8))), 0.0);
    frag_color = vec4(base.rgb * lit, 1.0);
}
"""


def rotation_matrix(yaw_degrees, pitch_degrees):
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return ry @ rx


def load_single_textured_mesh(path):
    scene = trimesh.load(path, force="scene")
    if len(scene.geometry) != 1:
        raise RuntimeError(f"expected one geometry, found {len(scene.geometry)}")
    geom = next(iter(scene.geometry.values()))
    if not hasattr(geom.visual, "uv") or geom.visual.uv is None:
        raise RuntimeError("mesh has no UVs")
    material = getattr(geom.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        raise RuntimeError("mesh material has no baseColorTexture")
    return geom, image.convert("RGBA")


def main():
    parser = argparse.ArgumentParser(description="Render a GLB texture preview with an EGL ModernGL context.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--yaw", type=float, default=-20.0)
    parser.add_argument("--pitch", type=float, default=3.0)
    parser.add_argument("--no-uv-flip-upload", action="store_true")
    args = parser.parse_args()

    geom, image = load_single_textured_mesh(args.input)
    vertices = np.asarray(geom.vertices, dtype=np.float32)
    faces = np.asarray(geom.faces, dtype=np.uint32)
    normals = np.asarray(geom.vertex_normals, dtype=np.float32)
    uvs = np.asarray(geom.visual.uv, dtype=np.float32)
    if vertices.shape[0] != normals.shape[0] or vertices.shape[0] != uvs.shape[0]:
        raise RuntimeError("position, normal, and uv counts do not match")

    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    if extent <= 0.0:
        raise RuntimeError("degenerate mesh bounds")
    rot = rotation_matrix(args.yaw, args.pitch)
    pos = ((vertices - center) @ rot.T) * (1.72 / extent)
    nrm = normals @ rot.T
    packed = np.concatenate([pos, nrm, uvs], axis=1).astype(np.float32)

    ctx = moderngl.create_standalone_context(backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vbo = ctx.buffer(packed.tobytes())
    ibo = ctx.buffer(faces.reshape(-1).astype(np.uint32).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")], ibo)

    tex_pixels = np.array(image, dtype=np.uint8)
    if not args.no_uv_flip_upload:
        tex_pixels = np.flipud(tex_pixels)
    tex = ctx.texture(image.size, 4, tex_pixels.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    tex.use(0)
    prog["tex0"].value = 0

    color = ctx.texture((args.size, args.size), 4)
    depth = ctx.depth_renderbuffer((args.size, args.size))
    fbo = ctx.framebuffer(color, depth)
    fbo.use()
    fbo.clear(0.945, 0.945, 0.935, 1.0)
    vao.render(mode=moderngl.TRIANGLES)
    data = fbo.read(components=4, alignment=1)
    out = Image.frombytes("RGBA", (args.size, args.size), data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    out.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"render_glb_preview: {exc}", file=sys.stderr)
        raise
