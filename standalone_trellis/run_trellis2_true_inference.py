#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image


DEFAULT_TRELLIS2_ROOT = pathlib.Path("/home/wimaxs/Documents/TRELLIS.2")
DEFAULT_MODEL_ROOT = DEFAULT_TRELLIS2_ROOT / "TRELLIS.2-4B"
DEFAULT_IMAGE = DEFAULT_TRELLIS2_ROOT / "assets/example_image/T.png"


def import_trellis2(root: pathlib.Path):
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    from trellis2 import models
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import samplers
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    return models, image_feature_extractor, samplers, Trellis2ImageTo3DPipeline


def load_pipeline_without_rembg(trellis2_root: pathlib.Path, model_root: pathlib.Path):
    models_mod, image_feature_extractor, samplers, PipelineCls = import_trellis2(trellis2_root)
    with open(model_root / "pipeline.json", "r", encoding="utf-8") as f:
        args = json.load(f)["args"]

    loaded = {}
    for name, rel in args["models"].items():
        if name not in PipelineCls.model_names_to_load:
            continue
        loaded[name] = models_mod.from_pretrained(str(model_root / rel))

    pipe = PipelineCls(loaded)
    pipe._pretrained_args = args
    pipe.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(**args["sparse_structure_sampler"]["args"])
    pipe.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipe.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(**args["shape_slat_sampler"]["args"])
    pipe.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]
    pipe.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(**args["tex_slat_sampler"]["args"])
    pipe.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]
    pipe.shape_slat_normalization = args["shape_slat_normalization"]
    pipe.tex_slat_normalization = args["tex_slat_normalization"]
    pipe.image_cond_model = getattr(image_feature_extractor, args["image_cond_model"]["name"])(**args["image_cond_model"]["args"])
    pipe.rembg_model = None
    pipe.low_vram = args.get("low_vram", True)
    pipe.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
    pipe.pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipe._device = "cpu"
    return pipe


def write_obj(path: pathlib.Path, vertices: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis2-root", type=pathlib.Path, default=DEFAULT_TRELLIS2_ROOT)
    parser.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--image", type=pathlib.Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/trellis2_true_mesh.obj"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    pipe = load_pipeline_without_rembg(args.trellis2_root, args.model_root)
    pipe.to(torch.device("cuda"))

    image = Image.open(args.image)
    meshes = pipe.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        preprocess_image=False,
        sparse_structure_sampler_params={"steps": args.steps},
        shape_slat_sampler_params={"steps": args.steps},
        tex_slat_sampler_params={"steps": args.steps},
    )
    mesh = meshes[0]
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    write_obj(args.out, vertices, faces)
    print(f"wrote {args.out}")
    print(f"vertices={vertices.shape[0]} faces={faces.shape[0]}")


if __name__ == "__main__":
    main()
