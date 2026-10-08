"""
Publish a fine-tuned Laya checkpoint to the self-hosted ollaya registry.

DJ Laya is laya:en with new weights and calibration: same ModernBERT-large
architecture, tokenizer and ONNX graphs (which are weightless and read the
weights file by tensor name). So its manifest is laya:en's with three layers
swapped -- weights, calibration and the model-info config -- and every blob is
copied into the registry so pulls never depend on ollaya.dev or Hugging Face.

Runs on the unraid box inside the training image:

    docker run --rm -v /mnt/mycelium/appdata/djlaya:/work \
        -v /mnt/mycelium/appdata/ollaya:/ollaya:ro \
        -v /mnt/mycelium/appdata/ollaya-registry:/registry \
        djlaya-train python /work/publish.py /work/djlaya-v1 djlaya v1
"""

import hashlib
import json
import shutil
import struct
import sys
from datetime import date
from pathlib import Path

STORE = Path("/ollaya/models")
REGISTRY = Path("/registry/v2/library")
BASE = STORE / "manifests/ollaya.dev/library/laya/en"

WEIGHTS = "application/vnd.ollaya.weights"
CALIBRATION = "application/vnd.ollaya.calibration"
CONFIG = "application/vnd.ollaya.config.v1+json"


GRAPH = "application/vnd.ollaya.graph.onnx"


def safetensors_index(path: Path) -> dict[str, tuple[int, int]]:
    """{tensor name: (absolute byte offset, length)} from a safetensors header."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0]) for k, v in header.items()}


def retarget_graph(graph: Path, base_weights: Path, new_weights: Path, location: str) -> bytes:
    """The graph with its external weights pointed at `new_weights`.

    ollaya's graphs are weightless: each initializer names the weights blob
    (`sha256-<hex>`, resolved next to the graph in the blob store) and a byte
    offset into it. Copying the stock graph would silently serve the stock
    weights, and the fine-tuned file's header is a different length, so every
    offset is translated through the tensor name it came from.
    """
    import onnx

    model = onnx.load(str(graph), load_external_data=False)
    by_offset = {v: k for k, v in safetensors_index(base_weights).items()}
    new = safetensors_index(new_weights)
    moved = 0
    for tensor in model.graph.initializer:
        fields = {e.key: e for e in tensor.external_data}
        if "location" not in fields:
            continue
        key = (int(fields["offset"].value), int(fields["length"].value))
        name = by_offset[key]  # KeyError: the graph reads bytes no tensor owns
        offset, length = new[name]
        assert length == key[1], f"{name} changed size"
        fields["location"].value = location
        fields["offset"].value = str(offset)
        moved += 1
    print(f"  retargeted {moved} external initializers of {graph.name[:19]}")
    return model.SerializeToString()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put(blobs: Path, data: bytes | Path) -> dict:
    """Store a blob under its digest and return {digest, size}."""
    if isinstance(data, Path):
        digest = sha256(data)
        target = blobs / f"sha256:{digest}"
        if not target.exists():
            shutil.copyfile(data, target)
        size = data.stat().st_size
    else:
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256:{digest}").write_bytes(data)
        size = len(data)
    return {"digest": f"sha256:{digest}", "size": size}


def main(checkpoint: str, name: str, tag: str):
    checkpoint = Path(checkpoint)
    base = json.loads(BASE.read_text())
    blobs = REGISTRY / name / "blobs"
    manifests = REGISTRY / name / "manifests"
    blobs.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    agent_cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    temps = agent_cfg["temperature"]
    calibration = json.dumps({"temperature": temps}, indent=2).encode()
    info = json.loads((STORE / "blobs" / base["config"]["digest"].replace(":", "-")).read_text())
    info.update(
        description="DJ Laya: laya:en distilled from Jev on sprintboy's routing and same-release questions.",
        source=f"local fine-tune {checkpoint.name}",
        release_date=date.today().isoformat(),
    )

    new_weights = checkpoint / "model.safetensors"
    base_weights_layer = next(l for l in base["layers"] if l["mediaType"] == WEIGHTS)
    base_weights = STORE / "blobs" / base_weights_layer["digest"].replace(":", "-")
    weights_blob = put(blobs, new_weights)
    weights_location = weights_blob["digest"].replace(":", "-")

    def swap(layer: dict) -> dict:
        kind = layer["mediaType"]
        if kind == WEIGHTS:
            new = weights_blob
        elif kind == GRAPH:
            src = STORE / "blobs" / layer["digest"].replace(":", "-")
            new = put(blobs, retarget_graph(src, base_weights, new_weights, weights_location))
        elif kind == CALIBRATION:
            new = put(blobs, calibration)
        else:
            src = STORE / "blobs" / layer["digest"].replace(":", "-")
            new = put(blobs, src)
            assert new["digest"] == layer["digest"], f"{src} does not match its digest"
        out = {k: v for k, v in layer.items() if k != "urls"}
        return out | new

    config = {"mediaType": CONFIG} | put(blobs, json.dumps(info, indent=2).encode())
    manifest = {k: v for k, v in base.items() if k not in ("config", "layers")}
    manifest["config"] = config
    manifest["layers"] = [swap(layer) for layer in base["layers"]]
    body = json.dumps(manifest, indent=2).encode()
    for t in (tag, "latest"):
        (manifests / t).write_bytes(body)
    print(f"published {name}:{tag} with temperatures {[round(t, 3) for t in temps]}")
    for layer in manifest["layers"]:
        print(f"  {layer['mediaType']:40} {layer['size']:>12,} {layer['digest'][:19]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
