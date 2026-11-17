#!/usr/bin/env python3
"""Convert the bundled Core ML graphs from the official Real-ESRGAN checkpoints."""
import argparse
import hashlib
import io
import pickle
import urllib.request
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.neural_network.quantization_utils import quantize_weights
from coremltools.proto.FeatureTypes_pb2 import ImageFeatureType

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
EXTREME_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
EXTREME_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
MODEL_DIR = Path(__file__).resolve().parents[1] / "App" / "Models"


class FloatStorage:
    """Marker for the one storage type in the verified release file."""


def rebuild_tensor(storage, offset, shape, strides, requires_grad, hooks):
    if offset != 0 or storage.size != np.prod(shape) or requires_grad or hooks:
        raise ValueError("Unexpected checkpoint tensor layout")
    view = storage.reshape(shape)
    if tuple(stride // view.itemsize for stride in view.strides) != strides:
        raise ValueError("Unexpected checkpoint tensor strides")
    return view


class CheckpointReader(pickle.Unpickler):
    # Support only the three globals used by this hash-pinned PyTorch checkpoint.
    globals = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch", "FloatStorage"): FloatStorage,
        ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
    }

    def __init__(self, archive):
        self.archive = archive
        super().__init__(io.BytesIO(archive.read("archive/data.pkl")))

    def find_class(self, module, name):
        try:
            return self.globals[module, name]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Unexpected checkpoint global {module}.{name}") from exc

    def persistent_load(self, identifier):
        kind, storage_type, key, _, count = identifier
        if kind != "storage" or storage_type is not FloatStorage or not key.isdecimal():
            raise pickle.UnpicklingError("Unexpected checkpoint storage")
        data = self.archive.read(f"archive/data/{key}")
        if len(data) != count * 4:
            raise ValueError("Unexpected checkpoint storage size")
        return np.frombuffer(data, dtype="<f4")


def checkpoint(path, url=URL, digest=SHA256, parameter_key="params"):
    if path:
        source = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            source = response.read()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError("Real-ESRGAN checkpoint SHA-256 mismatch")
    with ZipFile(io.BytesIO(source)) as archive:
        return CheckpointReader(archive).load()[parameter_key]


def build(params, name, size):
    inputs = [("image", datatypes.Array(3, size, size))]
    outputs = [("enhanced", datatypes.Array(3, size * 4, size * 4))]
    nn = NeuralNetworkBuilder(inputs, outputs, disable_rank5_shape_mapping=True)
    current = "rgb"
    for index in range(67):
        weight = params[f"body.{index}.weight"]
        layer = f"body_{index}"
        if index % 2:
            nn.add_activation(layer, "PRELU", current, layer, params=weight)
        else:
            bias = params[f"body.{index}.bias"]
            nn.add_convolution(
                layer, weight.shape[1], weight.shape[0], 3, 3, 1, 1, "same", 1,
                np.transpose(weight, (2, 3, 1, 0)), bias, True,
                input_name=current, output_name=layer,
            )
        current = layer

    nn.set_pre_processing_parameters(image_input_names=["image"], image_scale=1 / 255.0)
    # The image-input preprocessing is also used by the nearest-neighbor skip connection.
    nn.add_activation("input_identity", "LINEAR", "image", "rgb", params=[1.0, 0.0])
    layers = nn.spec.neuralNetwork.layers
    first = type(layers[0])()
    first.CopyFrom(layers[-1])
    del layers[-1]
    layers.insert(0, first)

    nn.add_reorganize_data("pixel_shuffle", current, "detail", mode="PIXEL_SHUFFLE", block_size=4)
    nn.add_upsample("skip", 4, 4, "rgb", "nearest", mode="NN")
    nn.add_elementwise("residual", ["nearest", "detail"], "rgb01", mode="ADD")
    export(nn, "rgb01", name, size, SHA256,
           "Real-ESRGAN general x4v3 (SRVGGNetCompact), x4 RGB super-resolution")


def build_extreme(params):
    # RRDBNet x4 uses the tensor names and residual topology from BasicSR.
    size = 96
    nn = NeuralNetworkBuilder([("image", datatypes.Array(3, size, size))],
                              [("enhanced", datatypes.Array(3, size * 4, size * 4))],
                              disable_rank5_shape_mapping=True)
    nn.set_pre_processing_parameters(image_input_names=["image"], image_scale=1 / 255.0)

    def conv(key, source, activate=False):
        name = key.replace(".", "_")
        weight = params[key + ".weight"]
        bias = params[key + ".bias"]
        nn.add_convolution(name, weight.shape[1], weight.shape[0], 3, 3, 1, 1, "same", 1,
                           np.transpose(weight, (2, 3, 1, 0)), bias, True,
                           input_name=source, output_name=name)
        if activate:
            result = name + "_activation"
            nn.add_activation(result, "LEAKYRELU", name, result, params=[0.2])
            return result
        return name

    def add_residual(name, base, residual):
        scaled = name + "_scaled"
        nn.add_activation(scaled, "LINEAR", residual, scaled, params=[0.2, 0.0])
        nn.add_elementwise(name, [base, scaled], name, mode="ADD")
        return name

    first = conv("conv_first", "image")
    trunk = first
    for block in range(23):
        block_input = trunk
        for dense in range(1, 4):
            prefix = f"body.{block}.rdb{dense}"
            features = [trunk]
            for index in range(1, 6):
                source = trunk
                if index > 1:
                    source = f"concat_{block}_{dense}_{index}"
                    nn.add_elementwise(source, features, source, mode="CONCAT")
                features.append(conv(f"{prefix}.conv{index}", source, activate=index < 5))
            trunk = add_residual(f"dense_{block}_{dense}", trunk, features[-1])
        trunk = add_residual(f"rrdb_{block}", block_input, trunk)

    body = conv("conv_body", trunk)
    nn.add_elementwise("long_skip", [first, body], "long_skip", mode="ADD")
    trunk = "long_skip"
    for index in (1, 2):
        up = f"nearest_{index}"
        nn.add_upsample(up, 2, 2, trunk, up, mode="NN")
        trunk = conv(f"conv_up{index}", up, activate=True)
    output = conv("conv_last", conv("conv_hr", trunk, activate=True))
    export(nn, output, "MuzzleExtremeSR", size, EXTREME_SHA256,
           "Real-ESRGAN x4plus (RRDBNet), x4 RGB photo restoration")


def export(nn, source, name, size, digest, description):
    nn.add_clip("clamp", source, "clamped", 0.0, 1.0)
    nn.add_activation("to_byte", "LINEAR", "clamped", "enhanced", params=[255.0, 0.0])

    spec = nn.spec
    color = spec.description.output[0].type.imageType
    color.width = size * 4
    color.height = size * 4
    color.colorSpace = ImageFeatureType.RGB
    metadata = spec.description.metadata
    metadata.shortDescription = description
    metadata.author = "Xintao Wang et al. (Real-ESRGAN)"
    metadata.license = "BSD-3-Clause"
    metadata.userDefined["checkpoint_sha256"] = digest

    model = ct.models.MLModel(spec, skip_model_load=True)
    quantized = quantize_weights(model, nbits=16, skip_model_load=True)
    # On non-macOS hosts, quantize_weights returns the serialized specification.
    spec = quantized.get_spec() if isinstance(quantized, ct.models.MLModel) else quantized
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output = MODEL_DIR / f"{name}.mlmodel"
    ct.models.utils.save_spec(spec, str(output))
    print(f"{output.name}: {size} → {size * 4}px, {hashlib.sha256(output.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Verified local copy; default: official GitHub release")
    parser.add_argument("--extreme-checkpoint", type=Path, help="Verified local x4plus checkpoint")
    args = parser.parse_args()
    params = checkpoint(args.checkpoint)
    build(params, "MuzzlePreviewSR", 48)
    build(params, "MuzzlePhotoSR", 128)
    build_extreme(checkpoint(args.extreme_checkpoint, EXTREME_URL, EXTREME_SHA256, "params_ema"))


if __name__ == "__main__":
    main()
