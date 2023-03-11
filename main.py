import shutil

from profiler.convert import convert_torch_to_onnx, convert_tensorflow_to_onnx
from profiler.onnx_model import profile_onnx_model
from profiler.tf_model import profile_tf_model
from profiler.torch_model import profile_torch_model


def torch2onnx_main():
    from pathlib import Path

    from torchvision.models import resnet50
    import torch
    import numpy as np

    save_path = Path('test')
    save_path.mkdir(exist_ok=True)
    model = resnet50()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    dynamic_axes = {
        'input': [0],
    }
    onnx_model_str = convert_torch_to_onnx(
        model, x, str(save_path / "test.onnx"),
        input_names=["input"],
        dynamic_axes=dynamic_axes,
        output_names=["output"]
    )
    profile = profile_onnx_model(
        onnx_model_str, np.random.randn(1, 3, 224, 224).astype(np.float32)
    )
    shutil.move(profile, str(save_path / "profile.json"))


def tensorflow2onnx_main():
    from pathlib import Path

    import tensorflow as tf
    import numpy as np

    save_path = Path('test')
    save_path.mkdir(exist_ok=True)
    model = tf.keras.applications.ResNet50()
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)
    onnx_model_str = convert_tensorflow_to_onnx(
        model, x, str(save_path / "test_tf.onnx"),
    )
    profile = profile_onnx_model(
        onnx_model_str, np.random.randn(1, 224, 224, 3).astype(np.float32)
    )
    shutil.move(profile, str(save_path / "profile_tf_onnx.json"))


def tensorflow_main():
    import tensorflow as tf
    import numpy as np

    model = tf.keras.applications.ResNet50()
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)

    profile = profile_tf_model(
        model, x
    )


def torch_main():
    from pathlib import Path

    from torchvision.models import resnet50
    import torch

    save_path = Path('test')
    save_path.mkdir(exist_ok=True)
    model = resnet50()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    prof_file = profile_torch_model(
        model, x
    )
    shutil.move(prof_file, save_path / "profile.json")


if __name__ == '__main__':
    torch_main()
