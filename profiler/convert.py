from typing import Union

import numpy as np
import torch.nn
import tensorflow as tf
import tf2onnx


def convert_torch_to_onnx(
        model: torch.nn.Module,
        input_data: torch.Tensor,
        model_name: str,
        **kwargs
):
    torch.onnx.export(
        model, input_data, model_name, opset_version=11, **kwargs
    )
    return model_name


def convert_tensorflow_to_onnx(
        model: tf.keras.Model,
        input_data: Union[np.ndarray, tf.Tensor],
        model_name: str,
        **kwargs
):
    signature = [tf.TensorSpec(shape=input_data.shape, dtype=tf.float32)]
    tf2onnx.convert.from_keras(
        model, signature, opset=11, output_path=model_name, **kwargs
    )
    return model_name
