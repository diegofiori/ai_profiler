# AI profiler
This repo implements an AI profiler that can be used with PyTorch, Tensorflow and ONNX models. 

## Profile Tensorflow
You can easily run the profiler on a Tensorflow model by modifying the following example:
```python
import tensorflow as tf
import numpy as np

from profiler.tf_model import profile_tf_model


model = tf.keras.applications.ResNet50()
x = np.random.randn(1, 224, 224, 3).astype(np.float32)
logdir = "logdir"

profile = profile_tf_model(
    model, x, logdir=logdir
)
```
The profiler will save the profile in the `logdir` directory. You can visualize the profile on tensorboard by running:
```bash
tensorboard --logdir logdir
```
The terminal will print the URL where you can access the tensorboard dashboard. Once in the tensorboard dashboard you can select the `profile` tab to visualize the profile. Note that the tab is located in the right upper side of screen.

If you want to simply trace the model execution we suggest to use the onnx trace instead. You have to convert the model to onnx first. You can do it by modifying the example below:
```python
from pathlib import Path

import tensorflow as tf
import numpy as np

from profiler.convert import convert_tensorflow_to_onnx


save_path = Path('test')
save_path.mkdir(exist_ok=True)
model = tf.keras.applications.ResNet50()
x = np.random.randn(1, 224, 224, 3).astype(np.float32)
onnx_model_str = convert_tensorflow_to_onnx(
    model, x, str(save_path / "test_tf.onnx"),
)
```
And then you can trace the model execution by running:
```python
import shutil
from profiler.onnx_model import profile_onnx_model

profile = profile_onnx_model(
    onnx_model_str, np.random.randn(1, 224, 224, 3).astype(np.float32)
)
shutil.move(profile, str(save_path / "profile_tf_onnx.json"))
```
The profile will be saved in the `save_path` directory. You can visualize it using your favorite browser by opening the `profile_tf_onnx.json` file. Go to `chrome://tracing` and load the profile file.

## Profile PyTorch

