import numpy as np
import onnxruntime as rt


def profile_onnx_model(onnx_model_str: str, input_data: np.ndarray):
    options = rt.SessionOptions()
    options.enable_profiling = True
    sess = rt.InferenceSession(onnx_model_str, options, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    sess.run(None, {input_name: input_data})
    prof_file = sess.end_profiling()
    print(prof_file)
    return prof_file
