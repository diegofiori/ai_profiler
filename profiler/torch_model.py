import torch
from torch.profiler import profile, record_function, ProfilerActivity


def profile_torch_model(
        model: torch.nn.Module, input_data: torch.Tensor, prof_file: str = None,
):
    if prof_file is None:
        prof_file = "torch_profile.json"
    if next(model.parameters()).is_cuda:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            _ = model(input_data)
    prof.export_chrome_trace(prof_file)
    return prof_file
