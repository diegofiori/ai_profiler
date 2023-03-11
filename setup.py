from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "torch>=1.7.0",
    "torchvision>=0.8.1",
    "onnx>=1.8.0",
    "onnxruntime>=1.6.0",
    "tensorflow>=2.3.0",
    "tensorboard>=2.3.0",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf8")

setup(
    name="profiler",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
