from setuptools import find_packages
from distutils.core import setup


INSTALL_REQUIRES = [
    # generic
    "numpy",
    "rospkg",
    "liegroups@git+https://github.com/mmattamala/liegroups",
    "kornia>=0.6.5",
    "torchmetrics",
    "pytorch_lightning>=1.6.5",
    "pytictac",
    "omegaconf",
    "hydra-core",
    "prettytable",
    "termcolor",
    "pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git",    
    "wget",    
    "wandb"
]
setup(
    name="unstruct_navigation",
    version="0.0.1",
    author="Jonas Frey, Matias Mattamala",
    author_email="jonfrey@ethz.ch, matias@robots.oex.ac.uk",
    packages=find_packages(),
    python_requires=">=3.7",
    description="A small example package",
    install_requires=[INSTALL_REQUIRES],
    include_package_data=True,
)
