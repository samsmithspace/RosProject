import os 
import io

from setuptools import setup, find_packages

# Package meta-data.
NAME = "pose_estimation"
DESCRIPTION = "Perform pose estimation on a single cube environment"
EMAIL = "perception@unity3d.com"
AUTHOR = "Unity Perception"
REQUIRES_PYTHON = ">=3.8"
VERSION = "0.1.0"

here = os.path.abspath(os.path.dirname(__file__))

# Define requirements
REQUIRED = [
    "torch>=1.7.0",
    "torchvision>=0.8.1",
    "pyyaml==5.3.1",
    "easydict==1.9",
    "tensorboardX==2.1",
    "click==7.1.2",
    "docopt==0.6.2",
]

# Optional requirements for development
EXTRAS = {
    "dev": [
        "jupyter==1.0.0",
        "pytest==5.4.3",
        "pytest-cov==2.10.0",
        "flake8==3.8.3",
        "isort==4.3.21",
        "black==19.10b0",
    ],
    "cloud": [
        "kfp==1.0.4",
        "google-cloud-storage",
    ]
}

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    entry_points={
        'console_scripts': [
            f"{NAME}={NAME}.cli:main"
        ]
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=REQUIRES_PYTHON,
)