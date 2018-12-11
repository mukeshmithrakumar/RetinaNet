import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = ['keras', 'tensorflow', 'keras-resnet', 'six', 'tensorflow', 'pandas', 'sklearn']

setuptools.setup(
    name="keras_retinanet",
    version="1.0.0",
    author="Mukesh Mithrakumar",
    author_email="mukesh.mithrakumar@jacks.sdstate.edu",
    description="Keras implementation of RetinaNet for object detection and visual relationship identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mukeshmithrakumar/RetinaNet",
    classifiers=(
        "Development Status :: 1.0.0.dev1 - Development release",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    keywords="sample setuptools development",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'retinanet_task = keras_retinanet.trainer.task:main',
            'retinanet_train = keras_retinanet.trainer.train:main',
            'retinanet_evaluate = keras_retinanet.trainer.evaluate:main',
        ]
    },
    python_requires='>=3',
)
