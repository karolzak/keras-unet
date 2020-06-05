from setuptools import setup
from setuptools import find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="keras-unet",
    version="0.1.1",
    description="Helper package with multiple U-Net implementations in Keras as well as useful utility tools helpful when working with image segmentation tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/karolzak/keras-unet",
    author="Karol Zak",
    author_email="karol.zak@hotmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[],
)
