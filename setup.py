from setuptools import setup
from setuptools import find_packages

setup(name='keras-unet',
      version='0.0.4',
      description='U-Net model for semantic segmentation implementation in Keras + useful utility functions for semantic segmentation tasks',
      url='http://github.com/karolzak/keras-unet',
      author='Karol Zak',
      author_email='karol.zak@hotmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires = ['keras'])