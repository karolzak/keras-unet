from setuptools import setup
from setuptools import find_packages

setup(name='keras_unet',
      version='0.1',
      description='U-Net model for semantic segmentation implementation in Keras',
      url='http://github.com/karolzak/keras-unet',
      author='Karol Zak',
      author_email='karol.zak@hotmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires = ['keras'])