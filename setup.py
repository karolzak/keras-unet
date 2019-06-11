from setuptools import setup
from setuptools import find_packages

setup(name='keras-unet',
      version='0.0.6',
      description='Helper package with multiple U-Net implementations in Keras as well as useful utility tools helpful when working with image segmentation tasks',
      url='http://github.com/karolzak/keras-unet',
      author='Karol Zak',
      author_email='karol.zak@hotmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires = ['keras'])