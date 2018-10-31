'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='mnist_mlp',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='MNIST MLP keras model on Cloud ML Engine',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'pillow'],
      zip_safe=False)
