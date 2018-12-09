from setuptools import setup, find_packages
from os import path
import os


setup(
    name='bikelearn',
    version=os.getenv('BIKELEARN_VERSION'),
    description='citibike learn codes',
    url='https://github.com/namoopsoo/learn-citibike',
    author='Michal Piekarczyk',
    packages=find_packages(exclude=['tests']),
    install_requires=['pandas', 'numpy', 'redis'], 
    )


