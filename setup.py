from setuptools import setup, find_packages
from os import path



setup(
    name='bikelearn',
    version='0.1.0',
    description='citibike learn codes',
    url='https://github.com/namoopsoo/learn-citibike',
    author='Michal Piekarczyk',
    packages=find_packages(exclude=['tests']),
    install_requires=['pandas', 'numpy'], 
    )


