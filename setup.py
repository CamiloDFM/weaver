# encoding: utf-8

from distutils.core import setup

setup(
    name='weaver',
    version='0.1.0',
    description='Word network builder',
    author='Camilo Pérez',
    url='https://github.com/CamiloDFM/weaver',
    packages=['weaver'],
    install_requires=[
        'nltk >=3.4.0, <4.0.0',
        'python-igraph >=0.7.0, <1.0.0',
    ],
    entry_points={
        'console_scripts': [
            'weaver = weaver.weaver:main',
        ],
    },
)