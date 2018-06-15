import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    'numpy',
    'keras',
    'tensorflow'
]

setup(
	name='keras_menagerie',
	version='0.0.1',
	description='Keras Menagerie :: a collection of custom objects to expand Keras utilitaries',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
	url='https://github.com/therhappy/keras_menagerie',
	author='Timon Ther',
	author_email='email dot thert at googles famous email service',
	keywords='',
	license='MIT',
	packages=find_packages(),
        include_package_data=False,
        zip_safe=False,
        install_requires=install_requires
)