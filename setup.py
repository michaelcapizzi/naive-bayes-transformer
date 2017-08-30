from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='naive-bayes-transformer',
      version='0.0.3',
      description='A package implementing a Naive Bayes transformer (from Wang, Manning) and an applicable one-v-all classifier',
      author='Michael Capizzi',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='mcapizzi@email.arizona.edu',
      )
