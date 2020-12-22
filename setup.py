import io
import os
import re

from setuptools import find_packages, setup

# Get the version from molbert/__init__.py
# Adapted from https://stackoverflow.com/a/39671214
this_directory = os.path.dirname(os.path.realpath(__file__))
version_matches = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open(f'{this_directory}/molbert/__init__.py', encoding='utf_8_sig').read(),
)
if version_matches is None:
    raise Exception('Could not determine MOLBERT version from __init__.py')
__version__ = version_matches.group(1)

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='molbert',
    version=__version__,
    author='BenevolentAI',
    author_email='chemval@benevolent.ai',
    description='Language modelling on chem/bio sequences',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BenevolentAI/MolBERT',
    packages=find_packages(),
    install_requires=[
        'flake8==3.8.4',
        'mypy==0.790',
        'pytest==5.3.2',
        'pytorch-lightning==0.8.4',
        'scikit-learn==0.21.3',
        'scipy==1.3.1',
        'transformers==3.5.1',
        'torch==1.4.0',
    ],
    include_package_data=True,
    zip_safe=True,
    classifiers=(
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ),
    entry_points={
        'console_scripts': [
            'molbert_smiles = molbert.apps.smiles:main',
            'molbert_finetune = molbert.apps.finetune:main',
        ],
    },
)
