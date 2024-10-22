from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='pplx',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python client for the Perplexity AI API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/pplx',
    packages=find_packages(exclude=['tests*']),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={'': ['LICENSE']},
    entry_points={
        'console_scripts': [
            'pplx=pplx.cli:main',
        ],
    },
)
