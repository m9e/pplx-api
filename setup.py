from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='pplx_api',
    version='0.1.2',
    author='Kamiwaza AI',
    author_email='opensource@kamiwaza.ai',
    description='A Python client for the Perplexity AI API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kamiwaza-ai/pplx_api',
    packages=['pplx_api'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={'pplx_api': ['LICENSE']},
)
