import transformers_konlpy
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requires = [line.strip() for line in f]

setup(
    name="transformers_konlpy",
    version=transformers_konlpy.__version__,
    author=transformers_konlpy.__author__,
    author_email='soy.lovit@gmail.com',
    url='https://github.com/lovit/transformers_konlpy',
    description="Utils for training masked language model with huggingface transformers and KoNLPy",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requires,
    packages=find_packages()
)
