import setuptools

with open('requirements.txt','r',encoding = 'utf-8') as f:
    requirements = f.read().split("\n")

setuptools.setup(
    name="ez_transformers",
    version="0.0.0",
    description="ez_transformers",
    author="Philip Huang",
    url="https://github.com/p208p2002/ez_transformers",
    packages= setuptools.find_packages(),
    install_requires = requirements
    )