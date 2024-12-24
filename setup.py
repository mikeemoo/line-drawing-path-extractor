from setuptools import setup, find_packages

setup(
    name="line-drawing-path-extractor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'Pillow>=9.0.0',
        'matplotlib>=3.5.0',
        'wandb>=0.15.0',
        'tqdm>=4.65.0',
    ],
) 
