from setuptools import setup, find_packages

setup(
    name="line-drawing-path-extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'tensorboard>=2.13.0',
        'Pillow>=9.5.0',
    ],
    python_requires='>=3.8',
) 
