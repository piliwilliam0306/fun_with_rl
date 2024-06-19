from setuptools import setup, find_packages

setup(
    name="fun_with_rl",
    version="0.0.1",
    author="Cheng-Wei Chen",
    author_email="piliwilliam0306@gmail.com",
    url="https://github.com/piliwilliam0306/fun_with_rl",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1"
        "numpy>=1.21",
        "torch>=1.13",
        "matplotlib",
        "imageio>=2.31.2"
    ],
    python_requires=">=3.7",
)
