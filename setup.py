from setuptools import setup, find_packages

setup(
    name="mmm-demonstrator",
    version="0.1.0",
    description="A comprehensive toolkit for demonstrating Marketing Mix Modeling concepts",
    author="Sameer M",
    author_email="sameerm1421999@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.22.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.10.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Business",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Office/Business :: Financial"
    ],
    python_requires=">=3.7",
)