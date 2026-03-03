#!/usr/bin/env python3
"""
AutoDebias setup.py
自动去偏见工具包的安装配置文件
"""

from setuptools import setup, find_packages
import os

# 读取 README 文件
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "AutoDebias: 文本到图像扩散模型的偏见检测和缓解工具包"

# 读取 requirements.txt
def read_requirements():
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith("#"):
                    # 跳过内置模块
                    builtin_modules = [
                        "argparse", "pathlib", "logging", "json", "random", 
                        "gc", "os", "datetime", "io", "uuid", "typing", "dataclasses"
                    ]
                    if line not in builtin_modules:
                        requirements.append(line)
    except FileNotFoundError:
        # 如果没有 requirements.txt，提供基本依赖
        requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "pillow>=10.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "pyyaml>=6.0",
            "requests>=2.31.0",
        ]
    return requirements

setup(
    name="autodebias",
    version="0.1.0",
    author="AutoDebias Team",
    author_email="autodebias@example.com",
    description="自动检测和缓解文本到图像扩散模型中偏见的工具包",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autodebias",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "accelerate>=0.20.0",
        ],
        "full": [
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "pandas>=2.0.0",
            "datasets>=2.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "autodebias=autodebias.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="bias detection, diffusion models, AI fairness, image generation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/autodebias/issues",
        "Source": "https://github.com/yourusername/autodebias",
        "Documentation": "https://autodebias.readthedocs.io/",
    },
) 