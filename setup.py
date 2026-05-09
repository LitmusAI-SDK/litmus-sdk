from setuptools import setup, find_packages

setup(
    name="litmus-sdk",
    version="0.1.0",
    description="LLM regression testing via embedding drift",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "chromadb>=1.4.1",
        "click>=8.1.8",
        "litellm>=1.81.13",
        "openai>=2.21.0",
        "pydantic>=2.12.5",
        "python-dotenv>=1.2.1",
        "PyYAML>=6.0.3",
        "scikit-learn>=1.8.0",
        "scipy>=1.17.0",
        "tomlkit>=0.13.2",
    ],
    extras_require={
        "dev": ["pytest>=9.0.2"],
    },
    entry_points={
        "console_scripts": [
            "litmus=litmus_sdk.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Posix",
    ],
)
