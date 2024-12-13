from setuptools import setup, find_packages

setup(
    name="auto_scroller",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for detecting hand gestures to control scrolling.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/auto_scroller",  # Update with your repo URL
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
