from setuptools import setup, find_packages

setup(
    name="psytest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Add dependencies here, e.g., ["numpy"]
    author="Your Name",
    author_email="you@example.com",
    description="Short description of psytest",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/psytest",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
