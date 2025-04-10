from setuptools import setup, find_packages

setup(
    name="psytest",
    version="0.1.0",
    packages=find_packages(),
    package_data={"psytest": ["data/*.csv"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy", "numba"],
    author="Jose Antunes Neto",
    author_email="joseparreiras@gmail.com",
    description="Python testing framework to test for the presence of bubbles.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joseparreiras/psytest",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
