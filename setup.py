import setuptools


# Parse requirements
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Get long description
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()


# Setup
setuptools.setup(
    name="nn_training",
    version="1.0.0",
    packages=setuptools.find_packages(),
    author="Michał Szaknis & Wiktor Łazarski",
    description="Module used to train neural network with different optimization algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/L0czek/Training-Neural-Networks-with-Evolution-Alg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    include_package_data=True,
)
