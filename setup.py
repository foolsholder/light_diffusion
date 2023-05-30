from setuptools import find_packages, setup

setup(
    name="diffusion",
    packages=find_packages(),
    include_package_data=True,
    version="0.1.0",
    author="Me",
    license="MIT",
    install_requires=[
        "torch==2.0.0",
        "lightning==2.0.0",
        "transformers==4.29.2",
        "datasets",
        "wandb",
        "hydra-core==1.3.2"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)