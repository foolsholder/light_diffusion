from setuptools import find_packages, setup

setup(
    name="diffusion",
    packages=find_packages(),
    include_package_data=True,
    version="0.1.0",
    author="Me",
    license="MIT",
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)