from setuptools import setup

install_requires = [
    "mrp",
    "pytest",
]

setup(
    name="home_robot",
    version=1.0,
    packages=["home_robot"],
    package_dir={"": "src"},
    install_requires=install_requires,
    zip_safe=False,
)
