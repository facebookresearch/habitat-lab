from setuptools import setup

install_requires = [
    "mrp",
    "pytest",
]

# For data_tools sub-module
data_tools_requires = ["h5py", "imageio", "pygifsicle"]
install_requires += data_tools_requires

setup(
    name="home_robot",
    version="1.0.0",
    packages=["home_robot"],
    package_dir={"": "src"},
    install_requires=install_requires,
    zip_safe=False,
)
