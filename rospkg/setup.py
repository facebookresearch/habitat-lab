from setuptools import setup

install_requires = ["numpy", "empy", "catkin_pkg", "rospkg"]

setup(
    name="home_robot",
    version="1.0.0",
    packages=["home_robot"],
    package_dir={"": "../src"},
    install_requires=install_requires,
    zip_safe=False,
)
