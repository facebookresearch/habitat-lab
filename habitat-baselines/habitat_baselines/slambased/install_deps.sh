#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DIR1=$(pwd)
MAINDIR=$(pwd)/3rdparty
mkdir "${MAINDIR}"
cd "${MAINDIR}" || exit
#conda create -y -n "HandcraftedAgents" python=3.7
source activate HandcraftedAgents
conda install opencv -y
conda install pytorch torchvision -c pytorch -y
conda install -c conda-forge imageio -y
conda install ffmpeg -c conda-forge -y
cd "${MAINDIR}" || exit
mkdir eigen3
cd eigen3 || exit
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.5/eigen-3.3.5.tar.gz
tar -xzf eigen-3.3.5.tar.gz
cd eigen-3.3.5 || exit
mkdir build
cd build || exit
cmake .. -DCMAKE_INSTALL_PREFIX="${MAINDIR}"/eigen3_installed/
make install
cd "${MAINDIR}" || exit
wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip
unzip glew-2.1.0.zip
cd glew-2.1.0/ || exit
cd build || exit
cmake ./cmake  -DCMAKE_INSTALL_PREFIX="${MAINDIR}"/glew_installed
make -j4
make install
cd "${MAINDIR}" || exit
#pip install numpy --upgrade
rm Pangolin -rf
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin || exit
mkdir build
cd build || exit
cmake .. -DCMAKE_PREFIX_PATH="${MAINDIR}"/glew_installed/ -DCMAKE_LIBRARY_PATH="${MAINDIR}"/glew_installed/lib/ -DCMAKE_INSTALL_PREFIX="${MAINDIR}"/pangolin_installed
cmake --build .
cd "${MAINDIR}" || exit
rm ORB_SLAM2 -rf
rm ORB_SLAM2-PythonBindings -rf
git clone https://github.com/ducha-aiki/ORB_SLAM2
git clone https://github.com/ducha-aiki/ORB_SLAM2-PythonBindings
cd "${MAINDIR}"/ORB_SLAM2 || exit
sed -i "s,cmake .. -DCMAKE_BUILD_TYPE=Release,cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=${MAINDIR}/eigen3_installed/include/eigen3/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/ORBSLAM2_installed ,g" build.sh
ln -s "${MAINDIR}"/eigen3_installed/include/eigen3/Eigen "${MAINDIR}"/ORB_SLAM2/Thirdparty/g2o/g2o/core/Eigen
./build.sh
cd build || exit
make install
cd "${MAINDIR}" || exit
cd ORB_SLAM2-PythonBindings/src || exit
ln -s "${MAINDIR}"/eigen3_installed/include/eigen3/Eigen Eigen
cd "${MAINDIR}"/ORB_SLAM2-PythonBindings || exit
mkdir build
cd build || exit
CONDA_DIR="$(dirname $(dirname $(which conda)))"
CONDA_DIR=\"${CONDA_DIR}/envs/HandcraftedAgents/lib/python3.7/site-packages/\"
sed -i "s,lib/python3.5/dist-packages,${CONDA_DIR},g" ../CMakeLists.txt
cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.7m.so -DPYTHON_EXECUTABLE:FILEPATH=$(which python) -DCMAKE_LIBRARY_PATH="${MAINDIR}"/ORBSLAM2_installed/lib -DCMAKE_INCLUDE_PATH="${MAINDIR}"/ORBSLAM2_installed/include;"${MAINDIR}"/eigen3_installed/include/eigen3 -DCMAKE_INSTALL_PREFIX="${MAINDIR}"/pyorbslam2_installed
make
make install
cp "${MAINDIR}"/ORB_SLAM2/Vocabulary/ORBvoc.txt "${DIR1}"/data/
