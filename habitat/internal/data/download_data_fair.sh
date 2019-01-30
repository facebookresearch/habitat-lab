#!/usr/bin/env bash

# Change to actual downloading from internet once we switch to S3
echo "Downdload Matterport 3D datasets"
ln -s /private/home/maksymets/data/habitat_datasets  data/datasets

echo "Download Matterport 3D data with navmeshes for Habitat-Sim"
mkdir -p data/scene_datasets
ln -s /private/home/maksymets/data/mp3d/mp3d_esp  data/scene_datasets/mp3d

echo "Download Matterport 3D test data with navmeshes for Habitat-Sim"
mkdir -p data/habitat-sim/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.glb data/habitat-sim/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.navmesh data/habitat-sim/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.obj data/habitat-sim/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.house data/habitat-sim/test/

echo "Download Matterport 3D test data for multihouse test"
mkdir -p data/habitat-sim/multihouse-resources
cp /private/home/akadian/habitat-api-data/data/habitat-sim/multihouse_initializations.json data/habitat-sim
cp data/scene_datasets/mp3d/17DRP5sb8fy data/habitat-sim/multihouse-resources/ -R
cp data/scene_datasets/mp3d/1LXtFkjw3qL data/habitat-sim/multihouse-resources/ -R
cp data/scene_datasets/mp3d/1pXnuDYAj8r data/habitat-sim/multihouse-resources/ -R
cp data/scene_datasets/mp3d/29hnd4uzFmX data/habitat-sim/multihouse-resources/ -R
cp data/scene_datasets/mp3d/2azQ1b91cZZ data/habitat-sim/multihouse-resources/ -R

