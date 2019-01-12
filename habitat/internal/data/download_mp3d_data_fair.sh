#!/usr/bin/env bash

# Change to actual downloading from internet once we switch to S3
echo "Downdload Matterport 3D datasets"
ln -s /private/home/maksymets/data/habitat_datasets  data/datasets

echo "Download Matterport 3D data with navmeshes for ESP"
mkdir -p data/scene_datasets
ln -s /private/home/maksymets/data/mp3d/mp3d_esp  data/scene_datasets/mp3d

echo "Download Matterport 3D test data with navmeshes for ESP"
mkdir -p data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.glb data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.navmesh data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.obj data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.house data/esp/test/

echo "Download Matterport 3D test data for multihouse test"
mkdir -p data/esp/multihouse-resources
cp /private/home/akadian/habitat-api-data/data/esp/multihouse_initializations.json data/esp
cp data/scene_datasets/mp3d/17DRP5sb8fy data/esp/multihouse-resources/ -R
cp data/scene_datasets/mp3d/1LXtFkjw3qL data/esp/multihouse-resources/ -R
cp data/scene_datasets/mp3d/1pXnuDYAj8r data/esp/multihouse-resources/ -R
cp data/scene_datasets/mp3d/29hnd4uzFmX data/esp/multihouse-resources/ -R
cp data/scene_datasets/mp3d/2azQ1b91cZZ data/esp/multihouse-resources/ -R

