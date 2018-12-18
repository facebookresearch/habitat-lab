#!/usr/bin/env bash

# Change to actual downloading from internet once we switch to S3
echo "Downdload Matterport 3D Embodied Question Answering dataset v1"
mkdir -p data/datasets/eqa_mp3d_v1
ln -s /datasets01/mp3d/eqa/082318/08_23_full_data.json.gz  data/datasets/eqa_mp3d_v1/full_data.json.gz
ln -s /datasets01/mp3d/eqa/082318/08_23_full_test.h5  data/datasets/eqa_mp3d_v1/full_test.h5
ln -s /datasets01/mp3d/eqa/082318/08_23_full_train.h5  data/datasets/eqa_mp3d_v1/full_train.h5
ln -s /datasets01/mp3d/eqa/082318/08_23_full_val.h5  data/datasets/eqa_mp3d_v1/full_val.h5


echo "Download Matterport 3D test data with navmeshes for ESP"
mkdir -p data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.glb data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.navmesh data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.obj data/esp/test/
cp /private/home/maksymets/data/mp3d_esp_test/test.house data/esp/test/

echo "Download Matterport 3D data with navmeshes for ESP"
mkdir -p data/scene_datasets
ln -s /private/home/maksymets/data/mp3d/mp3d_esp  data/scene_datasets/mp3d
