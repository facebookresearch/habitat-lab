from teas.datasets.registration import dataset_registry, register_dataset, \
    make_dataset

register_dataset(
    id_dataset='Suncg-v0',
    entry_point='teas.datasets.houses:SuncgDataset')

register_dataset(
    id_dataset='MP3DEQA-v1',
    entry_point='teas.datasets.eqa.mp3d_eqa_dataset:MP3DDatasetV1')
