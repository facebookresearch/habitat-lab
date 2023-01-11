
# Data Tools

Lightweight tools for creating hdf5 datasets and training on them with a variety of data.


## Installation

  - Open3d is an optional dependency. It is necessary for the scene visualization features in the `camera` and `point_cloud` submodules.


## Getting started

Take a look at [the simple write + read example](https://github.com/fairinternal/data_tools/blob/main/examples/simple_read_write.py).

To write examples to hdf5 file:
```python
from data_tools.writer import DataWriter

writer = DataWriter(filename)
writer.add_config(initial_pos=[x, y])
# Write multiple trajectoriesA
for data in trials:
	# Write each data point
	for x, y, z in data:
		writer.add_frame(pos=[x, y], res=[z])
	writer.write_trial()
```

Now you have an hdf5 file, which you might want to train a model on in pytorch. Essentially you just need to create a `get_datum()` function to read from hdf5:
```python
from data_tools.loader import DatasetBase

class SimpleDataset(DatasetBase):
    """Simple example data loader."""

    def get_datum(self, trial, idx):
        """Get a single training example given the index."""
        datum = {
                'pos': torch.FloatTensor(trial['pos'][idx]),
                'res': torch.FloatTensor(trial['res'][idx] / np.pi),
                }
        return datum


```
### Writing images

When writing images you need to use `add_img_frame(**data)`:
```
	rgb = # read rgb image from camera
	writer.add_img_frame(rgb=rgb)
```
