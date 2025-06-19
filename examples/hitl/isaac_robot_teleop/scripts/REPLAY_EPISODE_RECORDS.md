# Processing Episode Records with the Isaac Robot Tele-op HITL application

TL;DR Data collected from XR interactions in the teleop application can be processed and replayed locally with the tools and scripts provided and detailed here.

### Format of the data
First download the data dump directory and decompress locally. The contents should be a set of directories with names like `0-1-2-3_1111_1748977752`.

These directories correspond to individual experiment sessions collected over one or many episodes at one time by a single user. The directory name contains this information split by `_` characters:
- episode indices: e.g. `0-1-2-3` means episodes 0 through 3 were attempted by the user.
- user id: e.g. `1111` is a user tag. These can be used to aggregate or cull data collected by an individual.
- timestamp: e.g. `1748977752` can be processed by `datetime` package into `GMT: Tuesday, June 3, 2025 7:09:12 PM`. We use this to sort out data collected by particular app versions and restrict analysis.

Within each of these session directories there is:
- a `session.json.gz` file containing aggregated high-level information about the session and episodes as well as config values used to collect the data.
- a set of `<episode_index>.json.gz` files with metadata about that episode and per-frame data such as robot states and object transforms. These files can be replayed in the application directly or parsed for additional information

### Processing Sessions and Episode Stats with `record_post_process.py`

The `record_post_process.py` file in this directory reads the session and episode files to compute statistics and parse out useful subsets.

For example, running:
```
python examples/hitl/isaac_robot_teleop/scripts/record_post_process.py --session-records-dir ../teleop_data/25_06_16/ --start-version v0.1
```
will read all data in the `teleop_data/25_06_16/` directory, cull out test users data, aggregate counts of sessions per user, restrict data to version v0.1+, compute success ratio, and print a list of successful episodes for replay.


### Running the replay application

Assuming you know the relative path to an episode record file (e.g. a path like `"../teleop_data/25_06_16/184-183-181-179_8200_1749675203/181.json.gz"` produced by the `record_post_process.py` script), the teleop application can be easliy configured to replay the record file live.

To configure the episode record for replay, set the `episode_record_filepath` config value in `config/replay_episode_record.yaml` like:
```yaml
episode_record_filepath: "../teleop_data/25_06_16/184-183-181-179_8200_1749675203/181.json.gz"
```

Then run the application like:
```
python examples/hitl/isaac_robot_teleop/isaac_robot_teleop.py --config-name replay_episode_record.yaml
```

#### UI:
When the application launches, you should see a view centered on the robot with debug frames indicating the XR state (headset + controllers) of the user. Text on the screen will indicate the current time and range of the replay.

You can:
- Hover over objects to see their handle in the top right screen text.
- click-hold mouse wheel or press `'r'` to rotate the camera.
- scroll mouse wheel to zoom
- `'f'` to unlock the camera from the robot (or vice versa)
    - with camera unlocked `wasd` will move the camera focus point
- `SPACE` to pause and start replay.
- `TAB` to reverse replay.
- `'p'` to unpause physics (NOTE: this will be buggy in the current version and isn't recommended now!)
