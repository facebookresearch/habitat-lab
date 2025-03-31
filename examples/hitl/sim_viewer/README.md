# Sim_viewer HITL application

Basic viewer application using only the `habitat-sim` Simulator, without an `habitat-lab` environment.

## Example launch command
Run the following command from the root `habitat-lab` directory:

```bash
python examples/hitl/sim_viewer/sim_viewer.py
```

## Configuration
As an example, the app is configured to load a specific [hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab) scene. You may load any Habitat scene by changing the `sim_viewer.dataset` and `sim_viewer.scene` fields in `config/sim_viewer.yaml`.
