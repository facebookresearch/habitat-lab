# HITL Framework Internals

This doc is aimed at developers looking to understand and contribute to the framework.

The HITL framework consists of the `habitat-hitl` Python library, example [desktop applications](../examples/hitl/), and our Unity-based [VR client](../examples/hitl/pick_throw_vr/README.md#vr).

## Library architecture
* The library is logically divided into a Habitat environment wrapper (`HitlDriver`) and a GUI component (`GuiApplication` and `ReplayGuiAppRenderer`).
* `HitlDriver`
    * It creates a `habitat.Env` instance.
    * Camera sensors are rendered by the `habitat.Env` instance in the usual way; see `self.obs = self.env.step(action)` in `HitlDriver.sim_update`.
    * This class is provided a `gui_input` object that encapsulates OS input (keyboard and mouse input). HITL apps should avoid making direct calls to PyGame, GLFW, and other OS-specific APIs.
    * `sim_update` returns a `post_sim_update_dict` that contains info needed by the app renderer (below). E.g. a gfx-replay keyframe and a camera transform for rendering, plus optional "debug images" to be shown to the user.
    * This class also has access to a `debug_line_render` instance for visualizing lines in the GUI (the lines aren't rendered into camera sensors). This access is somewhat hacky; future versions of HITL apps will likely convey lines via `post_sim_update_dict` instead of getting direct access to this object.
    * This class is provided an AppState which provides application-specific logic, for example, specific keyboard/mouse controls and specific on-screen help text.
* `GuiApplication`
    * manages the OS window (via GLFW for now), including OS-input-handling (keyboard and mouse) and updating the display (invoking the renderer).
* `ReplayGuiAppRenderer`
    * `ReplayGuiAppRenderer` is a renderer. It receives the `post_sim_update_dict` from `HitlDriver` and updates the OS window by rendering the scene from the requested camera pose.
    * `ReplayGuiAppRenderer` is application-agnostic. Whereas the behavior of `HitlDriver` is customized by the externally-provided AppState, no external customization of `ReplayGuiAppRenderer` is required. All HITL apps use our provided combination of 3D model rendering, 3D debug lines, 2D debug images, and 2D on-screen help text.

## Directory structure
```
habitat-hitl
    habitat_hitl
        _internal
            networking
        app_states
        config
        core
        environment
        scripts
    setup.py
examples
    hitl
        basic_viewer
        pick_throw_vr
        ...
```
* **habitat_hitl:** project root
* **habitat_hitl:** package root
* **_internal:** Non-public modules. Not intended for external use by HITL applications.
* **_internal/networking:** Non-public modules related to network communication with the VR client.
* **app_states:** Code specifically related to AppStates.
* **config:** Hydra configuration yaml files
* **core:** Here, we narrowly define "core" as library classes that don't depend on the wrapped Habitat environment. As a rule, if your class imports habitat-lab or habitat-baselines, it shouldn't go in here.
* **environment:** Everything related to the wrapped Habitat environment, including all 3D "scene helpers", task-related code, and agent-related code.
* **scripts:** Standalone scripts not generally used in HITL apps, e.g. a reference script for consuming collected data.
* **examples/hitl**: Example HITL apps; each has its own subfolder here.
