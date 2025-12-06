# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import cv2
import attr
import numpy as np
from gym import spaces

from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectNavigationTask,
    ObjectViewLocation,
)
try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

import clip
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavThinkingEpisode(NavigationEpisode):
    r"""ObjectGoalThinking Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )

@registry.register_sensor(name="ThoughtSensor")
class ThoughtSensor(Sensor):
    cls_uuid: str = "instruction"

    def __init__(self, sim, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: ObjectGoalNavThinkingEpisode,
        *args: Any,
        **kwargs: Any
    ):
        # Return a float32 array to be compatible with PyTorch
        # TODO: Use task.thought if available from ThinkAction
        return np.random.normal(size=(512,)).astype(np.float32)

@registry.register_task(name="ObjectNavThinking-v1")
class ObjectNavigationThinkingTask(ObjectNavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.thought: Optional[np.ndarray] = None
        self.last_image: Optional[np.ndarray] = None
        self.target_object: Optional[str] = None

    def reset(self, episode):
        import sys
        observations = super().reset(episode)
        self.thought = None

        # Store the target object category from the episode
        if hasattr(episode, 'object_category'):
            self.target_object = episode.object_category
            sys.stderr.write(f"[TASK_RESET] Target object set to: '{self.target_object}'\n")
            sys.stderr.flush()
        else:
            sys.stderr.write(f"[TASK_RESET] WARNING: Episode has no object_category attribute\n")
            sys.stderr.flush()

        # Store initial observation so it's available for think action
        self.last_image = observations["rgb"]
        sys.stderr.write(f"[TASK_RESET] Stored initial image with shape: {self.last_image.shape}\n")
        sys.stderr.flush()

        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        # Store current observation BEFORE processing action
        # so ThinkAction can access it during execution
        current_obs = self._sim.get_observations_at()
        if "rgb" in current_obs:
            self.last_image = current_obs["rgb"]

        # Now process the action (ThinkAction.step() will have access to last_image)
        observation = super().step(action, episode)

        # Update with the observation after action
        self.last_image = observation["rgb"]
        return observation


@registry.register_task_action
class ThinkAction(SimulatorTaskAction):
    name: str = "think"

    def __init__(self, *args: Any, **kwargs: Any):
        import sys
        super().__init__(*args, **kwargs)

        # Determine device
        self.device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        sys.stderr.write(f"[THINK_ACTION_INIT] Initializing ThinkAction with device={self.device}\n")
        sys.stderr.flush()
        sys.stderr.write(f"[THINK_ACTION_INIT] Loading CLIP model ViT-B/32...\n")
        sys.stderr.flush()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        sys.stderr.write(f"[THINK_ACTION_INIT] CLIP model loaded successfully on {self.device}\n")
        sys.stderr.flush()
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    def think(self, observation_image, target_object):
        import sys
        sys.stderr.write(f"\n[VLM_THINK] ========== VLM INFERENCE START ==========\n")
        sys.stderr.write(f"[VLM_THINK] Target object: '{target_object}'\n")
        sys.stderr.write(f"[VLM_THINK] Image shape: {observation_image.shape}\n")
        sys.stderr.flush()

        prompt_text = f"What is the next thing that the robot must do to find {target_object}"
        sys.stderr.write(f"[VLM_THINK] Prompt: {prompt_text}\n")
        sys.stderr.flush()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a guide for a robot that is navigating a home. Yourr job is to provide subgoals for the robot's navigation. You will be given an image and a target object to find. You have to reply with the next step for the robot, such as find the bedroom door."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": observation_image,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        sys.stderr.write(f"[VLM_THINK] Processing chat template...\n")
        sys.stderr.flush()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.vlm.device)

        sys.stderr.write(f"[VLM_THINK] Generating response (max_new_tokens=128)...\n")
        sys.stderr.flush()

        generated_ids = self.vlm.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        sys.stderr.write(f"[VLM_THINK] VLM Response: '{output_text[0] if output_text else 'EMPTY'}'\n")
        sys.stderr.write(f"[VLM_THINK] ========== VLM INFERENCE END ==========\n\n")
        sys.stderr.flush()

        return output_text

    def embed_thought(self, thought):
        import sys
        sys.stderr.write(f"[THINK_ACTION_EMBED] Embedding thought: '{thought}'\n")
        sys.stderr.flush()
        text = clip.tokenize([thought]).to(self.device)
        sys.stderr.write(f"[THINK_ACTION_EMBED] Tokenized text shape: {text.shape}\n")
        sys.stderr.flush()
        text_features = self.model.encode_text(text)
        sys.stderr.write(f"[THINK_ACTION_EMBED] CLIP features shape: {text_features.shape}, dtype: {text_features.dtype}\n")
        sys.stderr.flush()
        result = text_features.detach().cpu().numpy()[0]
        sys.stderr.write(f"[THINK_ACTION_EMBED] Final embedding shape: {result.shape}, dtype: {result.dtype}\n")
        sys.stderr.flush()
        return result

    def reset(self, task: ObjectNavigationThinkingTask, *args: Any, **kwargs: Any):
        import sys
        sys.stderr.write(f"[THINK_ACTION_RESET] Called\n")
        sys.stderr.flush()
        thought = "resetted thought text"
        task.thought = self.embed_thought(thought)
        sys.stderr.write(f"[THINK_ACTION_RESET] Set task.thought shape={task.thought.shape}, dtype={task.thought.dtype}\n")
        sys.stderr.flush()


    def step(self, task: ObjectNavigationThinkingTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        import sys
        sys.stderr.write(f"\n{'='*60}\n")
        sys.stderr.write(f"[THINK_ACTION_STEP] *** STEP CALLED ***\n")
        sys.stderr.flush()

        # Debug: Check what's available
        sys.stderr.write(f"[THINK_ACTION_STEP] Checking availability:\n")
        sys.stderr.write(f"[THINK_ACTION_STEP]   task.last_image: {task.last_image is not None} (shape={task.last_image.shape if task.last_image is not None else 'N/A'})\n")
        sys.stderr.write(f"[THINK_ACTION_STEP]   task.target_object: {task.target_object is not None} (value='{task.target_object}')\n")
        sys.stderr.write(f"[THINK_ACTION_STEP]   self.vlm: {self.vlm is not None}\n")
        sys.stderr.write(f"[THINK_ACTION_STEP]   self.processor: {self.processor is not None}\n")
        sys.stderr.flush()

        # Get the target object and current image
        if task.last_image is None:
            sys.stderr.write(f"[THINK_ACTION_STEP] ERROR: task.last_image is None!\n")
            sys.stderr.flush()
            thought = "Navigate forward to explore the environment"
        elif task.target_object is None:
            sys.stderr.write(f"[THINK_ACTION_STEP] ERROR: task.target_object is None!\n")
            sys.stderr.flush()
            thought = "Navigate forward to explore the environment"
        else:
            sys.stderr.write(f"[THINK_ACTION_STEP] All checks passed. Calling VLM...\n")
            sys.stderr.flush()

            # Call VLM to generate thought based on observation and target
            sys.stderr.write(f"[THINK_ACTION_STEP] About to call self.think()\n")
            sys.stderr.flush()

            vlm_output = self.think(task.last_image, task.target_object)

            sys.stderr.write(f"[THINK_ACTION_STEP] self.think() returned successfully\n")
            sys.stderr.flush()

            thought = vlm_output[0] if isinstance(vlm_output, list) else vlm_output
            sys.stderr.write(f"[THINK_ACTION_STEP] VLM generated thought: '{thought}'\n")
            sys.stderr.flush()

        # Embed the thought using CLIP
        sys.stderr.write(f"[THINK_ACTION_STEP] About to embed thought with CLIP\n")
        sys.stderr.flush()

        task.thought = self.embed_thought(thought)

        sys.stderr.write(f"[THINK_ACTION_STEP] Set task.thought shape={task.thought.shape}, dtype={task.thought.dtype}\n")
        sys.stderr.write(f"{'='*60}\n\n")
        sys.stderr.flush()
