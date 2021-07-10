from collections import defaultdict
from typing import Dict, List, Optional

from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import generate_video


class BenchmarkRenderer(Benchmark):
    def __init__(
        self,
        config_paths: Optional[str],
        video_option: List[str],
        video_dir: str,
        writer=None,
    ) -> None:
        super().__init__(config_paths, False)
        self._video_option = video_option
        self._video_dir = video_dir
        self._writer = writer

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        rgb_frames = []
        should_render = len(self._video_option) > 0

        count_episodes = 0
        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset()
            if should_render:
                frame = observations_to_image(
                    observations, self._env.get_metrics()
                )
                rgb_frames.append(frame)

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)
                if should_render:
                    frame = observations_to_image(
                        observations, self._env.get_metrics()
                    )
                    rgb_frames.append(frame)

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            if should_render:
                generate_video(
                    video_option=self._video_option,
                    video_dir=self._video_dir,
                    images=rgb_frames,
                    episode_id=None,
                    # episode_id=self._env.episode_id,
                    checkpoint_idx=0,
                    metrics=metrics,
                    tb_writer=self._writer,
                )

            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
