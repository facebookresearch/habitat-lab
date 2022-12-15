from typing import Any, List, Tuple

import torch


class HighLevelPolicy:
    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """
        raise NotImplementedError()
