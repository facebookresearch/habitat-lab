from typing import Any, Dict, List, Union

from habitat_baselines.common.tensor_dict import TensorDict


def _get_agent_k(name: str, agent_s: str) -> str:
    if name.startswith(agent_s):
        return name[len(agent_s) :]
    else:
        return name


def filter_agent_names(
    names: Union[Dict[str, Any], TensorDict], agent_idx: int
) -> Union[Dict[str, Any], TensorDict]:
    was_td_dict = isinstance(names, TensorDict)
    agent_s = f"agent_{agent_idx}_"
    ret = {
        _get_agent_k(k, agent_s): v
        for k, v in names.items()
        if agent_s in k or not k.startswith("agent_")
    }
    if was_td_dict:
        return TensorDict(ret)
    else:
        return ret


def get_agent_name(k, agent_i):
    return f"agent_{agent_i}_{k}"


def add_agent_names(source_dict, dest_dict, agent_i):
    for k, v in source_dict.items():
        dest_dict[f"agent_{agent_i}_{k}"] = v
