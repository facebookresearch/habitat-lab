"""Utility functions for gym spaces: batch space and iterator."""
from collections import OrderedDict
from functools import singledispatch


from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple

# see https://github.com/openai/gym/blob/feea527a4fe66f48a077cf7e264ae60f86f745fa/gym/vector/utils/spaces.py for hints

@singledispatch
def unbatch_space(space: Space) -> Space:
    """Create a (unbatched) space, returns a single copy of a batched space.
    Example::
        >>> from gym.spaces import Box, Dict
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(6,3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(6,2,), dtype=np.float32)
        ... })
        >>> unbatch_space(space)
        Dict(position:Box(3,), velocity:Box(2,))
    Args:
        space: Space (e.g. the observation space) for a batched environment.
    Returns:
        Space (e.g. the observation space) for a single environment in the vectorized environment.
    Raises:
        ValueError: Cannot batch space that is not a valid :class:`gym.Space` instance
    """
    raise ValueError(
        f"Cannot batch space with type `{type(space)}`. The space must be a valid `gym.Space` instance."
    )



@unbatch_space.register(Box)
def _unbatch_space_box(space):
    return Box(low=space.low[0], high=space.high[0], dtype=space.dtype)


@unbatch_space.register(Tuple)
def _unbatch_space_tuple(space):
    return Tuple(
        tuple(unbatch_space(subspace) for subspace in space.spaces),
    )


@unbatch_space.register(Dict)
def _unbatch_space_dict(space):
    return Dict(
        OrderedDict(
            [
                (key, unbatch_space(subspace))
                for (key, subspace) in space.spaces.items()
            ]
        ),
    )
