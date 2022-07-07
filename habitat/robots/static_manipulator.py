# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

@attr.s(auto_attribs=True, slots=True)
class StaticManipulatorParams:
    pass 

class StaticManipulator(RobotInterface):
    """Robot with a fixed base and controllable arm."""

    def __init__(
        self,
        params: StaticManipulatorParams,
    ):
        r"""Constructor
        """
        super().__init__()

    def reconfigure(self) -> None:
        pass

    def update(self) -> None:
        pass

    def reset(self) -> None:
        pass