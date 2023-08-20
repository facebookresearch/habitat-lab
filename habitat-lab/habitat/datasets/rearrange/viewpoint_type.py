from enum import Enum, auto


class ViewpointType(Enum):
    not_on_active_island = auto()
    too_far = auto()
    down_unnavigable = auto()
    outdoor_viewpoint = auto()
    low_visibility = auto()
    good = auto()
