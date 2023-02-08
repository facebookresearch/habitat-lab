from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from habitat.core.env import Env
from habitat_sim.utils.common import d3_40_colors_rgb


class SemanticCategoryMapping(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        pass

    @abstractmethod
    def reset_instance_id_to_category_id(self, env: Env):
        pass

    @property
    @abstractmethod
    def instance_id_to_category_id(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def map_color_palette(self):
        pass

    @property
    @abstractmethod
    def frame_color_palette(self):
        pass

    @property
    @abstractmethod
    def categories_legend_path(self):
        pass

    @property
    @abstractmethod
    def num_sem_categories(self):
        pass


# ----------------------------------------------------
# Mukul 33 Indoor Categories
# ----------------------------------------------------

mukul_33categories_indexes = {
    1: "alarm_clock",
    2: "bathtub",
    3: "bed",
    4: "book",
    5: "bottle",
    6: "bowl",
    7: "cabinet",
    8: "carpet",
    9: "chair",
    10: "chest_of_drawers",
    11: "couch",
    12: "cushion",
    13: "drinkware",
    14: "fireplace",
    15: "fridge",
    16: "laptop",
    17: "oven",
    18: "picture",
    19: "plate",
    20: "potted_plant",
    21: "shelves",
    22: "shoes",
    23: "shower",
    24: "sink",
    25: "stool",
    26: "table",
    27: "table_lamp",
    28: "toaster",
    29: "toilet",
    30: "tv",
    31: "vase",
    32: "wardrobe",
    33: "washer_dryer",
}
mukul_33categories_padded = (
    ["."] + [mukul_33categories_indexes[i] for i in range(1, 34)] + ["other"]
)

mukul_33categories_legend_path = "/nethome/mkhanna37/flash1/language-rearrangement/habitat-lab/habitat-baselines/habitat_baselines/ml/visualizer/fp_semantics_legend.png"

mukul_33categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:34].flatten()
)
mukul_33categories_frame_color_palette = mukul_33categories_color_palette + [
    255,
    255,
    255,
]

mukul_33categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        *[x / 255.0 for x in mukul_33categories_color_palette],
    ]
]


class FloorplannertoMukulIndoor(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self.floorplanner_goal_id_to_goal_name = mukul_33categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.floorplanner_goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env: Env):
        # Identity everywhere except index 0 mapped to 34
        self._instance_id_to_category_id = torch.arange(
            self.num_sem_categories
        )
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> torch.Tensor:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return mukul_33categories_map_color_palette

    @property
    def frame_color_palette(self):
        return mukul_33categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return mukul_33categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 to 33 are semantic categories, 34 is "other/misc"
        return 35


# ----------------------------------------------------
# Sriram 16 Object Categories
# ----------------------------------------------------

sriram_16categories_indexes = {
    0: "action_figure",
    1: "basket",
    2: "book",
    3: "bowl",
    4: "cup",
    5: "dishtowel",
    6: "hat",
    7: "medicine_bottle",
    8: "pencil_case",
    9: "plate",
    10: "shoe",
    11: "soap_dish",
    12: "sponge",
    13: "stuffed_toy",
    14: "sushi_mat",
    15: "tape",
}

sriram_16categories_padded = (
    ["."] + [sriram_16categories_indexes[i] for i in range(0, 16)] + ["other"]
)

sriram_16categories_legend_path = "/nethome/mkhanna37/flash1/language-rearrangement/habitat-lab/habitat-baselines/habitat_baselines/ml/visualizer/fp_rearrange_semantics_legend.png"

sriram_16categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:18].flatten()
)
sriram_16categories_frame_color_palette = sriram_16categories_color_palette + [
    255,
    255,
    255,
]

sriram_16categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        *[x / 255.0 for x in sriram_16categories_color_palette],
    ]
]


class FloorplannertoSriramObjects(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self.floorplanner_goal_id_to_goal_name = sriram_16categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.floorplanner_goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env: Env):
        # Identity everywhere except index 0 mapped to 17
        self._instance_id_to_category_id = torch.arange(
            self.num_sem_categories
        )
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> torch.Tensor:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return sriram_16categories_map_color_palette

    @property
    def frame_color_palette(self):
        return sriram_16categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return sriram_16categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 to 16 are semantic categories"
        return 18


# ----------------------------------------------------
# (Single) Goal category mapping
# ----------------------------------------------------

goal_category_index = {1: "object_category", 2:"start_recep_category",  3:"goal_recep_category"}

goal_category_index_padded = (
    ["."] + [goal_category_index[i] for i in range(1, 2)] + ["other"]
)

goal_category_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:4].flatten()
)
goal_category_frame_color_palette = goal_category_color_palette + [
    255,
    255,
    255,
]

goal_category_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        *[x / 255.0 for x in sriram_16categories_color_palette],
    ]
]


class GoalObjectMapping(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        sriram_objects = FloorplannertoSriramObjects()
        self.floorplanner_goal_id_to_goal_name = (
            sriram_objects.floorplanner_goal_id_to_goal_name
        )
        # self.floorplanner_goal_id_to_goal_name = goal_category_index
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:

        return (goal_id, self.floorplanner_goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env: Env):
        self._instance_id_to_category_id = torch.arange(
            self.num_sem_categories
        )
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> torch.Tensor:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return goal_category_map_color_palette

    @property
    def frame_color_palette(self):
        return goal_category_frame_color_palette

    @property
    def categories_legend_path(self):
        return None

    @property
    def num_sem_categories(self):
        return 5


# ----------------------------------------------------
# Long-tail Indoor Categories
# ----------------------------------------------------

receptacle_categories = [
    "armchair",
    "bar_chair",
    "bathroom_cabinet",
    "bathtub",
    "beanbag_chair",
    "bed",
    "bench",
    "bidet",
    "bookshelf",
    "cabinet",
    "chair",
    "clothes_hanger",
    "coat_rack",
    "coffee_machine",
    "coffee_table",
    "copier",
    "couch",
    "countertop",
    "desk",
    "dining_chair",
    "dining_table",
    "dish_rack",
    "dishwasher",
    "drawer",
    "end_table",
    "fireplace",
    "garbage_can",
    "hand_towel_holder",
    "highchair",
    "ironing_board",
    "kitchen_cabinet",
    "laundry_hamper",
    "massage_table",
    "microwave",
    "mixer",
    "nightstand",
    "office_chair",
    "ottoman",
    "oven",
    "pantry",
    "pool_table",
    "printer",
    "rack",
    "recycling_bin",
    "refrigerator",
    "safe",
    "shelving",
    "shower",
    "sink",
    "stool",
    "storage_container",
    "stove",
    "table",
    "television_stand",
    "toilet_paper_holder",
    "towel_holder",
    "washing_machine",
]
carryable_categories = [
    "alarm_clock",
    "aluminum_foil",
    "apple",
    "apron",
    "ashtray",
    "backpack",
    "bag",
    "ball",
    "baseball_bat",
    "basket",
    "basket_ball",
    "bath_mat",
    "bathrobe",
    "bedside_lamp",
    "beer",
    "belt",
    "blanket",
    "book",
    "bottle",
    "bowl",
    "box",
    "bread",
    "briefcase",
    "broom",
    "brush",
    "bucket",
    "butter_knife",
    "camera",
    "candle",
    "cd",
    "cell_phone",
    "chandelier",
    "chest",
    "cloth",
    "clothes",
    "coat",
    "coffee_kettle",
    "cosmetics",
    "credit_card",
    "cup",
    "cushion",
    "cutting_board",
    "detergent_bottle",
    "dining_table_mat",
    "dish_sponge",
    "doll",
    "dress",
    "drum",
    "dustpan",
    "egg",
    "exercise_mat",
    "firewood",
    "footrest",
    "fork",
    "fruit_bowl",
    "garbage_bag",
    "glass",
    "globe",
    "grocery_bag",
    "guitar",
    "guitar_case",
    "hair_dryer",
    "hand_towel",
    "hanger",
    "hat",
    "headphones",
    "jacket",
    "jar",
    "kettle",
    "key_chain",
    "keyboard",
    "knife",
    "ladle",
    "laptop",
    "laundry_detergent",
    "lettuce",
    "magazine",
    "mouse",
    "mug",
    "newspaper",
    "pan",
    "paper",
    "pencil",
    "pepper_shaker",
    "pillow",
    "pitcher",
    "plate",
    "platter",
    "plunger",
    "pot",
    "potato",
    "projector",
    "purse",
    "radio",
    "remote_control",
    "rope",
    "salt_shaker",
    "scale",
    "scarf",
    "scrub_brush",
    "shampoo",
    "sheet",
    "shirt",
    "shoe",
    "skateboard",
    "soap_bar",
    "soap_bottle",
    "soap_dish",
    "soda_can",
    "spatula",
    "speaker",
    "spoon",
    "spray_bottle",
    "stuffed_animal",
    "suitcase",
    "table_lamp",
    "tablecloth",
    "tablet",
    "teapot",
    "teddy_bear",
    "telephone",
    "tennis_racket",
    "tissue_box",
    "toilet_brush_holder",
    "toilet_paper",
    "tomato",
    "toolbox",
    "towel",
    "toy",
    "tray",
    "umbrella",
    "vase",
    "watch",
    "water_dispenser",
    "wine_bottle",
]
located_topography_categories = [
    "air_conditioner",
    "bathroom_stall",
    "bicycle",
    "blackboard",
    "clock",
    "desktop",
    "display_case",
    "door_mat",
    "fire_alarm",
    "fire_extinguisher",
    "floor_lamp",
    "fruit_bowl",
    "grill",
    "gym_equipment",
    "jacuzzi",
    "ladder",
    "monitor",
    "piano",
    "ping_pong_table",
    "plant",
    "potted_plant",
    "projector_screen",
    "radiator",
    "television",
    "toaster",
    "toilet",
    "toilet_paper_dispenser",
    "treadmill",
    "urinal",
    "vacuum_cleaner",
    "water_cooler",
    "whiteboard",
]
generic_topography_categories = [
    "arch",
    "blinds",
    "curtain",
    "door",
    "door_frame",
    "fan",
    "heater",
    "lamp",
    "light",
    "mirror",
    "outlet",
    "picture",
    "pillar",
    "poster",
    "rug",
    "stair_railing",
    "stairs",
    "statue",
    "tapestry",
    "thermostat",
    "window",
    "window_frame",
]
object_part_categories = [
    "door_handle",
    "door_knob",
    "faucet",
    "lightswitch",
    "shower_curtain",
    "shower_door",
    "shower_handle",
    "shower_head",
    "sink_basin",
    "stove_knob",
]
long_tail_indoor_categories = (
    receptacle_categories
    + carryable_categories
    + located_topography_categories
    + generic_topography_categories
    + object_part_categories
    + ["other"]
)
hm3d_to_longtail_indoor = {
    "wall": "other",
    "door": "rack",
    "ceiling": "other",
    "floor": "other",
    "picture": "door",
    "window": "other",
    "chair": "other",
    "frame": "other",
    "remove": "statue",
    "pillow": "ironing_board",
    "object": "other",
    "light": "other",
    "cabinet": "other",
    "curtain": "scale",
    "table": "cabinet",
    "plant": "other",
    "decoration": "wine_bottle",
    "lamp": "other",
    "mirror": "other",
    "towel": "other",
    "sink": "other",
    "shelf": "gym_equipment",
    "couch": "chest",
    "dining": "window_frame",
    "bed": "other",
    "nightstand": "picture",
    "toilet": "other",
    "sofa": "gym_equipment",
    "pillar": "cabinet",
    "handrail": "other",
    "stair": "other",
    "stool": "other",
    "armchair": "statue",
    "kitchen": "tray",
    "vase": "other",
    "cushion": "bathroom_cabinet",
    "tv": "drawer",
    "unknown": "end_table",
    "pot": "chair",
    "desk": "other",
    "roof": "picture",
    "box": "other",
    "shower": "window_frame",
    "coffee": "other",
    "countertop": "other",
    "bench": "blanket",
    "trashcan": "couch",
    "fireplace": "other",
    "clothes": "door",
    "bathtub": "sink_basin",
    "duct": "bench",
    "bath": "other",
    "book": "vase",
    "beam": "alarm_clock",
    "vent": "other",
    "faucet": "fireplace",
    "photo": "lamp",
    "paper": "other",
    "counter": "fireplace",
    "fan": "other",
    "step": "ashtray",
    "wash": "sink",
    "/otherroom": "other",
    "washbasin": "other",
    "railing": "door_knob",
    "shelving": "bucket",
    "statue": "stairs",
    "dresser": "other",
    "rug": "other",
    "ottoman": "monitor",
    "bottle": "picture",
    "office": "blinds",
    "refrigerator": "other",
    "bookshelf": "cosmetics",
    "end": "other",
    "wardrobe": "basket",
    "toiletry": "other",
    "pipe": "towel_holder",
    "monitor": "soap_dish",
    "stand": "other",
    "drawer": "other",
    "container": "exercise_mat",
    "switch": "other",
    "skylight": "rack",
    "purse": "picture",
    "doorway": "blinds",
    "paneling": "picture",
    "basket": "other",
    "closet": "other",
    "arch": "other",
    "chandelier": "rack",
    "oven": "stair_railing",
    "clock": "other",
    "footstool": "television_stand",
    "stove": "other",
    "washing": "other",
    "machine": "potted_plant",
    "rack": "shelving",
    "fire": "nightstand",
    "alarm": "other",
    "bin": "other",
    "chest": "tray",
    "microwave": "potted_plant",
    "blinds": "armchair",
    "bowl": "other",
    "tree": "gym_equipment",
    "vanity": "other",
    "tissue": "light",
    "plate": "other",
    "shoe": "other",
    "heater": "other",
    "bedframe": "coffee_table",
    "headboard": "other",
    "post": "shelving",
    "swivel": "other",
    "pedestal": "printer",
    "fence": "other",
    "pew": "other",
    "bucket": "other",
    "decorative": "sheet",
    "mask": "shower",
    "candle": "jar",
    "flowerpot": "door_mat",
    "speaker": "other",
    "seat": "toy",
    "sign": "door",
    "air": "other",
    "conditioner": "monitor",
    "rod": "other",
    "clutter": "other",
    "extinguisher": "mirror",
    "mat": "cosmetics",
    "sculpture": "other",
    "printer": "end_table",
    "telephone": "other",
    "molding": "other",
    "handbag": "arch",
    "blanket": "gym_equipment",
    "dispenser": "other",
    "handle": "potted_plant",
    "/outside": "mug",
    "screen": "other",
    "showerhead": "washing_machine",
    "barricade": "desk",
    "soap": "other",
    "banister": "other",
    "keyboard": "statue",
    "thermostat": "scale",
    "radiator": "garbage_can",
    "island": "copier",
    "dryer": "other",
    "panel": "end_table",
    "glass": "other",
    "dishwasher": "towel_holder",
    "cup": "other",
    "bathroom": "other",
    "ladder": "other",
    "garage": "other",
    "hat": "cabinet",
    "of": "other",
    "drawers": "bench",
    "exit": "other",
    "side": "other",
    "piano": "cabinet",
    "board": "other",
    "archway": "cabinet",
    "rope": "floor_lamp",
    "ball": "laundry_hamper",
    "gym": "fireplace",
    "equipment": "tissue_box",
    "hanger": "other",
    "easy": "other",
    "lounge": "bottle",
    "furniture": "box",
    "carpet": "other",
    "food": "other",
    "ridge": "other",
    "candlestick": "other",
    "computer": "other",
    "sconce": "other",
    "scale": "other",
    "baseboard": "toilet",
    "bag": "other",
    "laptop": "bathroom_cabinet",
    "treadmill": "dress",
    "staircase": "water_dispenser",
    "guitar": "door",
    "fixture": "other",
    "display": "table",
    "case": "gym_equipment",
    "exercise": "other",
    "holder": "cabinet",
    "basin": "shower",
    "bar": "other",
    "tray": "window",
    "urn": "other",
    "shade": "toilet",
    "grass": "other",
    "pool": "toaster",
    "coat": "other",
    "cloth": "other",
    "water": "other",
    "cooler": "other",
    "ledge": "arch",
    "utensil": "box",
    "shrubbery": "shelving",
    "teapot": "coffee_machine",
    "locker": "other",
    "ornament": "refrigerator",
    "bidet": "suitcase",
    "window/door": "laundry_hamper",
    "stuffed": "door_frame",
    "animal": "other",
    "fencing": "tablet",
    "lampshade": "exercise_mat",
    "bust": "other",
    "car": "other",
    "figure": "display_case",
    "set": "mouse",
    "brush": "other",
    "doll": "jar",
    "drum": "gym_equipment",
    "dress": "shelving",
    "whiteboard": "jacuzzi",
    "opener": "fireplace",
    "range": "window_frame",
    "hood": "toilet",
    "easel": "stool",
    "fruit": "other",
    "appliance": "other",
    "candelabra": "other",
    "toy": "stair_railing",
    "top": "other",
    "highchair": "other",
    "footrest": "clothes",
    "dish": "bed",
    "altar": "picture",
    "place": "table",
    "sheet": "chair",
    "wood": "grill",
    "robe": "cabinet",
    "stall": "dining_table",
    "plush": "door_knob",
    "bush": "other",
    "valence": "other",
    "control": "grill",
    "tap": "arch",
    "shampoo": "storage_container",
    "massage": "shelving",
    "knob": "fan",
    "stopper": "other",
    "bulletin": "other",
    "electric": "other",
    "wire": "other",
    "casing": "door",
    "storage": "other",
    "maker": "other",
    "projector": "statue",
    "cubby": "washing_machine",
    "balcony": "couch",
    "/w": "other",
    "pan": "other",
    "luggage": "other",
    "hamper": "other",
    "trinket": "other",
    "backsplash": "other",
    "chimney": "door_frame",
    "person": "other",
    "tablet": "other",
    "smoke": "other",
    "weight": "pillar",
    "bedpost": "other",
    "file": "gym_equipment",
    "umbrella": "bedside_lamp",
    "laundry": "other",
    "jar": "urinal",
    "bike": "other",
    "hose": "other",
    "dormer": "firewood",
    "power": "rug",
    "breaker": "picture",
    "detector": "pillow",
    "jacuzzi": "other",
    "backpack": "stove",
    "hook": "oven",
    "elevator": "mirror",
    "tool": "recycling_bin",
    "recliner": "countertop",
    "recessed": "other",
    "tank": "other",
    "toaster": "other",
    "landing": "door_frame",
    "hunting": "bicycle",
    "trophy": "copier",
    "motion": "kitchen_cabinet",
    "can": "other",
    "paint": "other",
    "medicine": "other",
    "sensor": "lightswitch",
    "cart": "door_frame",
    "slab": "other",
    "bean": "clock",
    "pole": "other",
    "canister": "other",
    "pitcher": "other",
    "podium": "mirror",
    "grill": "other",
    "tapestry": "other",
    "doorknob": "other",
    "vacuum": "detergent_bottle",
    "cleaner": "other",
    "comforter": "other",
    "shirt": "other",
    "dressing": "jacket",
    "beside": "nightstand",
    "curb": "other",
    "support": "other",
    "globe": "chair",
    "pantry": "other",
    "skateboard": "other",
    "cabin": "light",
    "chaise": "door",
    "flower": "curtain",
    "and": "other",
    "chairs": "blinds",
    "cross": "water_dispenser",
    "sliding": "other",
    "cosmetics": "bench",
    "kettle": "other",
    "junk": "other",
    "stationery": "office_chair",
    "gate": "other",
    "safe": "other",
    "ventilation": "other",
    "firewood": "statue",
    "row": "other",
    "theater": "other",
    "toolbox": "speaker",
    "security": "stair_railing",
    "camera": "nightstand",
    "mantle": "other",
    "skirting": "stairs",
    "tile": "other",
    "outlet": "other",
    "doorframe": "other",
    "hedge": "outlet",
    "hand": "other",
    "christmas": "window",
    "column": "other",
    "casket": "blackboard",
    "centerpiece": "lamp",
    "bedside": "other",
    "item": "other",
    "fountain": "other",
    "soffit": "other",
    "urinal": "other",
    "barrel": "shelving",
    "roll": "other",
    "portrait": "chair",
    "pouffe": "other",
    "concrete": "other",
    "block": "bar_chair",
    "liner": "other",
    "patio": "other",
    "folding": "sink",
    "recycle": "shelving",
    "rafter": "chair",
    "stage": "other",
    "sprinkler": "monitor",
    "soil": "other",
    "bicycle": "other",
    "partition": "table",
    "led": "desktop",
    "under": "other",
    "books": "picture",
    "giraffe": "door_frame",
    "grandfather": "other",
    "jewelry": "other",
    "bottles": "other",
    "wine": "stair_railing",
    "dog": "briefcase",
    "valance": "door",
    "radio": "other",
    "seats": "other",
    "towels": "other",
    "sauna": "desktop",
    "fume": "cloth",
    "cupboard": "vacuum_cleaner",
    "mouse": "other",
    "boiler": "other",
    "hearth": "other",
    "round": "shower",
    "doorstep": "other",
    "binder": "chandelier",
    "runner": "other",
    "cubicle": "other",
    "overhang": "door",
    "bathrobe": "other",
    "doormat": "other",
    "jacket": "shelving",
    "trim": "other",
    "reflection": "stairs",
    "pulpit": "other",
    "armchairs": "door",
    "fish": "other",
    "objects": "tray",
    "lintel": "other",
    "lighting": "cloth",
    "freezer": "table",
    "extractor": "footrest",
    "platform": "cabinet",
    "hot": "projector",
    "tub": "other",
    "grab": "bathtub",
    "detail": "other",
    "whine": "cabinet",
    "painting": "bench",
    "buffet": "other",
    "billow": "other",
    "stairs": "other",
    "calendar": "other",
    "dome": "other",
    "poll": "shower_handle",
    "wet": "statue",
    "stovetop": "desktop",
    "vending": "door",
    "liquid": "stairs",
    "small": "other",
    "table/stand": "paper",
    "shutters": "other",
    "stone": "other",
    "tripod": "door_frame",
    "wreath": "other",
    "hinge": "cabinet",
    "french": "other",
    "night": "other",
    "picure": "chair",
    "stick": "stool",
    "fluorescent": "other",
    "trellis": "beanbag_chair",
    "dartboard": "other",
    "dirt": "other",
    "base": "bathtub",
    "chemical": "apron",
    "misc": "ironing_board",
    "cover": "other",
    "reading": "other",
    "steps": "end_table",
    "sideboard": "other",
    "separator": "chair",
    "vessel": "statue",
    "skirt": "couch",
    "rocking": "other",
    "blackboard": "chair",
    "closest": "other",
    "area": "other",
    "scroll": "other",
    "foot": "desk",
    "button": "bathtub",
    "art/clutter": "other",
    "shovel": "toilet_paper",
    "yard": "other",
    "semi": "other",
    "bouquet": "other",
    "corner": "other",
    "plunger": "potted_plant",
    "belt": "other",
    "sewing": "other",
    "water/cold": "other",
    "barbecue": "other",
    "cutting": "curtain",
    "soapbox": "mixer",
    "stuff": "statue",
    "copier": "bench",
    "picture/window": "shelving",
    "throne": "platter",
    "socket": "other",
    "art": "shelving",
    "tabletop": "bathrobe",
    "trash": "cushion",
    "l-shaped": "clothes",
    "cardboard": "towel",
    "hanging": "fire_extinguisher",
    "stand/small": "other",
    "indent": "stairs",
    "towel/curtain": "other",
    "iron": "other",
    "shelf/cabinet": "nightstand",
    "accessory": "other",
    "circular": "other",
    "dustpan": "other",
    "oil": "other",
    "scaffolding": "other",
    "baluster": "refrigerator",
    "leg": "stairs",
    "rest": "garbage_can",
    "/otheroom": "other",
    "hole": "pillow",
    "ping": "other",
    "pong": "towel",
    "hutch": "refrigerator",
    "foliage": "picture",
    "circle": "plant",
    "record": "window",
    "player": "window",
    "doorpost": "other",
    "briefcase": "towel_holder",
    "energy": "pillow",
    "beanbag": "door",
    "plumbing": "other",
    "moose": "toy",
    "head/sculpture/hunting": "toilet",
    "flowerbed": "other",
    "antique": "other",
    "rock": "light",
    "caddy": "faucet",
    "media": "other",
    "console": "other",
    "risers": "other",
    "for": "other",
    "seating": "other",
    "branch": "other",
    "tiled": "vase",
    "bedroom": "other",
    "hearst": "basket",
    "condiment": "other",
    "piping": "window",
    "shelves": "other",
    "watch": "other",
    "rail": "other",
    "fuse": "other",
    "knife": "other",
    "aquarium": "fire_alarm",
    "wheelbarrow": "armchair",
    "rods/table": "other",
    "gable": "other",
    "balustrade": "other",
    "three": "other",
    "rocky": "other",
    "ground": "other",
    "backrest": "other",
    "basketball": "toy",
    "hoop": "other",
    "spice": "other",
    "cluttered": "other",
    "transformer": "door",
    "gift": "table",
    "stack": "other",
    "papers": "shower",
    "holy": "other",
    "arcade": "other",
    "game": "fireplace",
    "-": "shower_head",
    "probably": "couch",
    "part": "other",
    "--": "other",
    "maybe": "other",
    "compound": "other",
    "plug": "other",
    "magazine": "floor_lamp",
    "rolling": "statue",
    "pin": "other",
    "sink/basin": "nightstand",
    "boarder": "other",
    "perfume": "other",
    "heat": "cabinet",
    "pump": "exercise_mat",
    "columned": "shower_curtain",
    "perimeter": "other",
    "shrine": "other",
    "canvas": "other",
    "art/man": "chair",
    "credenza": "other",
    "artwork": "other",
    "playpen": "other",
    "makeup": "other",
    "plant/art": "lamp",
    "bot": "door_frame",
    "horse": "other",
    "lower": "television_stand",
    "pack": "other",
    "pathway": "other",
    "tablecloth": "other",
    "tarp": "other",
    "clothing": "other",
}


class HM3DtoLongTailIndoor(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self.hm3d_goal_id_to_longtail_goal_name = {
            0: "chair",
            1: "bed",
            2: "potted plant",
            3: "toilet",
            4: "tv",
            5: "couch",
        }
        self.hm3d_goal_id_to_longtail_goal_id = {
            0: long_tail_indoor_categories.index("chair"),
            1: long_tail_indoor_categories.index("bed"),
            2: long_tail_indoor_categories.index("potted_plant"),
            3: long_tail_indoor_categories.index("toilet"),
            4: long_tail_indoor_categories.index("television"),
            5: long_tail_indoor_categories.index("couch"),
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_longtail_goal_id[goal_id],
            self.hm3d_goal_id_to_longtail_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, env: Env):
        self._instance_id_to_category_id = torch.tensor(
            [
                long_tail_indoor_categories.index(
                    hm3d_to_longtail_indoor.get(
                        obj.category.name().lower().strip(), "other"
                    )
                )
                for obj in env.sim.semantic_annotations().objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> torch.Tensor:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        # TODO Replace with appropriate color palette
        return coco_map_color_palette

    @property
    def frame_color_palette(self):
        # TODO Replace with appropriate color palette
        return coco_frame_color_palette

    @property
    def categories_legend_path(self):
        # TODO Replace with appropriate legend
        return coco_categories_legend_path

    @property
    def num_sem_categories(self):
        return len(long_tail_indoor_categories)
