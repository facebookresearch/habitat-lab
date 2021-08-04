import copy
import json
import os
import os.path as osp
import random
from abc import ABC, abstractmethod

import magnum as mn
import numpy as np
from matplotlib.path import Path
from orp.utils import euler_to_quat, get_aabb

import habitat_sim

MAX_HEIGHT_DIFF = 0.15


def did_object_fall(start_pos, pos):
    diff_y = abs(pos[1] - start_pos[1])
    if diff_y > MAX_HEIGHT_DIFF:
        return False
    return True


class ObjTypeSampler:
    def __init__(self, options, holdout_options):
        self.options = options
        self.holdout_options = holdout_options

    def sample(self, mode):
        if mode == "eval_unseen" and self.holdout_options is not None:
            return random.choice(self.holdout_options)
        else:
            return random.choice(self.options)


class ObjSampler(object):
    def sample(self, old_pos, obj_idx, start_idx=None):
        raise ValueError()

    def should_stabilize(self):
        return False

    def should_add_offset(self):
        return True

    def reset_balance(self):
        pass

    def is_legal(self, start_pos, pos):
        return def_is_legal(start_pos, pos)

    def set_sim(self, sim):
        self.sim = sim


class VoidSampler(ObjSampler):
    def sample(self, old_pos, obj_idx, start_idx=None):
        return "void"


class PolySurface(ObjSampler):
    def __init__(self, height, poly, height_noise=0.0, trans=None):
        self.height = height
        if trans is not None:
            self.poly = [
                trans.transform_point(mn.Vector3(p[0], height, p[1]))
                for p in poly
            ]
            self.poly = [[x[0], x[2]] for x in self.poly]
        else:
            self.poly = poly
        self.height_noise = height_noise

    def should_stabilize(self):
        return True

    def sample(self, old_pos, obj_idx, start_idx=None):
        size = 1000
        extent = 3

        # Draw a square around the average of the points
        avg = np.mean(np.array(self.poly), axis=0)
        x, y = np.meshgrid(
            np.linspace(avg[0] - extent, avg[0] + extent, size),
            np.linspace(avg[1] - extent, avg[1] + extent, size),
        )
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        p = Path(self.poly)  # make a polygon
        grid = p.contains_points(points)
        points = points.reshape(size ** 2, 2)
        valid_points = points[grid == True]
        self.p = p

        use_x, use_y = valid_points[np.random.randint(len(valid_points))]
        return [
            use_x,
            self.height + np.random.uniform(0, self.height_noise),
            use_y,
        ]

    def is_legal(self, start_pos, pos):
        pos_2d = [pos[0], pos[2]]
        # if not self.p.contains_point(pos_2d):
        #    return False
        diff_y = abs(pos[1] - start_pos[1])
        if diff_y > (MAX_HEIGHT_DIFF + self.height_noise):
            # print('Y diff', diff_y)
            return False
        return True


def tower_sampler_from_surface(surface):
    return TowerSampler(poly=surface.poly, height=surface.height)


class TowerSampler(PolySurface):
    def __init__(self, height, poly, height_increment=0.10):
        # does not support height noise since this is manually added in.
        super().__init__(height, poly)
        self.height_increment = height_increment
        self.obj_heights = {}

    def reset_balance(self):
        # sample a position on the table
        self.start_pos = super().sample(None, -1)
        # self.cur_h = self.height + (self.height_increment / 2)
        self.cur_h = self.height + self.height_increment
        self.obj_heights = {}

    def should_add_offset(self):
        return False

    def sample(self, old_pos, obj_idx, start_idx=None):
        if obj_idx not in self.obj_heights:
            self.obj_heights[obj_idx] = self.cur_h
            bb = get_aabb(obj_idx, self.sim)
            use_y = self.cur_h
            offset = (bb.size()[1] / 2.0) + 0.01
            # offset = self.height_increment
            self.cur_h += offset
        else:
            use_y = self.obj_heights[obj_idx]

        ret = [self.start_pos[0], use_y, self.start_pos[2]]
        return ret


class NoiseAdd(ObjSampler):
    def __init__(self, x_noise, y_noise, stab=False):
        self.noise_vec = np.array([x_noise, y_noise])
        self.stab = stab

    def should_stabilize(self):
        return self.stab

    def sample(self, old_pos, obj_idx, start_idx=None):
        add_noise = self.noise_vec * np.random.normal(0, 1, 2)
        new_pos = [
            old_pos[0] + add_noise[0],
            old_pos[1],
            old_pos[2] + add_noise[1],
        ]
        # Have to do this because of JSON serialization problems.
        return list(np.array(new_pos).astype(float))


class IdentitySampler(ObjSampler):
    def sample(self, old_pos, obj_idx, start_idx=None):
        return old_pos


class ReplaceSampler(ObjSampler):
    def __init__(self, new_pos):
        self.replace_pos = new_pos

    def sample(self, old_pos, obj_idx, start_idx=None):
        return list(np.array(self.replace_pos).astype(float))


class OneOfSampler(ObjSampler):
    def __init__(
        self,
        samplers,
        should_regen_idx=False,
        goal_on_diff=False,
        other_sampler=None,
    ):
        """
        - should_regen_idx: If True, a new region will be sampled every
          sampling attempt, NOT every episode. This could bias towards certain
          sampling regions which are easier to sample from.
        - goal_on_diff: Should the goal be sampled from a different sampler
          than the starting position?
        """
        self.samplers = samplers
        self.should_regen_idx = should_regen_idx
        self.goal_on_diff = goal_on_diff
        self.start_samplers = {}
        self.other_sampler = other_sampler

    def reset_balance(self):
        self.use_idx = np.random.randint(len(self.samplers))
        for sampler in self.samplers:
            sampler.reset_balance()

    def is_legal(self, start_pos, pos):
        return self.samplers[self.use_idx].is_legal(start_pos, pos)

    def should_stabilize(self):
        return self.samplers[self.use_idx].should_stabilize()

    def sample(self, old_pos, obj_idx, start_idx=None):
        if self.should_regen_idx:
            if start_idx is not None and self.goal_on_diff:
                if self.other_sampler is not None:
                    prev_idx = self.other_sampler.start_samplers[start_idx]
                else:
                    prev_idx = self.start_samplers[start_idx]
                # Sample from the set of allowed indices
                allowed_idxs = list(range(len(self.samplers)))
                del allowed_idxs[prev_idx]
                self.use_idx = random.choice(allowed_idxs)
            else:
                self.use_idx = np.random.randint(len(self.samplers))

        self.start_samplers[obj_idx] = self.use_idx

        return self.samplers[self.use_idx].sample(old_pos, obj_idx)

    def __str__(self):
        sampler_str = ",".join([str(x) for x in self.samplers])
        return (
            f"OneOfSampler[use_idx={self.use_idx}, samplers=({sampler_str})]"
        )


class CompositeSampler(ObjSampler):
    def __init__(self, samplers):
        self.samplers = samplers

    def reset_balance(self):
        for sampler in self.samplers:
            sampler.reset_balance()

    def should_stabilize(self):
        return any([x.should_stabilize() for x in self.samplers])

    def sample(self, old_pos, obj_idx, start_idx=None):
        for sampler in self.samplers:
            old_pos = sampler.sample(old_pos, obj_idx)
        return old_pos


class ArtLinkBetween(ObjSampler):
    def __init__(self, link, min_state, max_state, succ_thresh):
        self.link = link
        self.min_state = min_state
        self.max_state = max_state
        self.succ_thresh = succ_thresh

    def sample(self, old_pos, obj_idx, start_idx=None):
        set_link = np.random.uniform(self.min_state, self.max_state)
        self.set_val = set_link
        arr = old_pos
        arr[self.link] = set_link
        return arr

    def catch(self, catch_ids, name_to_id, sim):
        pass

    def is_satisfied(self, art_pos):
        art_pos = np.array(art_pos)[self.link]
        lower = art_pos < self.max_state + self.succ_thresh
        higher = art_pos > self.min_state - self.succ_thresh
        return lower and higher


class ArtLinkFromMarkerBetween(ArtLinkBetween):
    def __init__(
        self,
        marker_name,
        min_state,
        max_state,
        obj_offset_factor,
        sim,
        name_to_id,
        succ_thresh,
    ):
        real_marker_name = name_to_id[marker_name]
        link_idx = sim.markers[real_marker_name]["relative"][1] - 1
        self.marker_name = real_marker_name
        self.obj_offset_factor = obj_offset_factor
        super().__init__(link_idx, min_state, max_state, succ_thresh)

    def catch(self, move_id, name_to_id, sim):
        if move_id is None:
            return
        offset = mn.Vector3(self.set_val * self.obj_offset_factor, 0, 0)
        marker = sim.markers[self.marker_name]

        offset = marker["T"].transform_vector(offset)
        obj_id = name_to_id[move_id]
        obj_id = sim.scene_obj_ids[obj_id]
        pos = sim._sim.get_translation(obj_id)

        # Move from the object target starting position
        targ_idxs = sim.get_target_obj_idxs()
        if obj_id not in targ_idxs:
            raise ValueError("Cannot move a non-target object.")

        rel_targ_idx = targ_idxs.index(obj_id)
        start_pos = sim.get_target_objs_start()[rel_targ_idx]
        sim._sim.set_translation(start_pos + offset, obj_id)


class SceneSampler(ABC):
    @abstractmethod
    def num_scenes(self):
        pass


class SingleSceneSampler(SceneSampler):
    def __init__(self, scene):
        self.scene = scene

    def sample(self):
        return self.scene, {}

    def num_scenes(self):
        return 1


IGNORE_FILES = [
    #'data/scene_datasets/v3_sc1_staging_07.glb',
    "data/scene_datasets/v3_sc4_staging_04.glb",
    #'data/scene_datasets/v3_sc0_staging_09.glb',
]


class MultiSceneSampler(SceneSampler):
    def __init__(
        self,
        search_dir,
        search_name,
        holdout_frac=1.0,
        holdout_mode="train",
        filter_bbs=[],
        test_scene=None,
        baked_mode=False,
    ):
        """
        - filter_bbs: Filter scenes which don't contain a bounding box with
          this name.
        - holdout_frac: The ratio of scenes which should be used for training.
          0.8 means that 80% of scenes are for training and 20% are for eval.
        """

        self.scene_files = []
        macro_scenes = [0, 1, 2, 3, 4]
        for sc in macro_scenes:
            for f in os.listdir(search_dir):
                use_search_name = search_name % sc
                if use_search_name in f and ".glb" in f:
                    full_name = osp.join(search_dir, f)
                    if full_name in IGNORE_FILES:
                        continue
                    self.scene_files.append(full_name)
        if baked_mode:
            self.scene_files = [x for x in self.scene_files if "remake" in x]
        else:
            self.scene_files = [
                x
                for x in self.scene_files
                if ("remake" not in x) and ("Baked" not in x)
            ]

        if test_scene is not None:
            print("OVERRIDE WITH TEST SCENE %s!" % test_scene)
            self.scene_files = [test_scene]

        self.scene_files = sorted(self.scene_files)
        # Repeatable shuffle so the split is always the same
        np.random.RandomState(42).shuffle(self.scene_files)
        self.cur_scene_file_i = 0

        art_metaf = osp.join(search_dir, "v1_sc_staging_art_transform.json")
        with open(art_metaf, "r") as f:
            art_meta = json.load(f)

        name_to_art_trans = {}
        for d in art_meta:
            name_to_art_trans[d["name"]] = d

        self.metadata = {}
        remove_scene_ks = []
        for scenef in self.scene_files:
            name = scenef.split(".")[-2]
            metaf = name + ".json"
            if not osp.exists(metaf):
                raise ValueError("Metadata file %s does not exist" % metaf)
            with open(metaf, "r") as f:
                txt = f.readlines()
                if "," in txt[-1]:
                    txt[-1] = "]"
                txt = "".join(txt)
                bbs = json.loads(txt)
            configf = name + ".stage_config.json"

            if not osp.exists(configf):
                rel_path = scenef.split("/")[-1]
                with open(configf, "w") as f:
                    f.write(
                        """
{
	"render_asset": "%s",
	"up":[0,1,0],
	"front":[0,0,-1],
	"requires_lighting":true
}
                            """
                        % rel_path
                    )
            self.metadata[scenef] = self._load_bb(
                scenef, bbs, name_to_art_trans
            )
            found_all = True
            for bb in filter_bbs:
                if bb not in self.metadata[scenef]["bb"]:
                    found_all = False
                    break
            if not found_all:
                remove_scene_ks.append(scenef)

        for remove_scenef in remove_scene_ks:
            del self.scene_files[self.scene_files.index(remove_scenef)]
            del self.metadata[remove_scenef]
        # Ignore any baked files that are for some reason in the same folder
        if test_scene is not None and "Baked" not in test_scene:
            self.scene_files = [
                x for x in self.scene_files if "Baked" not in x
            ]

        total = len(self.scene_files)
        n_train = int(total * holdout_frac)
        n_eval = total - n_train
        HOLDOUT = "sc4"
        tmp = [x for x in self.scene_files if HOLDOUT in x]
        if holdout_mode == "scene_eval":
            self.scene_files = self.scene_files[-n_eval:]
        elif holdout_mode == "macro_scene_train":
            self.scene_files = [
                x for x in self.scene_files if HOLDOUT not in x
            ]
        elif holdout_mode == "macro_scene_eval":
            self.scene_files = [x for x in self.scene_files if HOLDOUT in x]
        else:
            # Regular train
            self.scene_files = self.scene_files[:n_train]

    def _adjust_art_trans_metadata(self, trans_d, offset):
        trans_d["position"][1], trans_d["position"][2] = (
            trans_d["position"][2],
            trans_d["position"][1],
        )
        trans_d["position"][2] *= -1.0
        trans_d["position"] = np.array(trans_d["position"]) + np.array(offset)
        quat_rot = mn.Quaternion.identity_init()
        if sum(trans_d["rotEuler"]) != 0.0:
            quat_rot = mn.Quaternion.from_matrix(
                mn.Matrix4.rotation_y(mn.Deg(90)).rotation()
            )
        trans_d["rotation"] = quat_rot
        return trans_d

    def _load_bb(self, scene_name, bbs, name_to_art_trans):
        # Convert metadata into samplers
        samplers = {}
        for meta_d in bbs:
            pos = meta_d["position"]
            scale = meta_d["scale"]
            rot = meta_d["rotEuler"]
            # Swap y,z
            pos[1], pos[2] = pos[2], pos[1]
            scale[1], scale[2] = scale[2], scale[1]
            pos[2] *= -1
            rot[1], rot[2] = rot[2], rot[1]
            scale = mn.Vector3(*scale)
            name = meta_d["name"]
            scale_factor = 0.8

            bb = mn.Range3D.from_center(
                mn.Vector3(0.0, 0.0, 0.0), scale * scale_factor
            )
            rot_quat = euler_to_quat(rot)
            rot_T = mn.Matrix4.from_(rot_quat.to_matrix(), mn.Vector3(*pos))

            corners = [
                bb.back_bottom_left,
                bb.back_bottom_right,
                bb.front_bottom_left,
                bb.front_bottom_right,
            ]

            corners = [rot_T.transform_point(corner) for corner in corners]
            y = min([x[1] for x in corners])
            # height_noise = max([x[1] for x in corners])
            height_noise = 0.0

            corners = [[x[0], x[2]] for x in corners]

            # Remove the count identifier
            reg_name = name.split(".")[0]

            surface = PolySurface(y, corners, height_noise)
            samplers[reg_name] = surface

        scene_id = scene_name.split("v3_sc")[1].split("_")[0]

        cab_trans = copy.deepcopy(
            name_to_art_trans["Empty_CabinetCounter_Estimation_0" + scene_id]
        )
        fridge_trans = copy.deepcopy(
            name_to_art_trans["Empty_Fridge_Estimation_0" + scene_id]
        )
        if sum(cab_trans["rotEuler"]) != 0.0:
            cab_trans = self._adjust_art_trans_metadata(
                cab_trans, [0.2 - 0.3, -0.03703, 0.0 - 0.1]
            )
            fridge_trans = self._adjust_art_trans_metadata(
                fridge_trans, [0.0, 1.0, 0.0 + 0.9]
            )
        else:
            cab_trans = self._adjust_art_trans_metadata(
                cab_trans, [0.2, -0.03703, 0.0]
            )
            fridge_trans = self._adjust_art_trans_metadata(
                fridge_trans, [0.0, 1.0, 0.0]
            )

        # Adjust position slightly
        return {
            "bb": samplers,
            "art": {"cab": cab_trans, "fridge": fridge_trans},
        }

    def sample(self):
        scene_file = self.scene_files[self.cur_scene_file_i]
        self.cur_scene_file_i += 1
        if self.cur_scene_file_i >= len(self.scene_files):
            self.cur_scene_file_i = 0
        use_meta = self.metadata[scene_file]
        print("Trying to load", scene_file)

        return scene_file, use_meta

    def num_scenes(self):
        return len(self.scene_files)
