# from abc import abstractmethod
#
# import magnum as mn
# import numpy as np
#
# from habitat.tasks.rearrange.rearrange_task import RearrangeTask
#
#
# class SetArticulatedObjectTask(RearrangeTask):
#    def __init__(self, *args, config, dataset=None, **kwargs):
#        super().__init__(config=config, *args, dataset=dataset, **kwargs)
#        self.force_marker = None
#
#    @abstractmethod
#    def _get_targ_art_obj(self):
#        pass
#
#    def set_args(self, **kwargs):
#        pass
#
#    @abstractmethod
#    def _gen_start_state(self):
#        pass
#
#    @abstractmethod
#    @abstractmethod
#    def _get_look_pos(self):
#        """
#        The point defining where the robot should face at the start of the
#        episode.
#        """
#
#    @abstractmethod
#    def _get_push_marker_name(self):
#        pass
#
#    @abstractmethod
#    def _get_name_id(self):
#        pass
#
#    @abstractmethod
#    def _get_succ_state(self):
#        pass
#
#    @property
#    def _is_start_global(self):
#        return False
#
#    @abstractmethod
#    def _sample_pos(self) -> np.ndarray:
#        """
#        Returns a 2D vector for the robot start position
#        """
#
#    def _sample_robot_start(self, T, sim):
#        start_pos = self._sample_pos()
#        start_pos = np.array([start_pos[0], 0.0, start_pos[1]])
#        targ_pos = np.array(self._get_look_pos())
#
#        if not self._is_start_global:
#            # Transform to global coordinates
#            start_pos = np.array(T.transform_point(mn.Vector3(*start_pos)))
#            start_pos = np.array([start_pos[0], 0, start_pos[2]])
#
#        targ_pos = np.array(T.transform_point(mn.Vector3(*targ_pos)))
#
#        # Spawn the robot facing the look pos
#        forward = np.array([1.0, 0, 0])
#        rel_targ = targ_pos - start_pos
#        angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
#        if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
#            angle_to_obj *= -1.0
#        return angle_to_obj, start_pos
#
#    def reset(self, super_reset=True):
#        self.set_task_data()
#        super().reset(super_reset)
#        self.push_point_viz = None
#        self.force_marker = None
#        self.prev_art_succ = False
#        self.has_picked = False
#        self.has_released = False
#
#        sim = self._env._sim
#
#        push_marker_name = self._get_push_marker_name()
#        if self.tcfg.USE_MARKER_T:
#            T = sim.markers[push_marker_name]["T"]
#        else:
#            relative_art_obj = sim.markers[push_marker_name]["relative"][0]
#            rel_art_obj_id = sim.art_obj_ids[relative_art_obj]
#            T = sim.get_articulated_object_root_state(rel_art_obj_id)
#
#        if super_reset:
#            num_timeout = 100
#            num_pos_timeout = 100
#            self._disable_art_sleep(sim)
#            for attempt in range(num_timeout):
#                for i in range(num_pos_timeout):
#                    angle_to_obj, start_pos = self._sample_robot_start(T, sim)
#                    if sim.pathfinder.is_navigable(start_pos):
#                        break
#
#                noise = np.random.normal(0.0, self.rlcfg.BASE_ANGLE_NOISE)
#                sim.set_robot_rot(angle_to_obj + noise)
#                sim.set_robot_pos(start_pos[[0, 2]])
#
#                # Set the articulated object state
#                self._set_link_state(self._gen_start_state())
#                did_collide = False
#                for i in range(self._reset_settle_time):
#                    sim.internal_step(-1)
#                    colls = sim.get_collisions()
#                    did_collide, details = rearrang_collision(
#                        colls,
#                        None,
#                        self.tcfg.COUNT_OBJ_COLLISIONS,
#                        ignore_base=False,
#                    )
#                    if did_collide:
#                        break
#                if not did_collide:
#                    break
#
#            # Step so the updated art position evaluates
#            sim.internal_step(-1)
#            self._reset_art_sleep(sim)
#
#        self.prev_art_state = self._get_link_state()
#        self.prev_dist_to_push = -1
#
#        self.prev_snapped_marker_name = None
#        return self.get_task_obs()
#
#    def _disable_art_sleep(self, sim):
#        targ_idx, _ = self._get_targ_art_obj()
#        abs_art_idx = self._env._sim.art_obj_ids[targ_idx]
#        self.prev_sleep = sim.get_articulated_object_sleep(abs_art_idx)
#        sim.set_articulated_object_sleep(abs_art_idx, False)
#
#    def _reset_art_sleep(self, sim):
#        targ_idx, _ = self._get_targ_art_obj()
#        abs_art_idx = self._env._sim.art_obj_ids[targ_idx]
#        sim.set_articulated_object_sleep(abs_art_idx, self.prev_sleep)
#
#    def _set_link_state(self, art_pos):
#        targ_idx, _ = self._get_targ_art_obj()
#        abs_art_idx = self._env._sim.art_obj_ids[targ_idx]
#        self._env._sim.set_articulated_object_positions(abs_art_idx, art_pos)
#
#    def _get_link_state(self):
#        targ_idx, targ_link = self._get_targ_art_obj()
#        abs_art_idx = self._env._sim.art_obj_ids[targ_idx]
#        art_pos = self._env._sim.get_articulated_object_positions(abs_art_idx)
#        return art_pos[targ_link]
#
#    def _get_art_pos(self):
#        art_idx = self._get_targ_art_obj()[0]
#        abs_art_idx = self._env._sim.art_obj_ids[art_idx]
#        return self._env._sim.get_articulated_object_root_state(
#            abs_art_idx
#        ).translation
#
#    def set_task_data(self):
#        self._env._sim.track_markers = [self._get_push_marker_name()]
