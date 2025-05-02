import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import magnum as mn
from typing import List, Union

class HitDetails:
    """
    Data class for details about a single raycast hit.
    Could be a Dict, but this provides IDE API reference.
    """

    def __init__(self):
        self.object_id: int = None
        self.obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ] = None
        self.obj_name: str = None
        self.link_id: int = None
        self.link_name: str = None
        self.point: mn.Vector3 = None


class HitObjectInfo:
    """
    A data class with a simple API for identifying which object and/or link was hit by a raycast
    """

    def __init__(
        self,
        raycast_results: habitat_sim.physics.RaycastResults,
        sim: habitat_sim.Simulator,
    ):
        self.raycast_results = raycast_results
        self.sim = sim  # cache this for simplicity
        assert (
            raycast_results is not None
        ), "Must provide a valid RaycastResults"

    def has_hits(self) -> bool:
        """
        Are there any registered hits. If not other API calls are likely invalid or return None.
        """
        return self.raycast_results.has_hits()

    @property
    def hits(self) -> List[habitat_sim.physics.RayHitInfo]:
        """
        Get the RayHitInfo objects associated with the RaycastResults.
        """
        return self.raycast_results.hits

    @property
    def ray(self) -> habitat_sim.geo.Ray:
        """
        The cast Ray.
        """
        return self.raycast_results.ray

    def hit_stage(self, hit_ix: int = 0) -> bool:
        """
        Return whether or not the hit index provided was the STAGE.
        Defaults to first hit.
        """
        if self.has_hits and self.hits[hit_ix] == habitat_sim.stage_id:
            return True
        return False

    def hit_obj(
        self, hit_ix: int = 0
    ) -> Union[
        habitat_sim.physics.ManagedArticulatedObject,
        habitat_sim.physics.ManagedRigidObject,
    ]:
        """
        Get the ManagedObject at the specified hit index.
        Defaults to first hit object.
        """
        return sutils.get_obj_from_id(self.sim, self.hits[hit_ix].object_id)

    def hit_link(
        self,
        hit_ix: int = 0,
        hit_obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ] = None,
    ) -> Union[int, None]:
        """
        Gets the link index of the given hit index or None if not a link.
        :param hit_obj: Optionally provide the hit object if already known to avoid a lookup.
        """
        if self.has_hits():
            if hit_obj is None:
                hit_obj = self.hit_obj(hit_ix)
            hit_obj_id = self.hits[hit_ix].object_id
            if hit_obj is not None and hit_obj.object_id != hit_obj_id:
                return hit_obj.link_object_ids[hit_obj_id]
        return None

    def get_hit_details(self, hit_ix: int = 0) -> HitDetails:
        """
        Returns a Dict with the details of the first hit for convenience.
        """
        hit = self.hits[hit_ix]
        hit_details = HitDetails()
        hit_details.obj = self.hit_obj(hit_ix)
        hit_details.obj_name = (
            hit_details.obj.handle
            if hit.object_id != habitat_sim.stage_id
            else "STAGE"
        )
        hit_details.link_id = self.hit_link(hit_ix, hit_details.obj)
        hit_details.link_name = (
            None
            if not hit_details.link_id
            else hit_details.obj.get_link_name(hit_details.link_id)
        )
        hit_details.object_id = hit.object_id
        hit_details.point = self.hits[hit_ix].point
        return hit_details
