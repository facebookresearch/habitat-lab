from typing import List

import magnum as mn
import numpy as np

import habitat_sim


def is_accessible(sim, point, nav_to_min_distance) -> bool:
    """
    Return True if the point is within a threshold distance of the nearest
    navigable point and that the nearest navigable point is on the same
    navigation mesh.

    Note that this might not catch all edge cases since the distance is
    based on Euclidean distance. The nearest navigable point may be
    separated from the object by an obstacle.
    """
    if nav_to_min_distance == -1:
        return True
    snapped = sim.pathfinder.snap_point(point)
    if snapped is None or np.isnan(snapped).any():
        return False
    island_idx: int = sim.pathfinder.get_island(snapped)
    dist = float(np.linalg.norm(np.array((snapped - point))[[0, 2]]))
    return (
        dist < nav_to_min_distance
        and island_idx == sim.navmesh_classification_results["active_island"]
    )


def compute_navmesh_island_classifications(
    sim: habitat_sim.Simulator, active_indoor_threshold=0.85, debug=False
):
    """
    Classify navmeshes as outdoor or indoor and find the largest indoor island.
    active_indoor_threshold is acceptacle indoor|outdoor ration for an active island (for example to allow some islands with a small porch or skylight)
    """
    if not sim.pathfinder.is_loaded:
        sim.navmesh_classification_results = None
        print("No NavMesh loaded to visualize.")
        return

    sim.navmesh_classification_results = {}

    sim.navmesh_classification_results["active_island"] = -1
    sim.navmesh_classification_results[
        "active_indoor_threshold"
    ] = active_indoor_threshold
    active_island_size = 0
    number_of_indoor = 0
    sim.navmesh_classification_results["island_info"] = {}
    sim.indoor_islands = []

    for island_ix in range(sim.pathfinder.num_islands):
        sim.navmesh_classification_results["island_info"][island_ix] = {}
        sim.navmesh_classification_results["island_info"][island_ix][
            "indoor"
        ] = island_indoor_metric(sim=sim, island_ix=island_ix)
        if (
            sim.navmesh_classification_results["island_info"][island_ix][
                "indoor"
            ]
            > active_indoor_threshold
        ):
            number_of_indoor += 1
            sim.indoor_islands.append(island_ix)
        island_size = sim.pathfinder.island_area(island_ix)
        if (
            active_island_size < island_size
            and sim.navmesh_classification_results["island_info"][island_ix][
                "indoor"
            ]
            > active_indoor_threshold
        ):
            active_island_size = island_size
            sim.navmesh_classification_results["active_island"] = island_ix
    if debug:
        print(
            f"Found active island {sim.navmesh_classification_results['active_island']} with area {active_island_size}."
        )
        print(
            f"     Found {number_of_indoor} indoor islands out of {sim.pathfinder.num_islands} total."
        )
    for island_ix in range(sim.pathfinder.num_islands):
        island_info = sim.navmesh_classification_results["island_info"][
            island_ix
        ]
        info_str = f"    {island_ix}: indoor ratio = {island_info['indoor']}, area = {sim.pathfinder.island_area(island_ix)}"
        if sim.navmesh_classification_results["active_island"] == island_ix:
            info_str += "  -- active--"
    if debug:
        print(info_str)


def island_indoor_metric(
    sim: habitat_sim.Simulator,
    island_ix: int,
    num_samples=100,
    jitter_dist=0.1,
    max_tries=1000,
) -> float:
    """
    Compute a heuristic for ratio of an island inside vs. outside based on checking whether there is a roof over a set of sampled navmesh points.
    """

    assert sim.pathfinder.is_loaded
    assert sim.pathfinder.num_islands > island_ix

    # collect jittered samples
    samples: List[np.ndarray] = []
    for _sample_ix in range(max_tries):
        new_sample = sim.pathfinder.get_random_navigable_point(
            island_index=island_ix
        )
        too_close = False
        for prev_sample in samples:
            dist_to = np.linalg.norm(prev_sample - new_sample)
            if dist_to < jitter_dist:
                too_close = True
                break
        if not too_close:
            samples.append(new_sample)
        if len(samples) >= num_samples:
            break

    # classify samples
    indoor_count = 0
    for sample in samples:
        raycast_results = sim.cast_ray(
            habitat_sim.geo.Ray(sample, mn.Vector3(0, 1, 0))
        )
        if raycast_results.has_hits():
            # assume any hit indicates "indoor"
            indoor_count += 1

    # return the ration of indoor to outdoor as the metric
    return indoor_count / len(samples)
