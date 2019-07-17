import numba
import numpy as np

from habitat.utils.visualizations import maps


@numba.jit(nopython=True)
def _draw(fog_of_war_mask, x, y):
    fog_of_war_mask[x, y] = 1


@numba.jit(nopython=True)
def draw_fog_of_war_line(top_down_map, fog_of_war_mask, pt1, pt2):
    r"""Draws a line on the fog_of_war_mask mask between pt1 and pt2
    """
    x1, y1 = pt1
    x2, y2 = pt2
    ystep = xstep = 1

    x, y = x1, y1
    dx, dy = pt2 - pt1

    if dy < 0:
        ystep *= -1
        dy *= -1

    if dx < 0:
        xstep *= -1
        dx *= -1

    _draw(fog_of_war_mask, x, y)

    ddx, ddy = 2 * dx, 2 * dy
    if ddx > ddy:
        errorprev = dx
        error = dx
        for i in range(int(dx)):
            x += xstep
            error += ddy

            if top_down_map[x, y] == maps.MAP_INVALID_POINT:
                break

            if x <= 0 or x >= (fog_of_war_mask.shape[0] - 1):
                break

            if y <= 0 or y >= (fog_of_war_mask.shape[1] - 1):
                break

            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    _draw(fog_of_war_mask, x, y - ystep)
                elif error + errorprev > ddx:
                    _draw(fog_of_war_mask, x - xstep, y)
                else:
                    _draw(fog_of_war_mask, x - xstep, y)
                    _draw(fog_of_war_mask, x, y - ystep)

            _draw(fog_of_war_mask, x, y)

            errorprev = error

    else:
        errorprev = dx
        error = dx
        for i in range(int(dy)):
            y += ystep
            error += ddx

            if top_down_map[x, y] == maps.MAP_INVALID_POINT:
                break

            if x <= 0 or x >= (fog_of_war_mask.shape[0] - 1):
                break

            if y <= 0 or y >= (fog_of_war_mask.shape[1] - 1):
                break

            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    _draw(fog_of_war_mask, x - xstep, y)
                elif error + errorprev > ddy:
                    _draw(fog_of_war_mask, x, y - ystep)
                else:
                    _draw(fog_of_war_mask, x - xstep, y)
                    _draw(fog_of_war_mask, x, y - ystep)

            _draw(fog_of_war_mask, x, y)

            errorprev = error


@numba.jit(nopython=True)
def _draw_loop(
    top_down_map,
    fog_of_war_mask,
    current_pt,
    current_angle,
    max_line_len,
    angles,
):
    for angle in angles:
        angle = np.deg2rad(angle)
        draw_fog_of_war_line(
            top_down_map,
            fog_of_war_mask,
            current_pt,
            current_pt
            + max_line_len
            * np.array(
                [np.cos(current_angle + angle), np.sin(current_angle + angle)]
            ),
        )


def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_pt: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    r"""Reveals the fog-of-war at the current location

    This works by simply drawing lines from the agents current location
    and stopping once a wall is hit

    Args:
        current_fog_of_war_mask: The current fog-of-war mask to reveal the fog-of-war on
        current_pt: The current location of the agent on the fog_of_war_mask
        current_angle: The current look direction of the agent on the fog_of_war_mask
        fov: The feild of view of the agent
        max_line_len: The maximum length of the lines used to reveal the fog-of-war

    Returns:
        The updated fog_of_war_mask
    """
    angles = np.arange(
        -fov / 2, fov / 2, step=50.0 / max_line_len, dtype=np.float32
    )

    fog_of_war_mask = current_fog_of_war_mask.copy()
    _draw_loop(
        top_down_map,
        fog_of_war_mask,
        current_pt,
        current_angle,
        max_line_len,
        angles,
    )

    return fog_of_war_mask
