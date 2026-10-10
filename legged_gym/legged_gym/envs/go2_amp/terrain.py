"""Pure Stage 1 terrain-column and logical-anchor allocation helpers."""

from typing import Tuple

import numpy as np


GO2_AMP_TERRAIN_COLUMNS: Tuple[str, ...] = (
    "slope down",
    "slope up",
    "rough pyramid",
    "stairs down",
    "stairs up",
    "discrete obstacles",
)

GO2_AMP_COURSE_COLUMN_CYCLE: Tuple[int, ...] = (0, 1, 2, 2, 3, 3, 4, 4, 5, 5)


def compute_anchor_count(num_envs: int, fraction: float = 0.15) -> int:
    """Return the logical-anchor quota while preserving one course environment."""
    if num_envs < 2:
        raise ValueError("Go2 AMP Stage 1 requires at least 2 environments")
    return min(num_envs - 1, max(1, round(num_envs * fraction)))


def assign_amp_columns(
    num_envs: int,
    fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign anchor-prefix and fixed-cycle course columns."""
    anchor_count = compute_anchor_count(num_envs, fraction)
    is_amp_anchor = np.zeros(num_envs, dtype=np.bool_)
    is_amp_anchor[:anchor_count] = True

    columns = np.zeros(num_envs, dtype=np.int64)
    course_count = num_envs - anchor_count
    columns[anchor_count:] = np.resize(
        np.asarray(GO2_AMP_COURSE_COLUMN_CYCLE, dtype=np.int64),
        course_count,
    )
    return columns, is_amp_anchor
