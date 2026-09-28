"""Helpers for resolving Isaac Gym device arguments."""


def resolve_sim_device_id(render_device, compute_device_id):
    """Use an explicit render GPU when set, otherwise Isaac Gym's compute GPU."""
    if render_device is not None:
        return render_device
    return 0 if compute_device_id is None else compute_device_id
