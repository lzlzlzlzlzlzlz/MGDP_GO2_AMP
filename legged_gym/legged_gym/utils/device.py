"""Helpers for resolving Isaac Gym device arguments."""


def resolve_sim_device_id(render_device, compute_device_id):
    """Use an explicit render GPU when set, otherwise Isaac Gym's compute GPU."""
    return compute_device_id if render_device is None else render_device
