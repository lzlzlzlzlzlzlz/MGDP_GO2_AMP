"""Replace auto-reset observations with physical terminal AMP states."""

import torch


def select_next_amp_state(next_state, dones, terminal_packet):
    done_ids = torch.nonzero(dones.reshape(-1) > 0, as_tuple=False).flatten()
    if done_ids.numel() == 0:
        if terminal_packet is not None and terminal_packet[0].numel() != 0:
            raise ValueError("terminal AMP packet without done environments")
        return next_state.clone()
    if terminal_packet is None:
        raise ValueError("missing terminal AMP states for done environments")
    ids, states = terminal_packet
    if (ids.shape != done_ids.shape or not torch.equal(ids, done_ids)
            or states.shape != (len(ids), next_state.shape[1])):
        raise ValueError("terminal AMP packet ids or shape do not match done rows")
    result = next_state.clone()
    result[ids] = states
    return result
