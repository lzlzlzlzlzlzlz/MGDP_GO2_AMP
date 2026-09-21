# Source provenance

This project contains a selective snapshot of `MGDP/MGDP-master` from the sibling source directory. The copied `legged_gym/legged_gym`, `legged_gym/rl`, `legged_gym/scripts`, Go2 robot asset, and `warp_sensor` directories retain their upstream code and notices. The new Go2 AMP adapter and reward code live alongside that snapshot. The original source projects remain untouched.

The expert trajectories are copied from `amp_go2/datasets/go2_motion` in Task 2. Their 12-joint ordering matches MGDP's Go2 joint ordering; the source URDFs have small calf-limit differences, so the AMP discriminator uses joint state and body velocities rather than foot geometry.

The AMP reward design follows the least-squares discriminator and bounded style reward in `WMP/WMP-master`. See the design specification for the exact mapping and weighting.
