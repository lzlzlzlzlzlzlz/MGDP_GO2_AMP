# Source provenance

This project contains a selective snapshot of `MGDP/MGDP-master` from the sibling source directory. The copied `legged_gym/legged_gym`, `legged_gym/rl`, `legged_gym/scripts`, Go2 robot asset, and `warp_sensor` directories retain their upstream code and notices. The new Go2 AMP adapter and reward code live alongside that snapshot. The original source projects remain untouched.

The 17 expert trajectories in `datasets/go2_motion` are copies of `amp_go2/datasets/go2_motion` from the sibling project. Their 12-joint ordering matches MGDP's Go2 joint ordering; the source URDFs have small calf-limit differences, so the AMP discriminator uses joint state and body velocities rather than foot geometry. The loader uses columns 7–18, 31–36, and 37–48 from each 61-value frame.

The AMP reward design follows the least-squares discriminator and bounded style reward in `WMP/WMP-master`. No WMP source files are copied. See the design specification for the exact mapping and weighting.

The copied MGDP and Warp package metadata retain their upstream license declarations. The `amp_go2` trajectory directory itself does not contain a separate license notice; verify redistribution terms before publishing the dataset outside this workspace.
