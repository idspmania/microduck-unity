## Microduck Locomotion Training (Experimental)

Microduck locomotion was trained on an NVIDIA RTX 3070 Laptop GPU using WSL2. The policy learns to track body-frame linear and angular velocity commands on flat terrain, including forward walking, backward walking, lateral movement, and turning in place.

### Training Setup

| Component | Configuration |
|---|---|
| Host / Linux environment | Windows with WSL2 / Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM |
| Python / PyTorch | Python 3.12 / PyTorch 2.9.1 |
| Simulation framework | mjlab 1.3.0 |
| Physics stack | MuJoCo 3.10.0, MuJoCo Warp 3.8.1, Warp 1.12.0 |
| Learning algorithm | PPO with RSL-RL 5.0.1 |
| Task | `Mjlab-Velocity-Flat-MicroDuck` |
| Parallel environments | 512 initially; 1,024 after resuming from `model_950.pt` |
| Physics / policy frequency | 200 Hz / 50 Hz |
| Policy input / output | 61 actor observations / 14 joint-position actions |
| Actuator model | BAM M6 for Dynamixel XL330 servos |
| Checkpoint interval | Every 50 training iterations |

### Training History of the Exported Policy

The exported ONNX policy was trained with **512 parallel environments up to iteration 950**, then resumed from `model_950.pt` with **1,024 parallel environments and trained through iteration 1,500** before export.

Iteration 1,500 is the cumulative iteration index across both phases, not 1,500 additional iterations after resuming.

(Images/Screenshot1.png)