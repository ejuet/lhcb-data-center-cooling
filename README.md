# LHCb Data Center Cooling Challenge

This is my solution to the **LHCb Data Center Cooling Challenge**, which is a reinforcement learning problem where the goal is to control the cooling system of the data center used for CERN's LHCb experiment in order to minimize energy consumption while maintaining safe operating temperatures.

We were provided with a simulator of the data center's cooling system, which allows us to train and test our control policies in a realistic environment. The challenge is to develop a control policy that can effectively manage the cooling system under varying workloads and environmental conditions.

## Solution

The idea is to use **model-predictive control**: forecast the next few minutes, use learned cooling-power models to compare feasible fan and water settings, and commit only the first actions before replanning as new observations arrive. The existing simulator supplies the system’s realistic dynamics and constraints, so the controller can be developed and evaluated safely against the same conditions it will face at runtime.

I also experimented with improving upon this by using reinforcement learning to learn a better model of the system dynamics and constraints, but I found that the existing simulator was already quite accurate and that the MPC approach was sufficient to achieve good performance and **win the challenge**.

## Installation

Use a Python 3.12 environment to run the code in this repository.

```sh
# install the requirements (numpy and scikit-learn versions must match exactly)
pip install -r requirements.txt
# download the xdt wheel file for the environment package and install it:
pip install xdt-*.whl
```

## Usage

### MPC

The MPC controller is implemented in [`mpc/model.py`](mpc/model.py). To evaluate the MPC controller locally, run:

```sh
source .venv/bin/activate # or use your own virtual environment
source env.sh
python evaluate_policy.py --agent-dir ./mpc
```

### Residual SAC

SAC (Soft Actor-Critic) produces five small water corrections on top of the MPC dispatcher's 15-value action. The wrapper adjusts the outside-fan command to preserve the MPC action's cooling capacity and applies the simulator's bounds, ramp rate, and dry-mode water interlock. However, the MPC controller is already quite good and the SAC agent does not significantly improve upon it.

To train the residual SAC agent, run:

```sh
source .venv/bin/activate
source env.sh
python -m residual_sac.train
```

This should produce a directory `residual_sac/weights` containing the trained agent's weights.

To then evaluate the residual SAC agent, run:

```sh
python evaluate_policy.py --agent-dir residual_sac
```