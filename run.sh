source .venv/bin/activate
export DT_ARTIFACTS_DIR="$PWD/starting_kit/dt_artifacts"
export DT_NODE_DETAILS="$PWD/starting_kit/node_details_v2.csv"
python evaluate_policy.py --agent-dir ./mpc