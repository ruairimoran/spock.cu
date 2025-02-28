import numpy as np
import argparse
import factories as f
import pandas as pd
import datetime


def check_spd(mat, name):
    eigs = np.linalg.eigvals(mat)
    is_positive_definite = eigs.all() > 0
    is_symmetric = np.allclose(mat, mat.T)
    print("Is " + name + " symmetric positive-definite?", is_positive_definite and is_symmetric,
          " with norm (", np.linalg.norm(mat), ").")


parser = argparse.ArgumentParser(description='Example: power distribution.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# # Sizes
# horizon = 5
# stopping = 1
# num_events = 2
# num_inputs = 2
# num_states = num_inputs * 2
#
# print(
#     "\n",
#     "Events:", num_events, "\n",
#     "Horizon:", horizon, "\n",
#     "Stop branch:", stopping, "\n",
#     "States:", num_states, "\n",
#     "Inputs:", num_inputs, "\n",
# )

# --------------------------------------------------------
# Generate scenario tree from energy data
# --------------------------------------------------------

# ---- Read wind file ----
folder = "examples/powerDistribution/niEnergyData/"
wind_df = pd.read_csv(folder + "wind_generation_and_forecast_2025.csv", parse_dates=["DATE & TIME"])
wind_df.columns = wind_df.columns.str.strip()  # Remove extra spaces
wind_df["hour"] = wind_df["DATE & TIME"].dt.hour + wind_df["DATE & TIME"].dt.minute / 60  # Convert to hours
wind_df["FORECAST WIND(MW)"] = pd.to_numeric(wind_df["FORECAST WIND(MW)"], errors="coerce")
wind_df["ACTUAL WIND(MW)"] = pd.to_numeric(wind_df["ACTUAL WIND(MW)"], errors="coerce")
wind_df["error"] = wind_df["ACTUAL WIND(MW)"] - wind_df["FORECAST WIND(MW)"]
wind_df["date"] = wind_df["DATE & TIME"].dt.date  # Extract date only
wind_daily = wind_df.groupby("date")  # Group by date
max_time_steps = wind_daily.size().max()
err_wind_list = []
for _, group in wind_daily:
    wind_daily_err = group.sort_values("hour")["error"].values  # Sort by time of day
    if len(wind_daily_err) < max_time_steps:
        raise Exception("Missing value!")
    err_wind_list.append(wind_daily_err)
err_wind = np.array(err_wind_list).reshape(len(err_wind_list), 1, max_time_steps)

# ---- Create tree from data ----
horizon = 47  # 15 minute periods
data = err_wind  # samples x dim x time
branching = np.ones(horizon, dtype=np.int32)
branching[:3] = [5, 2, 1]
tree = f.tree.FromData(data, branching).build()
scenarios = tree.get_scenarios()
values = tree.data_values

# ---- Plot tree ----
# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation

# ---- Plot data ----
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
times = .25 * np.arange(tree.num_stages)
for s in scenarios:
    plt.plot(times, values[s], marker="o", linestyle="-")

plt.xlabel("Time (Hours from Start)")
plt.ylabel("Wind Forecast Error (MW)")
plt.title("Wind Forecast Error vs. Time")
plt.grid(True)
plt.show()

# --------------------------------------------------------

# r = np.ones(3)
# v = r / sum(r)
#
# (final_stage, stop_branching_stage) = (3, 2)
# tree = f.tree.IidProcess(
#     distribution=v,
#     horizon=final_stage,
#     stopping_stage=stop_branching_stage
# ).build()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation
# print(tree)

# # --------------------------------------------------------
# # Generate problem data
# # --------------------------------------------------------
# # Dynamics
# dynamics = []
# A_base = np.eye(num_states)
# B_base = rng.normal(0., 1., size=(num_states, num_inputs))
# for i in range(num_events):
#     A = A_base + rng.normal(0., .01, size=(num_states, num_states))
#     # A = enforce_contraction(A)
#     B = B_base + rng.normal(0., .01, size=(num_states, num_inputs))
#     dynamics += [f.build.LinearDynamics(A, B)]
#
# # Costs
# nonleaf_costs = []
# Q_base = rng.normal(0., .02, size=(num_states, num_states))
# R_base = rng.normal(0., .01, size=(num_inputs, num_inputs))
# for i in range(num_events):
#     Q_w = Q_base + rng.normal(0., .01, size=(num_states, num_states))
#     R_w = R_base + rng.normal(0., .01, size=(num_inputs, num_inputs))
#     Q = Q_w @ Q_w.T
#     R = R_w @ R_w.T
#     # check_spd(Q, "Q")
#     # check_spd(R, "R")
#     nonleaf_costs += [f.build.NonleafCost(Q, R)]
#
# T = Q_base @ Q_base.T
# # check_spd(T, "T")
# leaf_cost = f.build.LeafCost(T)
#
# # Constraints
# nonleaf_state_ub = rng.uniform(1., 2., num_states)
# nonleaf_state_lb = -nonleaf_state_ub
# nonleaf_input_ub = rng.uniform(0., .1, num_inputs)
# nonleaf_input_lb = -nonleaf_input_ub
# nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
# nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
# nonleaf_constraint = f.build.Rectangle(nonleaf_lb, nonleaf_ub)
# leaf_ub = rng.uniform(1., 2., num_states)
# leaf_lb = -leaf_ub
# leaf_constraint = f.build.Rectangle(leaf_lb, leaf_ub)
#
# # Risk
# alpha = rng.uniform(0., 1.)
# risk = f.build.AVaR(alpha)
#
# # Generate problem data
# problem = (
#     f.problem.Factory(
#         scenario_tree=tree,
#         num_states=num_states,
#         num_inputs=num_inputs)
#     .with_stochastic_dynamics(dynamics)
#     .with_stochastic_nonleaf_costs(nonleaf_costs)
#     .with_leaf_cost(leaf_cost)
#     .with_nonleaf_constraint(nonleaf_constraint)
#     .with_leaf_constraint(leaf_constraint)
#     .with_risk(risk)
#     .with_julia()
#     .generate_problem()
# )
#
# # Initial state
# x0 = np.zeros(num_states)
# for k in range(num_states):
#     con = .5 * nonleaf_state_ub[k]
#     x0[k] = rng.uniform(-con, con)
# tree.write_to_file_fp("initialState", x0)
