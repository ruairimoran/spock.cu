import numpy as np
import argparse
import factories as f
import pandas as pd


def check_spd(mat, name):
    eigs = np.linalg.eigvals(mat)
    is_positive_definite = eigs.all() > 0
    is_symmetric = np.allclose(mat, mat.T)
    print("Is " + name + " symmetric positive-definite?", is_positive_definite and is_symmetric,
          " with norm (", np.linalg.norm(mat), ").")


def plot_scenario_values(ax_, data_, branching_, times_, lines_):
    tree_ = f.tree.FromData(data_, branching_).build(False)
    print(tree_)
    scenarios_ = tree_.get_scenarios()
    values_ = tree_.data_values
    num_scenarios_ = len(scenarios_)
    len_scenario_ = scenarios_[0].size
    for line_idx in range(len(lines_)):  # Clear lines
        lines_[line_idx].set_xdata([])
        lines_[line_idx].set_ydata([])
    for line_idx in range(num_scenarios_):  # Plot scenarios
        lines_[line_idx].set_xdata(times_[:len_scenario_])
        lines_[line_idx].set_ydata(values_[scenarios_[line_idx]])
    set_ticks(ax_, times_)


def set_ticks(ax_, times_):
    x_ticks_ = np.arange(times_[0], times_[-1], 200)  # tick labels every 2 hours
    ax_.set_xticks(x_ticks_)
    ax_.set_xticklabels([f"{((t % 2400) // 100) + ((t % 100) * .006):02.2f}" for t in x_ticks_])
    ax_.set_xlim(min(times_), max(times_) + 25)


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

# ---- Read file ----
print("Read...")
folder = "examples/powerDistribution/niEnergyData/"
wind_df = pd.read_csv(folder + "soni_data.csv", parse_dates=["date&time"])
wind_df.columns = wind_df.columns.str.strip()  # Remove extra spaces
wind_df["hour"] = wind_df["date&time"].dt.hour + wind_df["date&time"].dt.minute / 60  # Convert to hours
wind_df["wind actual (MW)"] = pd.to_numeric(wind_df["wind actual (MW)"], errors="raise")
wind_df["wind forecast (MW)"] = pd.to_numeric(wind_df["wind forecast (MW)"], errors="raise")
wind_df["error"] = wind_df["wind actual (MW)"] - wind_df["wind forecast (MW)"]
print("Sanitize...")
wind_df["date"] = wind_df["date&time"].dt.date  # Extract date only
wind_daily = wind_df.groupby("date")  # Group by date
max_time_steps = wind_daily.size().max()
err_wind_list = []
for _, group in wind_daily:
    wind_daily_err = group.sort_values("hour")["error"].values  # Sort by time of day
    if len(wind_daily_err) < max_time_steps or np.isnan(wind_daily_err).any():
        # print("Missing value! Skipping bad sample...")
        continue
    err_wind_list.append(wind_daily_err)
err_wind = np.array(err_wind_list).reshape(len(err_wind_list), 1, max_time_steps)
print(f"Done: ({err_wind.shape[0]}) wind samples.")

# ---- Create tree from data ----
horizon = max_time_steps - 1  # 15 minute periods
branching = np.ones(horizon, dtype=np.int32)
branching[0] = 3
for i in range(1, len(branching)):
    if i % 24 == 0:
        branching[i] = 2

# start = 4  # 15 minute periods
# data = err_wind  # samples x dim x time
# tree = f.tree.FromData(data, branching).build()
# scenarios = tree.get_scenarios()
# values = tree.data_values

# ---- Plot tree ----
# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation

# ---- Plot data ----
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 7))
ax.set_xlabel("Time (24 hr)")
ax.set_ylabel("Wind Forecast Error (MW)")
plt.title("Wind Forecast Error vs. Time")
ax.grid(True)
start = 0  # 15 minute periods
num_samples = 365
data = np.concatenate((
    err_wind[-num_samples:, :, start:],
    err_wind[-num_samples:, :, :start]), axis=2)  # samples x dim x time
times = .25 * (np.arange(start, start + max_time_steps)) * 100
times = [t if t < 2400 else t - 2400 for t in times]  # 24 hrs
lines = [None for _ in range(num_samples)]
for i in range(num_samples):
    lines[i], = ax.plot([], [], marker="o", linestyle="-")
set_ticks(ax, times)
ax.set_ylim([min(wind_df["error"]), max(wind_df["error"])])
jump = 8  # 15 minute periods
data_shape = data.shape
for _ in range(0, 200):
    print("Building tree and plotting values...")
    plot_scenario_values(ax, data, branching, times, lines)
    # data = np.concatenate((data[:, :, 1:], data[:, :, 0].reshape(-1, 1, 1)), axis=2)
    data = data.reshape(-1)
    data = np.concatenate((data[jump:], data[:jump]))
    data = data.reshape(data_shape)
    times = np.concatenate((times[jump:], [t + 2400 for t in times[:jump]]))
    plt.pause(.1)
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
