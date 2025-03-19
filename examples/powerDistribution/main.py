import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import spock as s


def plot_scenario_values(ax_, tree_, times_, var=0):
    print(tree_)
    scenarios_ = tree_.get_scenarios()
    values_ = tree_.data_values
    num_scenarios_ = len(scenarios_)
    len_scenario_ = scenarios_[0].size
    for scen in range(num_scenarios_):
        ax_.plot(times_[:len_scenario_], values_[scenarios_[scen], var])


def set_ticks(ax_, times_):
    x_ticks_ = np.arange(times_[0], times_[-1], 200)  # tick labels every 2 hours
    ax_.set_xticks(x_ticks_)
    ax_.set_xticklabels([f"{((t % 2400) // 100) + ((t % 100) * .006):02.2f}" for t in x_ticks_])
    ax_.set_xlim(min(times_), max(times_) + 25)


def compute_error(df):
    if np.any(np.isclose(df["forecast"], 0.)):
        raise Exception("Trying to divide by zero!")
    return (df["actual"] - df["forecast"]) / abs(df["forecast"])  # should this be "/ actual" ?


def daylight_savings(arr, max_t):
    if len(arr) == max_t - 1:
        arr = np.append(arr, arr[-1])  # Duplicate last value
    elif len(arr) == max_t + 1:
        arr = arr[:-1]  # Remove last value
    return arr


def sanitize_and_group(df, max_steps):
    print("Sanitize...")
    df["date"] = df["date&time"].dt.date  # Extract date only
    df["time"] = df["date&time"].dt.time  # Extract date only
    df_per_day = df.groupby("date")  # Group by date
    df_list = []
    for _, group in df_per_day:
        df_err_per_day = group.sort_values("time")["error"].values  # Sort by time of day
        df_err_per_day = daylight_savings(df_err_per_day, max_steps)
        if len(df_err_per_day) != max_steps or np.isnan(df_err_per_day).any():
            print("Bad sample!")
            print(group)
            continue
        df_list.append(df_err_per_day.reshape(1, 1, -1))
    return np.vstack(df_list)


parser = argparse.ArgumentParser(description='Example: power distribution: generate scenario tree.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

max_time_steps = 24  # 1-hour sampling time
folder = "deEnergyData/"

# --------------------------------------------------------
# Read demand data
# --------------------------------------------------------
print("Read demand...")
demand = pd.DataFrame()
# Actual
demand_actual = pd.read_csv(folder + "demandActual.csv", sep=";")
demand["date&time"] = pd.to_datetime(demand_actual["Start date"], format="%b %d, %Y %I:%M %p")
demand["actual"] = pd.to_numeric(demand_actual["grid load [MWh]"].str.replace(",", "", regex=True), errors="raise")
# Forecast
demand_forecast = pd.read_csv(folder + "demandForecast.csv", sep=";")
demand["forecast"] = pd.to_numeric(demand_forecast["grid load [MWh]"].str.replace(",", "", regex=True), errors="raise")
# Error
demand["error"] = compute_error(demand)
# Sanitize and group by day
err_demand = sanitize_and_group(demand, max_time_steps)
print(f"Done: ({err_demand.shape[0]}) demand samples.")

# --------------------------------------------------------
# Read renewables data
# --------------------------------------------------------
print("Read renewables...")
renewables = pd.DataFrame()
# Actual
renewables_actual = pd.read_csv(folder + "renewablesActual.csv", sep=";")
renewables["date&time"] = pd.to_datetime(renewables_actual["Start date"], format="%b %d, %Y %I:%M %p")
renewables["offshore"] = pd.to_numeric(renewables_actual["Wind offshore [MWh]"].str.replace(",", "", regex=True), errors="raise")
renewables["onshore"] = pd.to_numeric(renewables_actual["Wind onshore [MWh]"].str.replace(",", "", regex=True), errors="raise")
renewables["pv"] = pd.to_numeric(renewables_actual["Photovoltaics [MWh]"].str.replace(",", "", regex=True), errors="raise")
renewables["actual"] = renewables["offshore"] + renewables["onshore"] + renewables["pv"]
# Forecast
renewables_forecast = pd.read_csv(folder + "renewablesForecast.csv", sep=";")
renewables["forecast"] = pd.to_numeric(renewables_forecast["Photovoltaics and wind [MWh]"].str.replace(",", "", regex=True), errors="raise")
# Error
renewables["error"] = compute_error(renewables)
# Sanitize and group by day
err_renewables = sanitize_and_group(renewables, max_time_steps)
print(f"Done: ({err_renewables.shape[0]}) renewables samples.")

# --------------------------------------------------------
# Read price data
# --------------------------------------------------------
print("Read price...")
price = pd.DataFrame()
# Actual
price_actual = pd.read_csv(folder + "priceActual.csv", sep=";")
price["date&time"] = pd.to_datetime(price_actual["date"] + " " + price_actual["from"], dayfirst=True)
price["actual"] = pd.to_numeric(price_actual["price [ct/kWh]"], errors="raise")
# Forecast
price_forecast = pd.read_csv(folder + "priceForecast.csv", sep=";")
price["forecast"] = pd.to_numeric(price_forecast["price [ct/kWh]"], errors="raise")
# Error
price["error"] = compute_error(price)
# Sanitize and group by day
err_price = sanitize_and_group(price, max_time_steps)
print(f"Done: ({err_price.shape[0]}) price samples.")

# --------------------------------------------------------
# Combine samples [demand, renewables, price]
# into [samples x dim x time] array
# --------------------------------------------------------
err_samples = np.concatenate((err_demand, err_renewables, err_price), axis=1)

# --------------------------------------------------------
# Create tree from data
# --------------------------------------------------------
horizon = max_time_steps - 1  # 1 hour periods
branching = np.ones(horizon, dtype=np.int32)
branching[0:3] = [5, 5, 5]
data = err_samples
tree = s.tree.FromData(data, branching).build()
print(tree)

# --------------------------------------------------------
# Plot scenarios
# --------------------------------------------------------
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
# start = 0  # 1 hour periods
# times = np.arange(start, start + max_time_steps) * 100
# lims = [[-.2, .2], [-.2, .2], [-5., 5.]]
# titles = ["demand", "renewables", "price"]
# axes[1].set_ylabel("Error (1. = 100% of forecast value)")
# for i, ax in enumerate(axes):
#     ax.set_xlabel("Time (24 hr)")
#     ax.set_title(f"Error vs. Time ({titles[i]})")
#     ax.grid(True)
#     ax.set_ylim(lims[i])
#     plot_scenario_values(ax, tree, times, i)
#     set_ticks(ax, times)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
n_s = 10  # number of storage units
n_p = 3  # number of conventional generators
n_r = 1  # number of renewables
beta = np.ones((n_s, 1)) * 1 / n_s
T = 1  # hour
fuel_cost = 2.

num_states = n_s
num_inputs = n_s + n_p + 1

# Dynamics
A = .99 * np.eye(n_s)
B = np.hstack((
    np.eye(n_s) - beta @ np.ones((1, n_s)),
    beta @ np.ones((1, n_p)),
    beta,
)) * T
c = T * beta
dynamics = s.build.AffineDynamics(A, B)

# Costs
Q = None
R = np.diag(np.stack((np.zeros(n_s), np.ones(n_p) * fuel_cost, -T)))
q = None
r = np.stack((np.zeros(n_s), np.ones(n_p) * fuel_cost, 0.))
nonleaf_costs = s.build.NonleafCost(Q, R, q, r)

# Constraints
nonleaf_state_ub = np.ones(num_states)
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = np.ones(num_inputs)
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = s.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = np.ones(num_states)
leaf_lb = -leaf_ub
leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = .95
risk = s.build.AVaR(alpha)

# Generate
problem = (
    s.problem.Factory(scenario_tree=tree, num_states=num_states, num_inputs=num_inputs)
    .with_dynamics(dynamics)
    .with_nonleaf_cost(nonleaf_costs)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .with_julia()
    .generate_problem()
)
print(problem)

# --------------------------------------------------------
# Initial state
# --------------------------------------------------------
x0 = np.zeros(num_states)
tree.write_to_file_fp("initialState", x0)
