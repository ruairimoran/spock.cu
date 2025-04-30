import numpy as np
import argparse
import pandas as pd
import pickle
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


def compute_multiplier(df):
    if np.any(np.isclose(df["forecast"], 0.)):
        raise Exception("Trying to divide by zero!")
    return df["actual"] / df["forecast"]


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
        df_err_per_day = group.sort_values("time")["multiplier"].values  # Sort by time of day
        df_err_per_day = daylight_savings(df_err_per_day, max_steps)
        if len(df_err_per_day) != max_steps or np.isnan(df_err_per_day).any():
            print("Bad sample!")
            print(group)
            continue
        df_list.append(df_err_per_day.reshape(1, 1, -1))
    return np.vstack(df_list)


parser = argparse.ArgumentParser(description='Example: power distribution: generate scenario tree.')
parser.add_argument("--dt", type=str, default='d')
parser.add_argument("--br", type=int, default=0)
parser.add_argument("--ch", type=int, default=2)
parser.add_argument("--tree", type=int, default=1)
args = parser.parse_args()
dt = args.dt
br = args.br
ch = args.ch
make_tree = args.tree

max_time_steps = 24  # 1-hour sampling time
folder = "deEnergyData/"

if make_tree:
    # --------------------------------------------------------
    # Read demand data
    # --------------------------------------------------------
    print("Read demand...")
    demand = pd.DataFrame()
    # Actual
    demand_actual = pd.read_csv(folder + "demandActual24.csv", sep=";")
    demand["date&time"] = pd.to_datetime(demand_actual["Start date"], format="%b %d, %Y %I:%M %p")
    demand["actual"] = pd.to_numeric(demand_actual["grid load [MWh]"].str.replace(",", "", regex=True), errors="raise")
    # Forecast
    demand_forecast = pd.read_csv(folder + "demandForecast24.csv", sep=";")
    demand["forecast"] = pd.to_numeric(demand_forecast["grid load [MWh]"].str.replace(",", "", regex=True), errors="raise")
    # Error
    demand["multiplier"] = compute_multiplier(demand)
    # Sanitize and group by day
    err_demand = sanitize_and_group(demand, max_time_steps)
    print(f"Done: ({err_demand.shape[0]}) demand samples.")

    # --------------------------------------------------------
    # Read renewables data
    # --------------------------------------------------------
    print("Read renewables...")
    renewables = pd.DataFrame()
    # Actual
    renewables_actual = pd.read_csv(folder + "renewablesActual24.csv", sep=";")
    renewables["date&time"] = pd.to_datetime(renewables_actual["Start date"], format="%b %d, %Y %I:%M %p")
    renewables["offshore"] = pd.to_numeric(renewables_actual["Wind offshore [MWh]"].str.replace(",", "", regex=True), errors="raise")
    renewables["onshore"] = pd.to_numeric(renewables_actual["Wind onshore [MWh]"].str.replace(",", "", regex=True), errors="raise")
    renewables["pv"] = pd.to_numeric(renewables_actual["Photovoltaics [MWh]"].str.replace(",", "", regex=True), errors="raise")
    renewables["actual"] = renewables["offshore"] + renewables["onshore"] + renewables["pv"]
    # Forecast
    renewables_forecast = pd.read_csv(folder + "renewablesForecast24.csv", sep=";")
    renewables["forecast"] = pd.to_numeric(renewables_forecast["Photovoltaics and wind [MWh]"].str.replace(",", "", regex=True), errors="raise")
    # Error
    renewables["multiplier"] = compute_multiplier(renewables)
    # Sanitize and group by day
    err_renewables = sanitize_and_group(renewables, max_time_steps)
    print(f"Done: ({err_renewables.shape[0]}) renewables samples.")

    # --------------------------------------------------------
    # Read price data
    # --------------------------------------------------------
    print("Read price...")
    price = pd.DataFrame()
    # Actual
    price_actual = pd.read_csv(folder + "priceActual24.csv", sep=";")
    price["date&time"] = pd.to_datetime(price_actual["date"] + " " + price_actual["from"], dayfirst=True)
    price["actual"] = pd.to_numeric(price_actual["price [ct/kWh]"], errors="raise") * 10  # euro/MWh
    # Forecast
    price_forecast = pd.read_csv(folder + "priceForecast24.csv", sep=";")
    price["forecast"] = pd.to_numeric(price_forecast["price [ct/kWh]"], errors="raise") * 10  # euro/MWh
    # Error
    price["multiplier"] = compute_multiplier(price)
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
    horizon = 23  # max_time_steps - 1  # 1 hour periods
    branching = np.ones(horizon, dtype=np.int32)
    match br:
        case 0:
            branching[0:2] = [ch, ch]
        case 1:
            branching[0] = np.power(ch, 2)
    data = err_samples
    tree = s.tree.FromData(data, branching).build()
    with open('tree.pkl', 'wb') as f:
        pickle.dump(tree, f)
else:
    with open('tree.pkl', 'rb') as f:
        tree = pickle.load(f)
print(tree)

# --------------------------------------------------------
# Plot scenarios
# --------------------------------------------------------
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
# start = 0  # 1 hour periods
# times = np.arange(start, start + max_time_steps) * 100
# titles = ["demand", "renewables", "price"]
# axes[1].set_ylabel("Error (1. = 100% of forecast value)")
# for i, ax in enumerate(axes):
#     ax.set_xlabel("Time (24 hr)")
#     ax.set_title(f"Multiplier vs. Time ({titles[i]})")
#     ax.grid(True)
#     plot_scenario_values(ax, tree, times, i)
#     set_ticks(ax, times)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------
# Read forecast for problem
# --------------------------------------------------------
idx_d = 0
idx_r = 1
idx_p = 2
fc = pd.read_csv(folder + "demandForecast25.csv", sep=";")
fc["forecast"] = pd.to_numeric(fc["grid load [MWh]"].str.replace(",", "", regex=True), errors="raise")
fc_d = np.array(fc["forecast"] / 1e3)
fc = pd.read_csv(folder + "renewablesForecast25.csv", sep=";")
fc["forecast"] = pd.to_numeric(fc["Photovoltaics and wind [MWh]"].str.replace(",", "", regex=True), errors="raise")
fc_r = np.array(fc["forecast"] / 5e3)
fc = pd.read_csv(folder + "priceForecast25.csv", sep=";")
fc["forecast"] = pd.to_numeric(fc["price [ct/kWh]"], errors="raise") * 10  # euro/MWh
fc_p = np.array(fc["forecast"] / 1e3)

# print(np.mean(fc_r - fc_d), np.std(fc_r - fc_d))

# plt.plot(fc_p[:24])
# plt.plot(fc_r[:24])
# plt.plot(fc_d[:24])
# plt.show()

# --------------------------------------------------------
# Generate problem data
# state = [x, dx, p-] = [stored energy, change in stored energy, prev conventional power]
# input = [s, p, m] = [charge, conventional power, exchanged power]
# --------------------------------------------------------
n_s = 100  # number of storage units. CAUTION! MUST MAKE NON-RECURRING ENTRIES IN BETA!
n_p = 2  # number of conventional generators
n_m = 1  # number of markets
beta = np.ones((n_s, 1)) * 1 / n_s  # relative sizes of storage units. CAUTION! MUST BE NON-RECURRING ENTRIES!
T = 1.  # sampling time (hours)
fuel_cost = 60. / 1e3  # euro/MWh

num_states = n_s + n_s + n_p
num_inputs = n_s + n_p + n_m

# Dynamics
dynamics = [None]
eye_s = np.eye(n_s)
eye_p = np.eye(n_p)
zero_p = np.zeros((n_p, n_m))
A = .98 * eye_s
A_aug = np.zeros((num_states, num_states))
A_aug[:n_s, :n_s] = A
A_aug[n_s:n_s*2, :n_s] = A - eye_s
B = np.hstack((
    eye_s - beta @ np.ones((1, n_s)),
    beta @ np.ones((1, n_p)),
    beta,
)) * T
B_ = np.hstack((
    np.zeros((n_p, n_s)),
    eye_p,
    zero_p,
))
B_aug = np.vstack((B, B, B_))
for node in range(1, tree.num_nodes):
    renewables_node = fc_r[tree.stage_of_node(node)] * tree.data_values[node, idx_r]
    demand_node = fc_d[tree.stage_of_node(node)] * tree.data_values[node, idx_d]
    c = T * beta * (renewables_node - demand_node)
    c_aug = np.vstack((c, c, zero_p))
    dynamics += [s.build.Dynamics(A_aug, B_aug, c_aug)]

# Costs
nonleaf_costs = [None]
zero = 1e-3
Q = np.eye(num_states) * zero
R = np.eye(num_inputs) * zero
q = np.zeros(num_states)
for node in range(1, tree.num_nodes):
    price_node = fc_p[tree.stage_of_node(node)] * tree.data_values[node, idx_p]
    r = np.concatenate((
        np.zeros(n_s),
        np.ones(n_p) * fuel_cost,
        np.array([T * price_node])), axis=0
    ).reshape(-1, 1)
    nonleaf_costs += [s.build.CostQuadraticPlusLinear(Q, q, R, r)]
leaf_cost = s.build.CostQuadraticPlusLinear(Q, q, leaf=True)

# Constraints
# big = 5e3  # MWh
stored_energy_lb = np.ones(n_s) * .01  # MWh
stored_energy_ub = np.ones(n_s) * 1.  # MWh
stored_energy_rate_lb = np.ones(n_s) * -1.  # MWh
stored_energy_rate_ub = np.ones(n_s) * 1.  # MWh
charge_rate_lb = np.ones(n_s) * -1.  # MW
charge_rate_ub = np.ones(n_s) * 1.  # MW
conventional_supply_lb = np.ones(n_p) * 10.  # MW
conventional_supply_ub = np.ones(n_p) * 50.  # MW
exchange_lb = np.ones(n_m) * -20.  # MW
exchange_ub = np.ones(n_m) * 20.  # MW
conventional_supply_rate_lb = np.ones(n_p) * -50.  # MW
conventional_supply_rate_ub = np.ones(n_p) * 50.  # MW

nonleaf_rect_lb = np.hstack((
    stored_energy_lb,
    stored_energy_rate_lb,
    conventional_supply_lb,
    charge_rate_lb,
    conventional_supply_lb,
    exchange_lb,
))
nonleaf_rect_ub = np.hstack((
    stored_energy_ub,
    stored_energy_rate_ub,
    conventional_supply_ub,
    charge_rate_ub,
    conventional_supply_ub,
    exchange_ub,
))
nonleaf_rect = s.build.Rectangle(nonleaf_rect_lb, nonleaf_rect_ub)
poly_mat_x = np.hstack((np.zeros((n_p, n_s * 2)), -eye_p))
poly_mat_u = np.hstack((np.zeros((n_p, n_s)), eye_p, zero_p))
poly_mat = np.hstack((poly_mat_x, poly_mat_u))
nonleaf_poly = s.build.Polyhedron(poly_mat, conventional_supply_rate_lb, conventional_supply_rate_ub)
nonleaf_constraint = s.build.PolyhedronWithIdentity(nonleaf_rect, nonleaf_poly)

leaf_lb = np.hstack((
    stored_energy_lb,
    stored_energy_rate_lb,
    conventional_supply_lb,
))
leaf_ub = np.hstack((
    stored_energy_ub,
    stored_energy_rate_ub,
    conventional_supply_ub,
))
leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = 1.
risk = s.build.AVaR(alpha)

# Generate
problem = (
    s.problem.Factory(scenario_tree=tree, num_states=num_states, num_inputs=num_inputs)
    .with_dynamics_list(dynamics)
    .with_cost_nonleaf_list(nonleaf_costs)
    .with_cost_leaf(leaf_cost)
    .with_constraint_nonleaf(nonleaf_constraint)
    .with_constraint_leaf(leaf_constraint)
    .with_risk(risk)
    .with_julia()
    .with_preconditioning(True)
    .generate_problem()
)
print(problem)

# --------------------------------------------------------
# Initial state
# --------------------------------------------------------
if br == 0:
    x0 = np.zeros(num_states)
    for k in range(num_states):
        x0[k] = .5 * (leaf_lb[k] + leaf_ub[k])
        if x0[k] != 0.:
            x0[k] = leaf_lb[k]
    tree.write_to_file_fp("initialState", x0)
