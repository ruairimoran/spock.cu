import numpy as np
import argparse
import pandas as pd
import spock as s


def plot_scenario_values(ax_, data_, branching_, times_, lines_):
    tree_ = s.tree.FromData(data_, branching_).build(False)
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


def compute_error(df):
    return (df["actual"] - df["forecast"]) / df["actual"]  # should this be "/ forecast" ?


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
# Combine samples: [demand, renewables, price]
# --------------------------------------------------------
err_samples = np.concatenate((err_demand, err_renewables, err_price), axis=1)



# ---- Create tree from data ----
# horizon = max_time_steps - 1  # 15 minute periods
# branching = np.ones(horizon, dtype=np.int32)
# branching[0] = 3
# for i in range(1, len(branching)):
#     if i % 24 == 0:
#         branching[i] = 2

# start = 4  # 15 minute periods
# data = err_wind  # samples x dim x time
# tree = s.tree.FromData(data, branching).build()
# scenarios = tree.get_scenarios()
# values = tree.data_values

# ---- Plot tree ----
# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation

# ---- Plot data ----
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(figsize=(15, 7))
# ax.set_xlabel("Time (24 hr)")
# ax.set_ylabel("Wind Forecast Error (MW)")
# plt.title("Wind Forecast Error vs. Time")
# ax.grid(True)
# start = 0  # 15 minute periods
# num_samples = 365
# data = np.concatenate((
#     err_wind[-num_samples:, :, start:],
#     err_wind[-num_samples:, :, :start]), axis=2)  # samples x dim x time
# times = .25 * (np.arange(start, start + max_time_steps)) * 100
# times = [t if t < 2400 else t - 2400 for t in times]  # 24 hrs
# lines = [None for _ in range(num_samples)]
# for i in range(num_samples):
#     lines[i], = ax.plot([], [], marker="o", linestyle="-")
# set_ticks(ax, times)
# ax.set_ylim([min(wind_df["error"]), max(wind_df["error"])])
# jump = 8  # 15 minute periods
# data_shape = data.shape
# for _ in range(0, 200):
#     print("Building tree and plotting values...")
#     plot_scenario_values(ax, data, branching, times, lines)
#     # data = np.concatenate((data[:, :, 1:], data[:, :, 0].reshape(-1, 1, 1)), axis=2)
#     data = data.reshape(-1)
#     data = np.concatenate((data[jump:], data[:jump]))
#     data = data.reshape(data_shape)
#     times = np.concatenate((times[jump:], [t + 2400 for t in times[:jump]]))
#     plt.pause(.1)
# plt.show()
