import requests  # requires you to manually `pip install brotli`
import os
import csv

# Wind data
# https://www.soni.ltd.uk/api/graph-data?area=windactual&region=NI&date=01%20Jan%202015
# https://www.soni.ltd.uk/api/graph-data?area=windforecast&region=NI&date=01%20Jan%202015
# Demand data
# https://www.soni.ltd.uk/api/graph-data?area=demandactual&region=NI&date=01%20Jan%202015
# https://www.soni.ltd.uk/api/graph-data?area=demandforecast&region=NI&date=01%20Jan%202015

fields_soni = ["windactual", "windforecast"]  # , "demandactual", "demandforecast"]
fields = ["date&time", "wind actual (MW)", "wind forecast (MW)"]  # , "demand actual (MW)", "demand forecast (MW)"]
file = "soni_data.csv"
base_url = "https://www.soni.ltd.uk/api/graph-data?area={}&region=NI&date={}"
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def append_data(file_name, field_idx, json_data):
    field_idx += 1
    data_dict = {}
    if os.path.exists(file_name):
        with open(file_name, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_dict[row[fields[0]]] = row
    else:
        raise Exception(f"File ({file_name}) does not exist!")

    # Update data_dict with json values.
    for row in json_data['Rows']:
        time_val = row['EffectiveTime']
        if time_val in data_dict:
            data_dict[time_val][fields[field_idx]] = row['Value']
        else:
            data_dict[time_val] = {
                fields[0]: time_val,
                fields[field_idx]: row['Value'],
            }

    # Write (or rewrite) the CSV file.
    with open(file_name, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for time_val in data_dict.keys():
            writer.writerow(data_dict[time_val])


# Scrape data and append to file per day
# for year in range(2015, 2025):
year = 2024
for month in range(0, 12):
    for day in range(1, 32):
        date = f"{day:02d}%20{months[month]}%20{year}"
        print(f"Reading in ({date}) data...")
        for idx in range(len(fields_soni)):
            address = base_url.format(fields_soni[idx], date)
            response = requests.get(address)
            if response.status_code == 200:
                data = response.json()
                append_data(file, idx, data)
            else:
                raise Exception(f"Response status code ({response.status_code}) "
                                f"on day ({date}) in field ({fields_soni[idx]}).")
