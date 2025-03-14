import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import os
import csv


def fit(arimaModel):
    trainedArimaModel = arimaModel.fit(method="innovations_mle")
    return trainedArimaModel


def calculatePrmse(trainedModel, testData, nTestSamples, forecastHorizon):
    prmse = np.zeros(nTestSamples)
    for k in range(nTestSamples):
        forecast = trainedModel.forecast(forecastHorizon)
        actual = testData[k : k + forecastHorizon]
        error = actual - forecast
        prmse[k] = np.sqrt(1 / forecastHorizon * sum(error**2))
        trainedModel = trainedModel.extend(testData[k][None])
    return prmse


path = os.getcwd()
path_to_data = os.path.join(path, "priceData2019To2024.csv")
price_df = pd.read_csv(path_to_data, sep=";")
price_df.columns = price_df.columns.str.strip()
price_df["time"] = pd.to_datetime(price_df["Datum"] + " " + price_df["von"], format="%d.%m.%Y %H:%M")
start_time = price_df["time"].min()
price_df["hours"] = (price_df["time"] - start_time).dt.total_seconds() / 60 / 60  # time in hours since beginning of file
price_df["price"] = price_df["Spotmarktpreis in ct/kWh"]
valid_rows = price_df[np.isfinite(price_df["price"])]
time = np.array(valid_rows["hours"].tolist())
price = np.array(valid_rows["price"].tolist())

SAMPLES_PER_DAY = 24
PREDICTION_HORIZON = 24

# Test data
nTestSamples = int(366 * SAMPLES_PER_DAY)  # 2024 was a leap year
testData = price[-nTestSamples:]
testTime = time[-nTestSamples:]
nTest = nTestSamples - PREDICTION_HORIZON

# Training data
nTrainingSamples = price.size - nTestSamples
trainingData = price[:nTrainingSamples]
trainingTime = time[:nTrainingSamples]  # 1 hour sampling time

# Compute means
path_to_file = os.path.join(path, "modelMeans.csv")
for t in ["n", "t"]:
    for s in [24, 168]:
        for p in np.arange(0, 6):
            for d in np.arange(0, 2):
                for q in np.arange(0, 6):
                    for P in np.arange(0, 6):
                        for D in np.arange(0, 2):
                            for Q in np.arange(0, 6):
                                print(f"Building ARIMA with params: [({p}, {d}, {q}), ({P}, {D}, {Q}), {s}, {t}] ...")
                                try:
                                    arima = ARIMA(endog=trainingData, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=t)
                                    arimaTrained = fit(arima)
                                    prmse = calculatePrmse(arimaTrained, testData, nTest, PREDICTION_HORIZON)
                                    mean = np.mean(prmse)
                                    with open(path_to_file, mode="a", newline="") as f:
                                        writer = csv.writer(f)
                                        writer.writerow([p, d, q, P, D, Q, s, t, mean])
                                except Exception as e:
                                    print("Error, moving on...")

# possible alternative:  arima = ARIMA(endog=trainingData, order=(2,1,2), seasonal_order=(0,0,0,0), exog=trainingData.shift([24, 168]))
