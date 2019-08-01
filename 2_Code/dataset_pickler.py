import pandas as pd

dataset_locations = {
    'household_power_consumption': r'./data/household_power_consumption.txt'
}

# see https://www.kaggle.com/amirrezaeian/time-series-data-analysis-using-lstm-tutorial/notebook
def pickle_household_dataset(output_filename = "./data/household_power_consumption.p", delta_t = 1):
    dataframe = pd.read_csv(dataset_locations['household_power_consumption'], sep = ';', na_values=['nan','?'],
                            parse_dates={'t[s]' : ['Date', 'Time']}, infer_datetime_format=True)
    num_features = dataframe.shape[1]
    for j in range(1, num_features):
        dataframe.iloc[:, j] = dataframe.iloc[:, j].fillna(dataframe.iloc[:, j].mean())
        dataframe[dataframe.columns[j] + ' [Soll]'] = dataframe[dataframe.columns[j]].shift(-delta_t)
        dataframe.rename(columns = {dataframe.columns[j]: dataframe.columns[j] + ' [Ist]'}, inplace = True)
    dataframe = dataframe.head(-delta_t)
    dataframe['t[s]'] = pd.to_datetime(dataframe['t[s]'])
    # dataframe['data_id'] = dataframe['t[s]'].dt.date
    dataframe['data_id'] = dataframe['t[s]'].dt.strftime("%Y-%m")
    dataframe.to_pickle(output_filename)

def pickle_household_dataset_diff(output_filename = "./data/household_power_consumption_diff.p"):
    dataframe = pd.read_csv(dataset_locations['household_power_consumption'], sep = ';', na_values=['nan','?'],
                            parse_dates={'t[s]' : ['Date', 'Time']}, infer_datetime_format=True)
    num_features = dataframe.shape[1]
    for j in range(1, num_features):
        dataframe.iloc[:, j] = dataframe.iloc[:, j].fillna(dataframe.iloc[:, j].mean())
        diff_series = dataframe[dataframe.columns[j]].values[1:] - dataframe[dataframe.columns[j]].values[:-1]
        dataframe[dataframe.columns[j] + ' [Soll]'] = diff_series.tolist() + [0]
        dataframe.rename(columns = {dataframe.columns[j]: dataframe.columns[j] + ' [Ist]'}, inplace = True)
    dataframe = dataframe.head(len(dataframe)-1)
    dataframe['t[s]'] = pd.to_datetime(dataframe['t[s]'])
    # dataframe['data_id'] = dataframe['t[s]'].dt.date
    dataframe['data_id'] = dataframe['t[s]'].dt.strftime("%Y-%m")
    dataframe.to_pickle(output_filename)