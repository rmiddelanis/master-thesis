import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import scipy.cluster.hierarchy as sch
import os
import pickle


soll_labels = [
    'Soll_1',
    'Soll_2',
    'Soll_3',
    'Soll_4',
    'Soll_5',
    'Soll_6',
    'Soll_7',
    'Soll_8',
    'Soll_9',
    'Soll_10'
]

ist_labels = [
    'Ist_1',
    'Ist_2',
    'Ist_3',
    'Ist_4',
    'Ist_5',
    'Ist_6',
    'Ist_7',
    'Ist_8',
    'Ist_9',
    'Ist_10',
    'Ist_11',
    'Ist_12',
    'Ist_13',
    'Ist_14',
    'Ist_15',
    'Ist_16',
    'Ist_17',
    'Ist_18',
    'Ist_19',
    'Ist_20',
    'Ist_21',
    'Ist_22',
    'Ist_23',
    'Ist_24',
    'Ist_25',
    'Ist_26',
    'Ist_27',
 ]

dataset_settings = {
    "Synthetic": {
        "path": "./data/Synthetic/synthetic_data.p",
        "dataset_load_func": "get_synthetic_data",
        "subsampling": ['1S', '1H', '1D', '1W', '1M', '1Y'],
    },
}


def plot_time_series(data, path, max_samples, width=10, height=3,):
    for dataset_idx, data_ in enumerate(data):
        fig1, ax1 = plt.subplots(nrows=data_.shape[1], ncols=1, sharex=True, figsize=(width, height * data_.shape[1]),
                                 clear=True, num=1)
        if data_.shape[1]<=1:
            ax1 = [ax1]
        fig2, ax2 = plt.subplots(figsize=(width, height), clear=True, num=2)
        for col_idx, column in enumerate(data_.columns):
            plt.figure(1)
            data_[column].head(max_samples).plot(ax=ax1[col_idx])
            ax1[col_idx].set_title(column)
            ax1[col_idx].set_xlabel("Time [s]")
            ax1[col_idx].set_ylabel(column)
            plt.figure(2)
            ax2.clear()
            data_[column].head(max_samples).plot(ax=ax2)
            ax2.set_title(column)
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel(column)
            fig2.savefig(path + "c_Individual_Plot__" + column.split("[")[0] + ".svg", format = 'svg')
        if dataset_idx == 0:
            fig1.savefig(path + "a_AllSeries_Plot" + ".svg", format='svg')
        else:
            fig1.savefig(path + "b_Partition_" + str(dataset_idx) +"_Plot" + ".svg", format='svg')
    plt.close('all')


def plot_autocorrelation(data, path, max_lag=1000, width=10, height=3,):
    for dataset_idx, data_ in enumerate(data):
        fig1, ax1 = plt.subplots(nrows=data_.shape[1], ncols=1, sharex=True, figsize=(width, height * data_.shape[1]),
                                 clear=True, num=1)
        if data_.shape[1]<=1:
            ax1 = [ax1]
        fig2, ax2 = plt.subplots(figsize=(width, height), clear=True, num=2)
        for col_idx, column in enumerate(data_.columns):
            result, significance = calc_autocorrelation(data_[column].values, max_lag)
            plt.figure(1)
            ax1[col_idx].plot(result)
            ax1[col_idx].axhline(significance, linestyle='-.')
            ax1[col_idx].axhline(-significance, linestyle='-.')
            ax1[col_idx].set_title(column)
            ax1[col_idx].set_ylabel("Autocorrelation ")
            plt.figure(2)
            ax2.clear()
            ax2.plot(result)
            ax2.axhline(significance, linestyle = '-.')
            ax2.axhline(-significance, linestyle = '-.')
            ax2.set_title(column)
            ax2.set_xlabel("lag")
            ax2.set_ylabel("Autocorrelation ")
            fig2.savefig(path + "c_Individual_Plot__" + column.split("[")[0] + ".svg", format = 'svg')
        plt.figure(1)
        plt.xlabel("Lag")
        if dataset_idx == 0:
            fig1.savefig(path + "a_AllSeries_Plot" + ".svg", format='svg')
        else:
            fig1.savefig(path + "b_Partition_" + str(dataset_idx) +"_Plot" + ".svg", format='svg')
    plt.close('all')


def calc_autocorrelation(series, max_k):
    series_zero_mean = series - np.mean(series)
    if max_k > len(series):
        print('Warning: maximum lag max_k should be smaller than the length of the series for which autocorrelation shall be '
                         'computed. Using length-1 of the series instead.')
    denom = np.dot(series_zero_mean, series_zero_mean)
    if denom != 0:
        result = [np.dot(series_zero_mean[k:], series_zero_mean[:-k])/denom for k in range(1, min(max_k, len(series)-1))]
    else:
        result = [1 for k in range(1, min(max_k, len(series)-1))]
    significance = 2/np.sqrt(len(series))
    return result, significance


def plot_scatter_matrix(data, path, scatter_sample_size):
    for dataset_idx, data_ in enumerate(data):
        if len(data_)>scatter_sample_size:
            sns.pairplot(data_.sample(scatter_sample_size), kind='reg', markers='.', diag_kind="kde",
                         plot_kws={
                             'line_kws':{'linewidth':1},
                             'scatter_kws':{'s':0.5, 'linewidth': 1, 'color':'black',}
                         })
        if dataset_idx == 0:
            plt.savefig(path + "a_AllSeries_Scatter_Matrix" + ".svg", format='svg')
        else:
            plt.savefig(path + "b_Partition" + str(dataset_idx) +"_Scatter_Matrix" + ".svg", format='svg')
        plt.close('all')


def plot_correlations(data, path):
    ordered_dataframes = order_dataframes(data)
    correlations = [ordered_dataframes[i].corr() for i in range(len(ordered_dataframes))]
    plot_heatmaps(correlations, path)


def order_dataframes(data):
    result = []
    for df in data:
        correlation = df.corr().dropna(how='all').dropna(axis=1, how='all')
        distances = sch.distance.pdist(correlation)
        linkage = sch.linkage(distances, method='complete')
        indices = sch.fcluster(linkage, 0.5 * distances.min(), 'distance')
        columns = [correlation.columns.tolist()[i] for i in list((np.argsort(indices)))]
        df = df.drop([column for column in df.columns if column not in columns], axis = 1)
        df = df.reindex(columns, axis=1)
        result.append(df)
    return result


def plot_heatmaps(data, path, size=10):
    for idx, df in enumerate(data):
        if idx==0:
            title = "Correlation All Series"
        else:
            title = "Correlation Partial Set " + str(idx)
        fig, ax = plt.subplots(figsize=(size, size))
        cax = ax.matshow(df, cmap='RdYlGn')
        plt.xticks(range(len(df.columns)), df.columns, rotation=90)
        plt.yticks(range(len(df.columns)), df.columns)

        # Add the colorbar legend
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
        fig.savefig(path + title.replace(" ", "_") + ".svg", format = 'svg')
        plt.close('all')


def get_simply_cozy_data():
    complete_data = pickle.load(open(dataset_settings["Simply_Cozy"]["path"], "rb"))
    complete_data['t[s]'] = pd.date_range(start = '2019-01-01', periods = complete_data.shape[0], freq = '1s')
    complete_data.set_index('t[s]', inplace = True)
    complete_data.drop('data_id', inplace = True, axis = 1)
    all_data_x = complete_data[ist_labels]
    all_data_y = complete_data.drop(ist_labels, axis = 1)
    all_data_x.columns = ['Ist_'+label for label in all_data_x.columns]
    all_data_y.columns = ['Soll_'+label for label in all_data_y.columns]
    return [all_data_x, all_data_y]


def get_household_power_data():
    complete_data = pd.read_pickle(dataset_settings["Household_Power"]["path"])
    complete_data = complete_data.loc[:,:complete_data.columns[7]]
    complete_data.set_index('t[s]', inplace=True)
    return [complete_data]


def get_synthetic_data():
    complete_data = pickle.load(open(dataset_settings["Synthetic"]["path"], "rb"))
    complete_data['t[s]'] = pd.date_range(start='2019-01-01', periods=complete_data.shape[0], freq='1s')
    complete_data.set_index('t[s]', inplace=True)
    complete_data.drop('data_id', inplace=True, axis=1)
    all_data_x = complete_data["Ist"]
    all_data_y = complete_data["Soll"]
    all_data_x.columns = [all_data_x.columns[i] + str(i + 1) for i in range(len(all_data_x.columns))]
    all_data_y.columns = [all_data_y.columns[i] + str(i + 1) for i in range(len(all_data_y.columns))]
    return [all_data_x, all_data_y]


if __name__ == "__main__":
    root_dir = "./data/"
    max_samples_ = int(1e3)
    scatter_sample_size = int(1e3)
    for key, value in dataset_settings.items():
        dataset_name = key
        load_func = value['dataset_load_func']
        subsampling_frequencies = value['subsampling']
        corr_dir = root_dir + dataset_name + "/Dataset_Analysis/Correlation/"
        if not os.path.exists(corr_dir):
            os.makedirs(corr_dir)
        temporal_plots_dir = root_dir + dataset_name + "/Dataset_Analysis/Temporal_Plots/"
        if not os.path.exists(temporal_plots_dir):
            os.makedirs(temporal_plots_dir)
        data_ = globals()[load_func]()
        if len(data_) > 1:
            data_ = [pd.concat(data_, axis=1)] + data_
        for frequency in subsampling_frequencies:
            corr_freq_dir = root_dir + dataset_name + "/Dataset_Analysis/Correlation/Sampling_"+frequency+"/"
            if not os.path.exists(corr_freq_dir):
                os.makedirs(corr_freq_dir)
            temporal_plots_freq_dir = root_dir + dataset_name + "/Dataset_Analysis/Temporal_Plots/Sampling_"+frequency+"/"
            auto_corr_freq_dir = root_dir + dataset_name + "/Dataset_Analysis/AutoCorrelation/Sampling_" + frequency + "/"
            if not os.path.exists(auto_corr_freq_dir):
                os.makedirs(auto_corr_freq_dir)
            temporal_plots_freq_dir = root_dir + dataset_name + "/Dataset_Analysis/Temporal_Plots/Sampling_" + frequency + "/"
            if not os.path.exists(temporal_plots_freq_dir):
                os.makedirs(temporal_plots_freq_dir)
            for dataset_idx, dataset in enumerate(data_):
                data_[dataset_idx] = dataset.resample(frequency).mean()
            plot_autocorrelation(data_, auto_corr_freq_dir)
            plot_time_series(data=data_, path=temporal_plots_freq_dir, max_samples=max_samples_)