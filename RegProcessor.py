import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

class RegProcessor(object):
    def __init__(self, reg_file_1, reg_file_2):
        df_1 = pd.read_csv(reg_file_1)
        df_1['data_hora'] = pd.to_datetime(df_1['data_hora'], dayfirst=True)
        df_1.set_index('data_hora')
        print('read first')
        df_2 = pd.read_csv(reg_file_2)
        df_2['data_hora'] = pd.to_datetime(df_2['data_hora'], dayfirst=True)
        df_2.set_index('data_hora')
        print('read second')
        tmp_1 = df_1['pressao'].values
        tmp_2 = df_2['pressao'].values
        tmp_1 = tmp_1 - np.average(tmp_1)
        tmp_2 = tmp_2 - np.average(tmp_2)
        tmp = np.c_[tmp_1, tmp_2]
        tmp = tmp/tmp.std(axis=0)
        
        ica = FastICA(n_components=2)
        signals = ica.fit_transform(tmp)
        pca = PCA(n_components=2)
        signals_pca = pca.fit_transform(tmp)
        
        ax_1 = plt.subplot(311)
        ax_2 = plt.subplot(312)
        ax_3 = plt.subplot(313)
        ax_1.plot(tmp.T[0])
        ax_1.plot(tmp.T[1])
        ax_2.plot(signals.T[0])
        # ax_2.plot(signals_pca.T[0])
        ax_3.plot(signals.T[1])
        # ax_3.plot(signals_pca.T[1])
        plt.show()


def main():
    reg_proc = RegProcessor('7-UB-66D-RNS-RP-4#26043-IIF.TPR.csv', '7-UB-66D-RNS-RP-4#30682-ISP.TPR.csv')

if __name__ == '__main__':
    main()