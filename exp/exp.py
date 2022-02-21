import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from model.param import Param
from model.squeeze import Squeeze
from utils.attribute_combination import AttributeCombination as AC
from utils.compare import compare

warnings.filterwarnings("ignore")


class Exp:
    def __init__(self, input_path, output_path, columns, num_workers=10, derived=False):
        self.input_path = input_path
        self.output_path = output_path
        self.derived = derived
        self.columns = columns
        self.num_workers = num_workers

    def executor(self, timestamp):
        if self.derived:
            dfa = pd.read_csv(self.input_path + timestamp + '.a.csv')
            dfb = pd.read_csv(self.input_path + timestamp + '.b.csv')

            df = dfa.copy(deep=True)
            df['real'] = dfa['real'] / dfb['real']
            df['predict'] = dfa['predict'] / dfb['predict']
        else:
            df = pd.read_csv(self.input_path + timestamp + '.csv')

        model = Squeeze(data=df, param=Param())

        model.run()

        try:
            root_cause = AC.batch_to_string(
                frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))
        except IndexError:
            root_cause = ""

        return [timestamp, root_cause]

    def location(self):
        result = pd.DataFrame()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        filenames = sorted(os.listdir(self.input_path))[:-1]
        timestamps = sorted(list(set([filename.split('.')[0] for filename in filenames])))

        # 并行化
        root_cause = Parallel(n_jobs=self.num_workers, backend="multiprocessing")(
            delayed(self.executor)(timestamp)
            for timestamp in tqdm(timestamps))

        root_cause = np.array(root_cause).T

        result['timestamp'] = root_cause[0]
        result['root_cause'] = root_cause[1]
        result.sort_values("timestamp", inplace=True)

        result.to_csv(self.output_path + self.input_path.split('/')[-2] + '_output.csv', index=False)



    def evaluation(self):
        label = pd.read_csv(self.input_path + 'injection_info.csv')
        predict = pd.read_csv(self.output_path + self.input_path.split('/')[-2] + '_output.csv')

        df = pd.merge(label, predict, on='timestamp')
        df.rename(columns={'set': 'label', 'root_cause': 'predict'}, inplace=True)
        df.sort_values("timestamp", inplace=True)

        df['predict'].fillna("", inplace=True)
        df['FN'] = df.apply(lambda x: compare(x['label'], x['predict'], self.columns)[0], axis=1)
        df['TP'] = df.apply(lambda x: compare(x['label'], x['predict'], self.columns)[1], axis=1)
        df['FP'] = df.apply(lambda x: compare(x['label'], x['predict'], self.columns)[2], axis=1)

        df[['timestamp', 'label', 'predict', 'FN', 'TP', 'FP']].to_csv(self.input_path.split('/')[-2] + '_result.csv',
                                                                       index=False)

        f1 = 2 * np.sum(df['TP']) / (2 * np.sum(df['TP']) + np.sum(df['FP']) + np.sum(df['FN']))
        precision = np.sum(df['TP']) / (np.sum(df['TP']) + np.sum(df['FP']))
        recall = np.sum(df['TP']) / (np.sum(df['TP']) + np.sum(df['FN']))

        print("f1: %.4f" % f1)
        print("precision: %.4f" % precision)
        print("recall: %.4f" % recall)
