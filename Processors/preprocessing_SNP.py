import pandas as pd
import math
import numpy as np
from random import shuffle
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import os

class processor:


    Batchno = 0
    Market = ''
    Target = ''
    Input_window_length = 0
    Forecast_horizon = 0
    Target_is_feat = False

    Ts = dict()

    def init(self, batch_size = 0, _input_window_l = 0, _horizon = 0):

        self.Input_window_length = _input_window_l
        self.Forecast_horizon = _horizon

        # all data is the co collection of all time series and tables
        self.Data_loaded = None
        self.batchsize = batch_size

        # all data with new devised features
        self.with_feats = None

        # Selected features and predicted column
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

        # Training samples in format like:
        # X = X[][x(t) for t in INPUT TIME INTERVAL]
        # Y = [y(t) for t in OUTPUT INTERVAL]

        # prepared for export
        self.X_tr= []
        self.X_w_tr = []
        self.Y_tr = []
        self.Y_w_tr = []

        # similarly develop and test samples
        self.X_dev = []
        self.X_w_dev = []
        self.Y_dev = []
        self.Y_w_dev = []

        self.X_te = []
        self.X_w_te = []
        self.Y_te = []
        self.Y_w_te = []

    def u(self):
        X = self.Data_loaded
        return np.log(np.array(X['high']) / np.array(X['open']))
    def d(self):
        X =  self.Data_loaded
        return np.log(np.array(X['low']) / np.array(X['open']))
    def c(self):
        X = self.Data_loaded
        return np.log(np.array(X['close']) / np.array(X['open']))

    # loading data
    def load_SNP(self, market):
        self.Market = market
        if market == "All":
            self.Data_loaded = pd.read_csv("../Data/sandp500/all_stocks_5yr.csv")
        else:
            self.Data_loaded = pd.read_csv("../Data/sandp500/individual_stocks_5yr/" + str(market) + "_data.csv")

    def load_SNP_processed(self, market):
        self.Market = market
        self.Data_loaded = pd.read_csv('../Data/SnP_processed/SnP_market_' + str(self.Market) + "_with_features.csv")

    def handle_nans(self):
        rows, cols = self.Data_loaded.shape
        for j in range(cols):
            i=0
            while i in range(rows):
                if pd.isna(self.Data_loaded.iloc[i,j]):
                    first_na = i
                    while(pd.isna(self.Data_loaded.iloc[i+1,j])):
                        i+=1
                    last_na = i

                    if first_na == 0:
                        self.Data_loaded.iloc[first_na:last_na+1,j] = self.Data_loaded.iloc[last_na+1,j]
                    elif last_na == rows:
                        self.Data_loaded.iloc[first_na:, j] = self.Data_loaded.iloc[first_na-1, j]
                    else:
                        mean = (float(self.Data_loaded.iloc[last_na+1, j]) +
                                float(self.Data_loaded.iloc[first_na-1, j]))/2
                        self.Data_loaded.iloc[first_na:last_na + 1, j] = mean
                i+=1


        # This function used once to create files in SnP processed no need until fresh clone


    def make_feats(self, save_file):
        data = self.Data_loaded

        # Add t as the time variable
        data['t'] = range(len(data))

        date = data['date']
        data = data.drop('date',  axis=1)
        data['year'] = [d.split('-')[0] for d in date]
        data['month'] = [d.split('-')[1] for d in date]
        data['day_of_month'] = [d.split('-')[2] for d in date]
        data['day_of_week'] = data['t'] % 7

        # boolean is it a weekend
        data['weekend'] = [int(x) for x in np.array(data['day_of_week'] == 1).__or__(np.array(data['day_of_week'] == 2))]

        # returns is at t is the log difference close prices of t to t-1
        data['return'] = np.zeros_like(len(data['t']))
        data['return'][0] = 0


        normal = data['open'][0]


        data['normal_close'] = list(np.array(data['close']) / normal)
        data['normal_open'] = list(np.array(data['open']) / normal)
        data['normal_high'] = list(np.array(data['high']) / normal)
        data['normal_low'] =  list(np.array(data['low']) / normal)
        data['return'][1:] = list(np.array(data['close'][1:] - np.array(data['close'][:-1])))

        # some financial feature engineering from
        # Deep Learning Stock Volatility with Google Domestic Trends
        # Ruoxuan Xiong1, Eric P. Nichols2 and Yuan Shen
        # the above paper references to an older paper on financial
        # volatiliy where this formula is devised
        u = np.array(self.u())
        d = np.array(self.d())
        c = np.array(self.c())
        data['volatility'] = 0.511*np.square(u-d) + 0.019*(c*( u+d)-2*u*d) - 0.383*np.square(c)

        self.with_feats = data
        if save_file:
            self.with_feats.to_csv('../SnP_processed/SnP_market_' + str(self.Market) + "_with_features.csv")

    def process_all_snp(self):
        for file in os.listdir('../sandp500/individual_stocks_5yr'):
            if file in [".DS_Store"]:
                continue
            market = file.split('_')[0]
            self.load_SNP(market)
            for col in self.Data_loaded.columns:
                if any(pd.isna(self.Data_loaded[col])):
                    self.handle_nans()

            self.make_feats(True)

    def add_Y(self, _target = None):
        if _target == None:
            self.Target = _target
            self.Y = None
            print("Target is selected None ==> Y is empty(None)")
        else:
            self.Target = _target
            self.Y = self.Data_loaded[_target]
            print("Regressed(predicted/targeted) variable(s) is " + str(_target) + " of "+ self.Market)

    def add_X(self, _feats = None):

        data = self.Data_loaded
        for feat in _feats:
            if feat == None:
                for col in data.columns:
                    self.X[str(self.Market) + "_" + str(col)] = data[col]
            elif feat == "4normals":
                for col in ['normal_high', 'normal_low', 'normal_open', 'normal_close']:
                    self.X[str(self.Market) + "_" + str(col)] = data[col]
            else:
                self.X[str(self.Market) + "_" + str(feat)] = data[feat]

    def gen_seq(self, _seq, _len):
        self.X = pd.DataFrame({ "x" :  list(np.arange(0, _len)) })

        if _seq == "sin":
            self.Y = pd.DataFrame({ "x" : list(np.sin(np.array(self.X)*np.pi/30)) })

    def make_windows(self, _slide =1, _trend = False, _sets ='all'):

        # slides over t in X and generates input output time series
        # Slided windows are subsets of Original self.X Data Frame ==>  they preserve structure as:
        # first dim: features(with names of columns), second is time index(index in original data frame)
        # This way until you cast it to an array you will always have the real time index

        sets = dict()
        if _sets == 'all':
            for set in ['train', 'dev', 'test']:
                sets[set] = dict()

            sets['train'] = [self.X_tr, self.Y_tr, self.X_w_tr, self.Y_w_tr]
            sets['dev'] = [self.X_dev, self.Y_dev, self.X_w_dev, self.Y_w_dev]
            sets['test'] = [self.X_te, self.Y_te, self.X_w_te, self.Y_w_te]

        w = self.Input_window_length
        h = self.Forecast_horizon

        for set in sets:
            T_0 = sets[set][0].index[0]
            t_0 = 0
            T_1 = sets[set][0].index[-1]
            ts = []
            while (T_0 + w + h <= T_1):
                x = np.array(sets[set][0][:][t_0:t_0 + w])
                y = np.array([sets[set][1][T_0 + w + h-1]])
                ts.append(T_0 + w + h-1)

                trend = 1 if x[-1][0] < y[0] else 0
                sets[set][2].append(x)
                if _trend:
                    y = trend

                sets[set][3].append(y)
                t_0 += _slide
                T_0 += _slide
            self.Ts[str(set)] = ts
        if len(self.X_tr) < 10:
            raise ValueError("Too few number of samples(<10) choose a smaller input window, sliding or sight")

        self.maxn_full_batches = int(len(self.X_tr) / self.batchsize)

        self.X_w_tr_original = list(self.X_w_tr)
        self.Y_w_tr_original = list(self.Y_w_tr)

    def normalize(self):

        df = self.X_tr
        mean = df.mean()
        std = df.std()
        self.X_tr = (df - mean) / std

        df = self.X_dev
        self.X_dev = (df - mean) / std

        df = self.X_te
        self.X_te = (df - mean) / std

    def make_dataset(self, inputs, target):

        for input in inputs:
            if input == "target":
                self.Target_is_feat = True
                self.add_X([target[1]])
                continue
            if input.split(',')[0] == "all":
                for market_file_name in os.listdir('../Data/SnP_processed'):
                    market = market_file_name.split('_')[2]
                    if market == target[1]:
                        continue
                    self.load_SNP_processed(market)
                    feats = input.split(',')[1:]
                    self.add_X(feats)
            else:
                market = input.split(',')[0]
                self.load_SNP_processed(market)
                feats = input.split(',')[1:]
                self.add_X(feats)

    def split_Train_dev_test(self, _slide, _divide ='temporal', _perc_of_test = 20, _perc_of_dev = 20):

        h = self.Forecast_horizon
        n = len(self.X)
        ratio_te = _perc_of_test/100
        te_ind_begin = int((1-ratio_te) * n)
        ratio_dev = _perc_of_dev / 100
        dev_ind_begin =int((1 - ratio_te - ratio_dev) * n)

        # DECIDED !!NOT!! TO USE WINDOWS SPLITTING
        if _divide == "windows":
            n = len(self.X_tr)
            self.X_te = self.X_tr[int((1 - ratio_te) * n):]
            self.Y_te = self.Y_tr[int((1 - ratio_te) * n):]

            self.X_dev = self.X_tr[int((1 - ratio_te - ratio_dev) * n):int(n * (1 - ratio_te))]
            self.Y_dev = self.Y_tr[int((1 - ratio_te - ratio_dev) * n):int(n * (1 - ratio_te))]

            # we delete some few number of patterns from training set that have horizon index in dev sets input
            k = int(h / _slide) + 1
            self.X_tr = self.X_tr[:int((1 - ratio_te - ratio_dev) * n) - k]
            self.Y_tr = self.Y_tr[:int((1 - ratio_te - ratio_dev) * n) - k]

            self.maxn_full_batches = int(len(self.X_tr) / self.batchsize)
        elif _divide =="temporal":

            self.X_te = self.X[te_ind_begin:]
            self.Y_te = self.Y[te_ind_begin:]

            self.X_dev = self.X[dev_ind_begin:te_ind_begin]
            self.Y_dev = self.Y[dev_ind_begin:te_ind_begin]

            self.X_tr = self.X[:dev_ind_begin]
            self.Y_tr = self.Y[:dev_ind_begin]

    def flatten_Xs(self, data):
        flats = []
        for x in data:
            flat = []
            for observations_at_t in x:
                flat.extend(list(observations_at_t))
            flats.append(flat)
        return flats

    def shuffle_samples(self):
        combined = list(zip(self.X_w_tr, self.Y_w_tr))
        shuffle(combined)
        self.X_w_tr[:], self.Y_w_tr[:] = zip(*combined)

    # TODO Check baseline functions are indeed correctly functioning
    def get_baseline_(self, _baseline):

        # Baseline prediction where prediction = last observation(assuming observation
        if _baseline == 'same':

            if not self.Target_is_feat:
                print("The target feat is not contained in regressor(predictors)")
                return
            pred = []
            for i in self.Ts['test']:
                pred.append(self.Y[i-self.Forecast_horizon])

            return np.sqrt(mean_squared_error(pred, np.array(self.Y_w_te))), pred

        elif _baseline == "linear":
            lm = linear_model.LinearRegression()
            lm.fit(self.flatten_Xs(self.X_w_tr), self.Y_w_tr)
            print("Parameters of linear model:\n" + str(lm.coef_))
            pred = lm.predict(self.flatten_Xs(self.X_w_te))
            return np.sqrt(mean_squared_error(pred, self.Y_w_te)), pred


        # elif _baseline == "vol_wei_mean":
        #     pred = []
        #     for i in range(len(self.Y_te)):
        #         x = self.X_te[i]
        #         pred.append(x[len(x) - 1])
        # # volume weighted average of window
        # if baseline == "vol_we_ave":

