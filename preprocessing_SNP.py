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

    def init(self, batch_size = 0, _input_window_l = 0, _horizon = 0):

        self.Input_window_length = _input_window_l
        self.Forecast_horizon = _horizon

        # all data is the co collection of all time series and tables
        self.raw_data = None
        self.batchsize = batch_size

        # all data with new devised features
        self.with_feats = None

        # Selected features and predicted column
        self.X = []
        self.Y = []

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



    # some financial feature engineering from
    # Deep Learning Stock Volatility with Google Domestic Trends
    # Ruoxuan Xiong1, Eric P. Nichols2 and Yuan Shen

    def u(self):
        X = self.X
        return math.log(X['high'] / X['open'])

    def d(self):
        X =  self.X
        return math.log(X['low'] / X['open'])


    # loading data
    def load_SNP(self, market):
        self.Market = market
        if market == "All":
            self.raw_data = pd.read_csv("sandp500/all_stocks_5yr.csv")
        else:
            self.raw_data = pd.read_csv("sandp500/individual_stocks_5yr/" + str(market) + "_data.csv")

    def load_SNP_processed(self, market):
        self.Market = market
        self.with_feats = pd.read_csv('./SnP_processed/SnP_market_' + str(self.Market) + "_with_features.csv")



    def make_feats(self, save_file):
        data = self.raw_data
        # Add t as the time variable
        data['t'] = range(len(data))

        date = data['date']
        data = data.drop('date',  axis=1)
        data['year'] = [d.split('-')[0] for d in date]
        data['month'] = [d.split('-')[1] for d in date]
        data['day_of_month'] = [d.split('-')[2] for d in date]
        data['day_of_week'] = data['t'] % 7

        data['weekend'] = [float(int(x)) for x in np.array(data['day_of_week'] == 1).__or__(np.array(data['day_of_week'] == 2))]
        cl = data['close']
        # returns is at t is the log difference close prices of t to t-1
        data['return'] = [float(1) for row in range(len(data))]
        data['normal_close'] = cl

        for ind in range(len(data['normal_close'])):
            data['normal_close'][ind] = data['normal_close'][ind] / data['open'][0]

        for t in range(1, len(data)):
            data['return'][t] = ((cl[t] / cl[t - 1])-1) * 100

        self.with_feats = data
        if save_file:
            self.with_feats.to_csv('./SnP_processed/SnP_market_' + str(self.Market) + "_with_features.csv")


    def select_feats(self, _feats, _target):
        self.Target = _target
        if _feats == None:
            self.X = self.with_feats
        else :
            self.X = self.with_feats[_feats]

        if _target == None:
            self.Y = self.with_feats
        else:
            self.Y = self.with_feats[_target]


    # slides over t in X and generates input output time series
    # Slided windows are subsets of Original self.X Data Frame ==>  they preserve structure as:
    # first dim: features(with names of columns), second is time index(index in original data frame)
    # This way until you cast it to an array you will always have the real time index
    def make_sliding_wind_samples(self, _in_column, _slide, _sets = 'all'):

        sets = dict()
        if _sets == 'all':
            sets['tr'] = [self.X_tr, self.Y_tr, self.X_w_tr, self.Y_w_tr]
            sets['dev'] = [self.X_dev, self.Y_dev, self.X_w_dev, self.Y_w_dev]
            sets['te'] = [self.X_te, self.Y_te, self.X_w_te, self.Y_w_te]


        t_0 = 0
        w = self.Input_window_length
        h = self.Forecast_horizon

        for set in sets:
            while (t_0 + w + h < len(sets[set][0]['t'])):
                x = np.array(sets[set][0][_in_column][t_0: t_0 + w])
                y = np.array([sets[set][1][t_0 + w + h]])
                t_0 += _slide
                sets[set][2].append(x)
                sets[set][3].append(y)


        if len(self.X_tr) < 10:
            raise ValueError("Too few number of samples(<10) choose a smaller input window, sliding or sight")

        self.maxn_full_batches = int(len(self.X_tr) / self.batchsize)


    def preprocess_markets(self):
        for file_name in os.listdir("sandp500/individual_stocks_5yr")[1:]:
            market = file_name.split('_')[0]
            self.load_SNP(market)
            self.make_feats(True)


    def normalize(self):
        self.X[self.X.columns.drop(['return', 't'])] = preprocessing.normalize(self.X[self.X.columns.drop(['return', 't'])])

    # TODO: Decide(find out) if the output of training set can intersect with inout of validation/test
    # divide the time series or divide samples
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
        combined = list(zip(self.X_tr, self.Y_tr))
        shuffle(combined)
        self.X_tr[:], self.Y_tr[:] = zip(*combined)


    # def get_next_batch(self):
    #
    #     N = self.maxn_full_batches
    #
    #     # if it is the last batch just return remaining samples
    #     if self.Batchno == N:
    #         self.Batchno = 0
    #         return self.X_tr[N-1*self.batchsize:len(self.X_tr)],\
    #                self.Y_tr[N-1*self.batchsize:len(self.X_tr)],\
    #                True
    #     else:
    #         n = self.Batchno
    #         self.Batchno+=1
    #         return self.X_tr[n*self.batchsize:(n+1)*self.batchsize],\
    #                self.Y_tr[n*self.batchsize:(n+1)*self.batchsize],\
    #                False


    def get_baseline_(self, _baseline):

        # Baseline prediction where prediction = last observation(assuming observation
        if _baseline == 'same':
            pred = []
            for i in range(len(self.Y_dev)):
                # Conversion to NP array here no more
                # FINAL DECISION MADE: CONVERT TO MATRIX FROM DATA FRAME IN SLIDING WINDOWS
                x=np.array(self.X_dev[i][self.Input_window_length-1])

                # Hence zero is always target(regressed observation)
                pred.append(x[0])
            return np.sqrt(mean_squared_error(pred, np.array(self.Y_dev)))

        elif _baseline == "linear":
            lm = linear_model.LinearRegression()
            lm.fit(self.flatten_Xs(self.X_tr), self.Y_tr)
            return np.sqrt(mean_squared_error(lm.predict(self.flatten_Xs(self.X_dev)), self.Y_dev))


        elif _baseline == "vol_wei_mean":
            pred = []
            for i in range(len(self.Y_dev)):
                x = self.X_dev[i]
                pred.append(x[len(x) - 1])
        # # volume weighted average of window
        # if baseline == "vol_we_ave":



# print()


# prcs = processor()
# prcs.init(10,10,10)
# prcs.preprocess_markets()
