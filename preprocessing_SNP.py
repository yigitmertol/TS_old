import pandas as pd
import math
import numpy as np
from random import shuffle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class processor:


    _batchno = 0
    market = ''

    def init(self, batch_size):


        # all data is the co collection of all time series and tables
        self.raw_data = None
        self.batchsize = batch_size

        # all data with new devised features
        self.with_feats = None

        # Selected features and predicted column
        self.input_ts = []
        self.Y = []

        # Training samples in format like:
        # X = [x(t) for t in INPUT TIME INTERVAL]
        # Y = [y(t) for t in OUTPUT INTERVAL]
        # prepared for export
        self.X_tr= []
        self.Y_tr = []

        # similarly validation and test samples
        self.X_val = []
        self.Y_val = []
        self.X_te = []
        self.Y_te = []


    # some financial feature engineering from
    # Deep Learning Stock Volatility with Google Domestic Trends
    # Ruoxuan Xiong1, Eric P. Nichols2 and Yuan Shen

    def u(self):
        X = self.input_ts
        return math.log(X['high'] / X['open'])

    def d(self):
        X =  self.input_ts
        return math.log(X['low'] / X['open'])


    # loading data
    def load_SNP(self, market):
        self.market = market
        if market == "All":
            self.raw_data = pd.read_csv("sandp500/all_stocks_5yr.csv")
        else:
            self.raw_data = pd.read_csv("sandp500/individual_stocks_5yr/" + str(market) + "_data.csv")




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

        data['weekend'] = np.array(data['day_of_week'] == 1).__or__(np.array(data['day_of_week'] == 2))

        # returns is at t is the log difference close prices of t to t-1
        data['return'] = [float(1) for row in range(len(data))]
        data['normal_close'] = [float(1) for row in range(len(data))]
        cl = data['close']
        for t in range(1, len(data)):
            data['return'][t] = ((cl[t] / cl[t - 1])-1) * 100
            data['normal_close'][t] = 100 * float(cl[t]) / float(data['open'][0])


            # print("unique")
        self.with_feats = data
        if save_file:
            self.with_feats.to_csv('SnP_market_'+str(self.market)+"_with_features.csv")

    def load_SNP_with_feats(self, market):
        self.market = market
        self.with_feats = pd.read_csv('SnP_market_'+str(self.market)+"_with_features.csv")

    def select_feats(self, feats, output):
        if feats == None:
            self.input_ts = self.with_feats
        else :
            self.input_ts = self.with_feats[feats]

        if output == None:
            self.Y = self.with_feats
        else:
            self.Y = self.with_feats[output]


    # slides over t in X and generates input output time series
    def sliding_window_samples(self, _in_column, _window_len, _slide, _sight):
        series_X = self.input_ts[_in_column]
        series_Y = self.Y

        # t_0 here is beginning_of_in_wind
        t_0=0
        while(t_0 + _window_len + _sight < len(self.input_ts['t'])):
            x = series_X[t_0 : t_0 + _window_len]
            y = [float(series_Y[t_0 + _window_len + _sight])]
            t_0 += _slide
            self.X_tr.append(x)
            self.Y_tr.append(y)

        if len(self.X_tr) < 10:
            raise ValueError("Too few number of samples(<10) choose a smaller input window, sliding or sight")

        self.maxn_full_batches = int(len(self.X_tr) / self.batchsize)


    # TODO: Decide(find out) if the output of training set can intersect with inout of validation/test


    # divide the time series or divide samples
    def divide_data(self, _sight, _slide, _divide = 'samples' , _perc_of_test = 10, _perc_of_val = 10):
        ratio_te = _perc_of_test/100
        ratio_val = _perc_of_val/100

        if _divide == "samples":
            n = len(self.X_tr)


            self.X_te = self.X_tr[int((1 - ratio_te) * n):n - 1]
            self.Y_te = self.Y_tr[int((1 - ratio_te) * n):n - 1]

            self.X_val = self.X_tr[int((1 - ratio_te - ratio_val) * n):int(n * (1 - ratio_te)) ]
            self.Y_val = self.Y_tr[int((1 - ratio_te - ratio_val) * n):int(n * (1 - ratio_te)) ]

            # we delete some few number of patterns from training set that are have sight in test set

            k = int(_sight / _slide) + 1
            del self.X_tr[int(n * (1 - ratio_te - ratio_val))- k :int((1 - ratio_te - ratio_val) * n)]
            del self.Y_tr[int(n * (1 - ratio_te - ratio_val))- k :int((1 - ratio_te - ratio_val) * n)]

            self.maxn_full_batches = int(len(self.X_tr) / self.batchsize)


    def shuffle_samples(self):
        combined = list(zip(self.X_tr, self.Y_tr))
        shuffle(combined)
        self.X_tr[:], self.Y_tr[:] = zip(*combined)


    def get_next_batch(self):
        N = self.maxn_full_batches

        # if it is the last batch just return remaining samples
        if self._batchno == N:
            self._batchno = 0
            return self.X_tr[N-1*self.batchsize:len(self.X_tr)-1],\
                   self.Y_tr[N-1*self.batchsize:len(self.X_tr)-1],\
                   True
        else:
            n = self._batchno
            self._batchno+=1
            return self.X_tr[n*self.batchsize:(n+1)*self.batchsize],\
                   self.Y_tr[n*self.batchsize:(n+1)*self.batchsize],\
                   False


    def get_baseline_(self, baseline):

        # Baseline prediction where prediction = last observation(assuming observation
        if baseline == 'same':
            pred = []
            for i in range(len(self.Y_val)):
                x = self.X_val[i]
                pred.append(x[len(x)-1])
            return np.sqrt(mean_squared_error(pred, self.Y_val))

        elif baseline == "linear":
            lm = linear_model.LinearRegression()
            lm.fit(self.X_tr, self.Y_tr)
            return np.sqrt(mean_squared_error(lm.predict(self.X_val), self.Y_val))

        elif baseline == "vol_wei_mean":
            pred = []
            for i in range(len(self.Y_val)):
                x = self.X_val[i]
                pred.append(x[len(x) - 1])
        # # volume weighted average of window
        # if baseline == "vol_we_ave":



# print()


prcs = processor()
prcs.init(10)
prcs.load_SNP_with_feats('AAL')
# prcs.make_feats(save_file=True)
prcs.select_feats(['t', 'return','month', 'day_of_week'], 'return')
prcs.sliding_window_samples(_in_column=['return', 'month'],_window_len=10, _slide=1, _sight=1)
prcs.divide_data( _sight=1, _slide=1)