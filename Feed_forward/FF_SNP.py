import tensorflow as tf
import numpy as np
import pandas as pd
from Processors import preprocessing_SNP
import matplotlib.pyplot as plt

class Network:
    I_WINDOW = 30
    O_WINDOW = 1

    def __init__(self, threads, input_dim, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.INPUTS_DIM = input_dim
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.x = tf.placeholder(tf.float32, [None, self.I_WINDOW, self.INPUTS_DIM], name="inputs")

            # Outputs
            self.y = tf.placeholder(tf.float32, [None, self.O_WINDOW], name="output")

            # FOR NORMAL FF NETWORK WE JUST FLATTEN ALL INPUTS ALONG THE INPUT DIM
            features = tf.layers.flatten(self.x)
            hidden_layer = features
            # hidden_layer = tf.layers.dense(hidden_layer, 64, activation=tf.nn.relu , name="hidden_layer")
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="hidden_layer2")
            # hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu, name="hidden_layer3")
            # hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu, name="hidden_layer4")
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="hidden_layer5")
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu)
            hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu, name="hidden_layer7")
            hidden_layer = tf.layers.dense(hidden_layer, 512, activation=tf.nn.relu, name="hidden_layer8")
            hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer9")
            hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer10")
            hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer11")
            hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu)
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer11")
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer12")
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer13")
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer14")
            # hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu, name="hidden_layer15")
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="hidden_layer16")
            hidden_layer = tf.layers.dropout(hidden_layer, rate=0.8)
            # add_layer = tf.layers.dense(features, 16, activation=None)
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer6")
            output_layer = tf.layers.dense(hidden_layer, self.O_WINDOW, activation=None, name="output_layer")
            self.predictions = output_layer

            # Training
            # loss = tf.losses.mean_squared_error(self.y, output_layer, scope="loss")
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=0.005, scope=None
            )
            weights = tf.trainable_variables()  # all vars of your graph
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

            loss = tf.losses.mean_squared_error(self.y, output_layer, scope='loss') + regularization_penalty
            self.loss =loss
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # self.training = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss=loss)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.predictions, self.y))

            # tf.reduce_mean((tf.square(tf.subtract(y, y_)))

            self.ratio_offset = self.offset = tf.reduce_mean(tf.multiply(tf.div(tf.abs(tf.subtract(self.y, self.predictions)), self.predictions), 100))

            # # Summaries
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.next_val, self.predictions), tf.float32))
            # confusion_matrix = tf.reshape(tf.confusion_matrix(self.next_val, self.predictions,
            #                                                   weights=tf.not_equal(self.trend, self.predictions), dtype=tf.float32),
            #                               [1, self.LABELS, self.LABELS, 1])

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/total_error", self.rmse)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, _inputs, _output):
        rmse, pred, _, __ = self.session.run([self.rmse, self.predictions, self.training, self.summaries["train"]], {self.x: _inputs, self.y: _output})
        return rmse, pred

    def evaluate(self, dataset, _inputs, _output):
        rmse, pred,  _ = self.session.run([self.rmse, self.predictions,  self.summaries[dataset]], {self.x: _inputs, self.y: _output})
        return rmse, pred


    def make_plot(self, _data_set, _plot, _till, _base=[],):

        if _data_set == "train":
            data = [prcs.X_w_tr_original, prcs.Y_w_tr_original]
        elif _data_set == "test":
            data = [prcs.X_w_te, prcs.Y_w_te]
        elif _data_set == "dev":
            data = [prcs.X_w_dev, prcs.Y_w_dev]


        rmse_te, pred = self.evaluate(_data_set, data[0],data[1])
        print("\n\nError in set "+_data_set+": " + str(rmse_te))
        print("\nRandom Sequence: " + str([round(y[0], 2) for y in data[1][ind:ind + 10]]))
        print("Predictions for it: " + str([round(y[0], 2) for y in pred[ind:ind + 10]]))
        # print("correlation of predictions to real values:" + str(np.corrcoef(pred_te, list(x[0] for x in prcs.Y_w_te) )) )

        fig, ax1 = plt.subplots()
        plots = _plot.split('-')
        x_s = prcs.Ts[_data_set]
        x_s = x_s[0:int(len(x_s)*_till)]

        for plot in plots:
            kind, col, val = plot.split(',')
            if val == 'real':
                y_s = prcs.Y[x_s]
            elif val == 'pred':
                y_s = np.array([p[0] for p in pred][0:int(len(pred)*_till)])
            if kind == 'bar':
                plt.bar(x_s, y_s, alpha=0.5)
            elif kind == 'line':
                ax1.plot(x_s, y_s, str(col)+"o-", linewidth=0.1, markersize=0.7)

        if len(_base)>0:
            y_s = _base
            ax1.plot(x_s, y_s, "bo--", linewidth=0.1, markersize=0.7)

        # # Plot bars
        # k = 1000
        # ax1.plot(x_s[0:int(len(x_s)*_till)], pred[0:2*k+50],'go--', linewidth=0.1, markersize=1)
        # ax1.set_xlabel('$Time$')
        #
        # ax1.set_ylabel('real', color='r')
        #
        # ax1.plot(x_s[0:2*k+50], prcs.Y[x_s][0:2*k+50],'ro--', linewidth=0.1, markersize=1)
        # ax1.set_ylabel('actual', color='r')
        # [tl.set_color('r') for tl in ax1.get_yticklabels()]
        #

        plt.show()


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Training Parameters Epoch Batch Thread
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    # Architecture of the network
    parser.add_argument("--arch", default="", type=str, help="Size of the hidden layer.")

    # Task parameters(input window length and how far the predictions are)
    parser.add_argument("--horizon", default=1, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--window_len", default=Network.I_WINDOW, type=int, help="Window length of input")

    # parser.add_argument("--markets", default='AAL-', type=str, help="Window length of input")
    parser.add_argument("--target", default='ABC-close', type=str, help="The predicted value")


    parser.add_argument("--inputs", default='target-ABC,return,volatility,close,volume', type=str,
                        help="features that will be loaded fromeach market data set")


    # First feature is at always also the regressed one

    # parser.add_argument("--a", default=10, type=int, help="Window length of input")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(map(lambda arg:"{}={}".format(*arg), sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself


###
    prcs = preprocessing_SNP.processor()
    prcs.init(args.batch_size, args.window_len, args.horizon)
    target = args.target.split('-')
    # prcs.load_SNP_processed(target[0])
    # prcs.add_Y(target[1])
    trend = True if target[1]=='trend' else False
    # inputs = args.inputs.split('-')
    # prcs.make_dataset(inputs, target)
    # prcs.handle_Inf()

    prcs.gen_seq("sin", 1200)
    prcs.split_Train_dev_test(_slide=1)
    prcs.normalize()
    prcs.make_windows(_slide=1, _trend=trend)
###

    dev_y_ts = prcs.Ts['dev']
    test_y_ts = prcs.Ts['test']
    tr_y_ts = prcs.Ts['train']

    prcs.shuffle_samples()
    print("\nVariance of Y(dev set): " + str(round(np.var(prcs.Y_dev), 3)))
    sal_rmse, sal_pred = prcs.get_baseline_("same")
    print("\nSame as last baseline rmse: " + str(sal_rmse))
    lm_rmse, lm_pred = prcs.get_baseline_("linear")
    print("\nLinear model baseline rmse: " + str(lm_rmse))




    # Construct the network
    network = Network(threads=args.threads, input_dim=len(prcs.X.columns))
    network.construct(args)


    print("\nLenght of train set: " + str(len(prcs.X_tr)) + " and dev set: " +str(len(prcs.X_dev)))
    # Train
    for i in range(args.epochs):
        epoch_finished=False
        prcs.shuffle_samples()

        if True:
            rmse_tr, pred_ = network.evaluate("train", prcs.X_w_tr, prcs.Y_w_tr)
            rmse, pred= network.evaluate("dev", prcs.X_w_dev, prcs.Y_w_dev)

            ind = np.random.randint(len(prcs.Y_w_dev) - 10)
            print("some real values: " + str([round(y[0], 2) for y in prcs.Y_w_dev[ind:ind + 10]]))
            print("predictions were: " + str([round(y[0], 2) for y in pred[ind:ind + 10]]))

            print("Dev rmse: " + str(round(rmse, 4)))
            print("Tr rmse: " + str(round(rmse_tr, 4)))

        batch_no = 0
        while not epoch_finished:

            x_batch = prcs.X_w_tr[batch_no*args.batch_size:(batch_no+1)*args.batch_size]
            y_batch = prcs.Y_w_tr[batch_no*args.batch_size:(batch_no+1)*args.batch_size]

            # for x in x_batch:
            #     print(x)
            if len(x_batch)==0:
                break
            rmse_ba , _ = network.train(x_batch, list(y_batch))

            batch_no += 1
            if batch_no * args.batch_size >= len(prcs.X_tr):
                epoch_finished = True

    network.make_plot("train", "line,r,real-line,g,pred" ,1)
    network.make_plot("test", "line,r,real-line,g,pred", 1, lm_pred)
    # network.make_plot("train", "bar,r,real-bar,g,pred", 1)
    # network.make_plot("dev", "bar,r,real-bar,g,pred", 1, lm_pred)