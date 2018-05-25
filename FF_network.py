import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing

class Network:
    I_WINDOW = 10
    O_WINDOW = 1

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.input_series = tf.placeholder(tf.float32, [None, self.I_WINDOW], name="inputs")
            self.next_val = tf.placeholder(tf.float32, [None, self.O_WINDOW], name="output")

            features = self.input_series
            hidden_layer = tf.layers.dense(features, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer2")
            hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer3")
            hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer4")
            # hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer5")
            # hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer6")

            output_layer = tf.layers.dense(hidden_layer, self.O_WINDOW, activation=None, name="output_layer")
            self.predictions = output_layer

            # Training
            loss = tf.losses.mean_squared_error(self.next_val, output_layer, scope="loss")
            self.loss =loss
            global_step = tf.train.create_global_step()
            self.training = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step=global_step, name="training")

            self.rmse = tf.sqrt(loss)

            self.ratio_offset = self.offset = tf.reduce_mean(tf.multiply(tf.div(tf.abs(tf.subtract(self.next_val, self.predictions)), self.predictions), 100))

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
        rmse, pred, _, __ = self.session.run([self.rmse, self.predictions, self.training, self.summaries["train"]], {self.input_series: _inputs, self.next_val: _output})
        return rmse, pred
    def evaluate(self, dataset, _inputs, _output):
        rmse, pred, real, _ = self.session.run([self.rmse, self.predictions, self.next_val , self.summaries[dataset]], {self.input_series: _inputs, self.next_val: _output})
        return rmse, pred, real, self.session.run(self.summaries[dataset], {self.input_series: _inputs, self.next_val: _output})


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--sight", default=5, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--epochs", default=10000, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=5, type=int, help="Size of the hidden layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window_len", default=Network.I_WINDOW, type=int, help="Window length of input")
    parser.add_argument("--market", default='ABBV', type=str, help="Window length of input")
    # parser.add_argument("--a", default=10, type=int, help="Window length of input")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(map(lambda arg:"{}={}".format(*arg), sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # # Load the data
    # from tensorflow.examples.tutorials import mnist
    # mnist = mnist.input_data.read_data_sets("mnist_data/", reshape=False, seed=42)
    import preprocessing
    prcs = preprocessing.processor()
    prcs.init(args.batch_size)
    prcs.load_SNP(args.market)
    prcs.make_feats()
    prcs.select_feats(['t', 'normal_close'], 'normal_close')
    prcs.sliding_window_samples('normal_close', _window_len=args.window_len, _slide=1, _sight=args.sight)
    prcs.divide_data( _sight=args.sight, _slide=1)
    prcs.shuffle_samples()
    print("Same as last baseline rmse: " +str(prcs.get_baseline_("same")))
    print("Linear model baseline rmse: " + str(prcs.get_baseline_("linear")))

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)



    # Train
    for i in range(args.epochs):
        epoch_finished=False
        while not epoch_finished:
            prcs.shuffle_samples()
            input_, output_, epoch_finished = prcs.get_next_batch()

            if(len(input_)==0):
                continue
            rmse_, pred = network.train(input_, output_)
            # print("Prediction " + str(pred) + ", output:" + str(output_))
            # print("Error: ", mse_)
        rmse, pred, real,  _ = network.evaluate("dev", prcs.X_val, prcs.Y_val)
        if i%100==0:
            # print("Validation set outputs and predictions: ")
            # print(real, pred)
            print("Val error: " ,round(rmse, 4))
            print("Training error: ", round(rmse_, 4))
    summary = network.evaluate("test", prcs.X_te, prcs.Y_te)


