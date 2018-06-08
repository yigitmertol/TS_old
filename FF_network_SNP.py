import tensorflow as tf
import numpy as np

class Network:
    I_WINDOW = 10
    INPUTS_DIM = 0
    O_WINDOW = 1

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.INPUTS_DIM = len(args.feats.split('-'))
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.X = tf.placeholder(tf.float32, [None,  self.I_WINDOW, self.INPUTS_DIM], name="inputs")

            # Outputs
            self.Y = tf.placeholder(tf.float32, [None, self.O_WINDOW], name="output")

            # FOR NORMAL FF NETWORK WE JUST FLATTEN ALL INPUTS ALONG THE INPUT DIM
            features = tf.layers.flatten(self.X)
            hidden_layer = tf.layers.dense(features, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer = tf.layers.dense(hidden_layer, 100, activation=tf.nn.relu, name="hidden_layer2")
            hidden_layer = tf.layers.dense(hidden_layer, 50, activation=tf.nn.relu, name="hidden_layer3")
            hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.sigmoid, name="hidden_layer4")
            hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer5")
            # hidden_layer = tf.layers.dense(hidden_layer, 20, activation=tf.nn.relu, name="hidden_layer6")

            output_layer = tf.layers.dense(hidden_layer, self.O_WINDOW, activation=None, name="output_layer")
            self.predictions = output_layer

            # Training
            loss = tf.losses.mean_squared_error(self.Y, output_layer, scope="loss")
            self.loss =loss
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.predictions, self.Y))

            # tf.reduce_mean((tf.square(tf.subtract(y, y_)))

            self.ratio_offset = self.offset = tf.reduce_mean(tf.multiply(tf.div(tf.abs(tf.subtract(self.Y, self.predictions)), self.predictions), 100))

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
        rmse, pred, _, __ = self.session.run([self.rmse, self.predictions, self.training, self.summaries["train"]], {self.X: _inputs, self.Y: _output})
        return rmse, pred
    def evaluate(self, dataset, _inputs, _output):
        rmse, pred, real, _ = self.session.run([self.rmse, self.predictions, self.Y , self.summaries[dataset]], {self.X: _inputs, self.Y: _output})
        return rmse, pred, real, self.session.run(self.summaries[dataset], {self.X: _inputs, self.Y: _output})


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
    parser.add_argument("--horizon", default=1, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window_len", default=Network.I_WINDOW, type=int, help="Window length of input")
    parser.add_argument("--market", default='ABC', type=str, help="Window length of input")
    parser.add_argument("--feats", default='return-normal_close-open-high-low-volume', type=str, help="Window length of input")
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
    import preprocessing_SNP
    prcs = preprocessing_SNP.processor()
    prcs.init(args.batch_size, args.window_len, args.horizon)
    prcs.load_SNP_with_feats(args.market)
    # prcs.make_feats(save_file=True)
    feats = args.feats.split('-')
    prcs.select_feats(['t'] + feats, 'return')
    prcs.make_sliding_wind_samples(feats,  _slide=1)
    prcs.split_Train_dev_test(_slide=1)
    prcs.shuffle_samples()
    print("Same as last baseline rmse: " + str(prcs.get_baseline_("same")))
    print("Linear model baseline rmse: " + str(prcs.get_baseline_("linear")))

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        epoch_finished=False
        prcs.shuffle_samples()
        while not epoch_finished:
            input_, output_, epoch_finished = prcs.get_next_batch()
            network.train(input_, list(output_))
            # print("Prediction " + str(pred) + ", output:" + str(output_))
            # print("Error: ", mse_)
        if i%30==0:
            rmse_tr, pred_, real_, _ = network.evaluate("dev", prcs.X_tr, prcs.Y_tr)
            rmse, pred, real, _ = network.evaluate("dev", prcs.X_dev, prcs.Y_dev)



            # print("for time series : " + str(prcs.X_dev[some_patters][0:4]) + "...")
            ind = np.random.randint(len(prcs.Y_dev)-30)
            print("some real values: " + str([round(y[0], 3) for y in prcs.Y_dev[ind:ind+5]]))
            print("prediction was:   " + str([round(y[0], 3) for y in pred[ind:ind+5]]))


            # print("Val error: " ,round(rmse, 4))
            print("Dev rmse: " + str(round(rmse, 4)))
            # print("Dev rmse: ", round(np.sqrt(mean_squared_error(pred, real)), 4))
            print("Tr rmse: " + str(round(rmse_tr ,4)))
            # print("Tr  rmse: ", round(np.sqrt(mean_squared_error(pred_, real_)), 4))
    summary = network.evaluate("test", prcs.X_te, prcs.Y_te)


