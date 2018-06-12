import tensorflow as tf
import numpy as np

class Network:
    I_WINDOW = 20
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
            self.x = tf.placeholder(tf.float32, [None, self.I_WINDOW, self.INPUTS_DIM], name="inputs")

            # Outputs
            self.y = tf.placeholder(tf.float32, [None, self.O_WINDOW], name="output")

            # FOR NORMAL FF NETWORK WE JUST FLATTEN ALL INPUTS ALONG THE INPUT DIM
            features = tf.layers.flatten(self.x)
            hidden_layer = tf.layers.dense(features, 256, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer2")
            hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu, name="hidden_layer3")
            # hidden_layer = tf.layers.dense(hidden_layer, 64, activation=tf.nn.sigmoid, name="hidden_layer4")
            hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="hidden_layer5")
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer6")
            #
            output_layer = tf.layers.dense(hidden_layer, self.O_WINDOW, activation=None, name="output_layer")
            self.predictions = output_layer

            # Training
            # loss = tf.losses.mean_squared_error(self.y, output_layer, scope="loss")

            loss = tf.losses.absolute_difference(self.y, output_layer, scope='loss')
            self.loss =loss
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

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


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
    parser.add_argument("--horizon", default=10, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=50, type=int, help="Size of the hidden layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window_len", default=Network.I_WINDOW, type=int, help="Window length of input")
    parser.add_argument("--market", default='AAL', type=str, help="Window length of input")
    parser.add_argument("--feats", default='close-open-high-low', type=str, help="Window length of input")
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
    prcs.load_SNP_processed(args.market)
    # prcs.make_feats(save_file=True)
    feats = args.feats.split('-')
    prcs.select_feats(['t'] + feats, 'close')

    prcs.split_Train_dev_test(_slide=1)
    prcs.make_sliding_wind_samples(feats,  _slide=1)

    prcs.shuffle_samples()
    print("Same as last baseline rmse: " + str(prcs.get_baseline_("same")))
    print("Linear model baseline rmse: " + str(prcs.get_baseline_("linear")))




    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)


    print("lenght of train: " + str(len(prcs.X_tr)) + " and dev: " +str(len(prcs.X_dev)))
    # Train
    for i in range(args.epochs):
        epoch_finished=False
        prcs.shuffle_samples()

        if i % 10 == 0:
            rmse_tr, pred_ = network.evaluate("train", prcs.X_tr, prcs.Y_tr)
            rmse, pred= network.evaluate("dev", prcs.X_dev, prcs.Y_dev)

            ind = np.random.randint(len(prcs.X_dev) - 5)
            print("some real values: " + str([round(y[0], 2) for y in prcs.Y_dev[ind:ind + 5]]))
            print("predictions were: " + str([round(y[0], 2) for y in pred[ind:ind + 5]]))

            print("Dev rmse: " + str(round(rmse, 4)))
            print("Tr rmse: " + str(round(rmse_tr, 4)))

        batch_no = 0
        while not epoch_finished:
            input_ = prcs.X_w_tr[batch_no:batch_no+args.batch_size]
            output_ = prcs.Y_w_tr[batch_no:batch_no+args.batch_size]
            rmse_ba , _ = network.train(input_, list(output_))

            batch_no += 1
            if batch_no * args.batch_size > len(prcs.X_tr):
                epoch_finished = True

    summary = network.evaluate("test", prcs.X_te, prcs.Y_te)


