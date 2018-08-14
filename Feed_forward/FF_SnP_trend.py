import tensorflow as tf
import numpy as np

class Network:
    I_WINDOW = 7
    INPUTS_DIM = 0

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
            self.trend = tf.placeholder(tf.int32, [None, 1], name="output")

            # FOR NORMAL FF NETWORK WE JUST FLATTEN ALL INPUTS ALONG THE INPUT DIM
            features = tf.layers.flatten(self.x)
            hidden_layer = features
            hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer = tf.layers.dense(hidden_layer, 128, activation=tf.nn.relu, name="hidden_layer2")
            hidden_layer = tf.layers.dense(hidden_layer, 64, activation=tf.nn.relu, name="hidden_layer3")
            # hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="hidden_layer5")
            # hidden_layer = tf.layers.dense(hidden_layer, 256, activation=tf.nn.relu, name="hidden_layer6")
            #
            output_layer = tf.layers.dense(hidden_layer, 1,activation=None, name="output_layer")
            self.predictions = tf.cast(tf.round(output_layer), dtype=tf.int32 )

            # Training
            # loss = tf.losses.mean_squared_error(self.y, output_layer, scope="loss")

            loss = tf.losses.sparse_softmax_cross_entropy(self.trend,output_layer, scope='loss')
            self.loss =loss
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.trend, self.predictions), tf.float32))
            # confusion_matrix = tf.reshape(tf.confusion_matrix(self.trend, self.predictions,
            #                                                   weights=tf.not_equal(self.trend, self.predictions), dtype=tf.float32),
            #                               [1, self.LABELS, self.LABELS, 1])

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/total_error", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, _inputs, _output):
        acc, pred, _, __ = self.session.run([self.accuracy, self.predictions, self.training, self.summaries["train"]], {self.x: _inputs, self.trend: _output})
        return acc, pred
    def evaluate(self, dataset, _inputs, _output):
        acc, pred,  _ = self.session.run([self.accuracy, self.predictions,  self.summaries[dataset]], {self.x: _inputs, self.trend: _output})
        return acc, pred


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Training Parameters Epoch Batch Thread
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    # Architecture of the network
    parser.add_argument("--arch", default="", type=str, help="Size of the hidden layer.")

    # Task parameters(input window length and how far the predictions are)
    parser.add_argument("--horizon", default=1, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--window_len", default=Network.I_WINDOW, type=int, help="Window length of input")

    # parser.add_argument("--markets", default='AAL-', type=str, help="Window length of input")
    parser.add_argument("--target", default='ABT-return', type=str, help="The predicted value")


    parser.add_argument("--inputs", default='target-all,normal_close,normal_close', type=str,
                        help="features that will be loaded fromeach market data set")

    # First feature is at the same time the regressed one

    # parser.add_argument("--a", default=10, type=int, help="Window length of input")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(map(lambda arg:"{}={}".format(*arg), sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself


    # Create the processor object and get the data
    import preprocessing_SNP
    prcs = preprocessing_SNP.processor()
    prcs.init(args.batch_size, args.window_len, args.horizon)
    target = args.target.split('-')
    prcs.load_SNP_processed(target[0])
    prcs.select_n_add_to_Y(target[1])

    inputs = args.inputs.split('-')

    for input in inputs:
        if input == "target":
            prcs.Target_is_feat = True
            prcs.select_n_add_to_X([target[1]])
            continue
        if input.split(',')[0] == "all":
            for market_file_name in os.listdir('./SnP_processed'):
                market = market_file_name.split('_')[2]
                if market == target[1]:
                    continue
                prcs.load_SNP_processed(market)
                feats = input.split(',')[1:]
                prcs.select_n_add_to_X(feats)
        else:
            market = input.split(',')[0]
            prcs.load_SNP_processed(market)
            feats = input.split(',')[1:]
            prcs.select_n_add_to_X(feats)


    prcs.handle_Inf()
    prcs.split_Train_dev_test(_slide=1)
    # prcs.normalize()
    prcs.make_sliding_wind_samples(_slide=1, _trend=True)

    # prcs.shuffle_samples()
    print("Variance of Y(dev set): " + str(round(np.var(prcs.Y_dev), 3)))
    # print("Same as last baseline rmse: " + str(prcs.get_baseline_("same")))
    # print("Linear model baseline rmse: " + str(prcs.get_baseline_("linear")))




    # Construct the network
    network = Network(threads=args.threads, input_dim=len(prcs.X.columns))
    network.construct(args)


    print("lenght of train: " + str(len(prcs.X_tr)) + " and dev: " +str(len(prcs.X_dev)))
    # Train
    for i in range(args.epochs):
        epoch_finished=False
        prcs.shuffle_samples()

        if i % 2 == 0:
            acc_tr, pred_ = network.evaluate("train", prcs.X_w_tr, prcs.Y_w_tr)
            acc, pred= network.evaluate("dev", prcs.X_w_dev, prcs.Y_w_dev)

            ind = np.random.randint(len(prcs.Y_w_dev) - 10)
            print("some real values: " + str([round(y[0], 2) for y in prcs.Y_w_dev[ind:ind + 10]]))
            print("predictions were: " + str([round(y[0], 2) for y in pred[ind:ind + 10]]))

            print("Dev rmse: " + str(round(acc*100, 4)))
            print("Tr rmse: " + str(round(acc_tr*100, 4)))

        batch_no = 0
        while not epoch_finished:
            input_ = prcs.X_w_tr[batch_no:batch_no+args.batch_size]
            output_ = prcs.Y_w_tr[batch_no:batch_no+args.batch_size]
            # output_ = 1 if output_ > input
            acc_ba , _ = network.train(input_, list(output_))

            batch_no += 1
            if batch_no * args.batch_size > len(prcs.X_tr):
                epoch_finished = True

    summary = network.evaluate("test", prcs.X_te, prcs.Y_te)


