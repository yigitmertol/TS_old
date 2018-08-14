from Feed_forward import FF_SNP
from Processors import preprocessing_SNP
import numpy as np



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
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

    # Architecture of the network
    parser.add_argument("--arch", default="", type=str, help="Size of the hidden layer.")

    # Task parameters(input window length and how far the predictions are)
    parser.add_argument("--horizon", default=1, type=int, help="how far in t after last observation we try to predict")
    parser.add_argument("--window_len", default=FF_SNP.Network.I_WINDOW, type=int, help="Window length of input")

    # parser.add_argument("--markets", default='AAL-', type=str, help="Window length of input")
    parser.add_argument("--target", default='AAL-return', type=str, help="The predicted value")


    parser.add_argument("--inputs", default='target-AAL,4normals', type=str,
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
    # import preprocessing_SNP
    prcs = preprocessing_SNP.processor()
    prcs.init(args.batch_size, args.window_len, args.horizon)
    target = args.target.split('-')
    prcs.load_SNP_processed(target[0])
    prcs.add_Y(target[1])
    trend = True if target[1]=='trend' else False

    inputs = args.inputs.split('-')

    for input in inputs:
        if input == "target":
            prcs.Target_is_feat = True
            prcs.add_X([target[1]])
            continue
        if input.split(',')[0] == "all":
            for market_file_name in os.listdir('./SnP_processed'):
                market = market_file_name.split('_')[2]
                if market == target[1]:
                    continue
                prcs.load_SNP_processed(market)
                feats = input.split(',')[1:]
                prcs.add_X(feats)
        else:
            market = input.split(',')[0]
            prcs.load_SNP_processed(market)
            feats = input.split(',')[1:]
            prcs.add_X(feats)


    # prcs.handle_Inf()
    prcs.split_Train_dev_test(_slide=1)
    # prcs.handle_nans()
    # prcs.normalize()
    prcs.make_windows(_slide=1, _trend=trend)

    # prcs.shuffle_samples()
    print("Variance of Y(dev set): " + str(round(np.var(prcs.Y_dev), 3)))
    print("Mean of of Y(dev set):  " + str(round(np.mean(prcs.Y_dev), 3)))
    print("Same as last baseline rmse: " + str(prcs.get_baseline_("same")))
    print("Linear model baseline rmse: " + str(prcs.get_baseline_("linear")))




    # Construct the network
    network = FF_SNP.Network(threads=args.threads, input_dim=len(prcs.X.columns))
    network.construct(args)


    print("lenght of train: " + str(len(prcs.X_tr)) + " and dev: " +str(len(prcs.X_dev)))
    # Train
    for i in range(args.epochs):
        epoch_finished=False
        prcs.shuffle_samples()

        if i % 2 == 0:
            rmse_tr, pred_ = network.evaluate("train", prcs.X_w_tr, prcs.Y_w_tr)
            rmse, pred= network.evaluate("dev", prcs.X_w_dev, prcs.Y_w_dev)

            ind = np.random.randint(len(prcs.Y_w_dev) - 10)
            print("some real values: " + str([round(y[0], 2) for y in prcs.Y_w_dev[ind:ind + 10]]))
            print("predictions were: " + str([round(y[0], 2) for y in pred[ind:ind + 10]]))

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

    print("\n\nTraining Finished")
    rmse_te, pred_te = network.evaluate("test", prcs.X_w_te, prcs.Y_w_te)
    print("test error: " + str(rmse_te))
    print("some real values: " + str([round(y[0], 2) for y in prcs.Y_w_te[ind:ind + 10]]))
    print("predictions were: " + str([round(y[0], 2) for y in pred_te[ind:ind + 10]]))






