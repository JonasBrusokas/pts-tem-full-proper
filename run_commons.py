import argparse

class ExperimentConstants:
    fix_seed = 2021

    # Settings prefixes
    SETTINGS_PREFIX = "v3"
    FM_SETTINGS_PREFIX = "v5_fm"
    DETERMINISTIC_SETTINGS_PREFIX = "v2_det"  # bumped 20240910

    # Result object prefixes
    RESULT_OBJECT_VERSION_POSTFIX = (
        "20240919_v1"  # Use this, if we're running non-HeatFlex
    )
    # RESULT_OBJECT_VERSION_POSTFIX = "20240120_heatflex"  # Use this, if we're running HeatFlex with updates!

class TorchDeviceUtils:

    @staticmethod
    def check_if_should_use_gpu(args) -> (bool, bool):
        import torch
        should_use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if (should_use_gpu):
            # If CUDA is not found, check for the MPS backend
            try:
                mps_available = torch.backends.mps.is_available()
                if (mps_available):
                    print("USING MPS !!!")
                    should_use_gpu = mps_available
            except Exception as e:
                print(f"Exception occurred while trying to check for 'mps' availability : {e}")
                mps_available = False
        else:
            mps_available = False
        print(f">> USING GPU: {should_use_gpu}")

        if should_use_gpu and not mps_available:
            cuda_device_count = torch.cuda.device_count()
            print(f">> Number of available CUDA GPUs: {cuda_device_count}")
            if cuda_device_count > 1:
                print(
                    f"***\n***\nWARNING: more than 1 CUDA GPU is available in this scripts scope\n***\n***"
                )
        return (should_use_gpu, mps_available)

class ExperimentUtils:
    # TODO: deprecated, use compat mode on the FM one instead!!!
    @staticmethod
    def get_arg_parser():
        parser = argparse.ArgumentParser(
            description="Autoformer & Transformer family for Time Series Forecasting"
        )
        # basic config
        parser.add_argument(
            "--is_training", type=int, required=True, default=1, help="status"
        )
        parser.add_argument(
            "--model_id", type=str, required=True, default="test", help="model id"
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            default="Autoformer",
            help="model name, options: [Autoformer, Informer, Transformer, ]",
        )

        # logging configuration
        parser.add_argument(
            "--should_log",
            type=bool,
            required=False,
            default=False,
            help="should use ClearML to log",
        )
        parser.add_argument(
            "--name", type=str, required=False, default="NeoEBM experiment"
        )

        # data loader
        parser.add_argument(
            "--data", type=str, required=True, default="ETTm1", help="dataset type"
        )
        parser.add_argument(
            "--site_id",
            type=str,
            required=False,
            default="None",
            help="Site id for HeatFlex data",
        )
        parser.add_argument(
            "--target_site_id",
            type=str,
            required=False,
            default="None",
            help="Analysis targets site id for HeatFlex data",
        )
        parser.add_argument(
            "--root_path",
            type=str,
            default="./data/ETT/",
            help="root path of the data file",
        )
        parser.add_argument(
            "--data_path", type=str, default="ETTh1.csv", help="data file"
        )

        # NOTE: this can do multivariate input -> univariate predict
        parser.add_argument(
            "--features",
            type=str,
            default="M",
            help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
        )
        parser.add_argument(
            "--feature",
            type=str,
            default=None,
            help="Unused arg, guard against typos for --feature. Fails validation if used",
        )
        parser.add_argument(
            "--target", type=str, default="OT", help="target feature in S or MS task"
        )
        parser.add_argument(
            "--freq",
            type=str,
            default="h",
            help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            default="./checkpoints/",
            help="location of model checkpoints",
        )

        # forecasting task
        parser.add_argument(
            "--seq_len", type=int, default=96, help="input sequence length"
        )
        parser.add_argument(
            "--label_len", type=int, default=48, help="start token length"
        )
        parser.add_argument(
            "--pred_len", type=int, default=96, help="prediction sequence length"
        )

        # DLinear
        # >>> only for DLinear - individual channels for each feature
        parser.add_argument(
            "--individual",
            action="store_true",
            default=False,
            help="DLinear: a linear layer for each variate(channel) individually",
        )
        # Formers
        # >>> Embedding types for the transformers
        parser.add_argument(
            "--embed_type",
            type=int,
            default=0,
            help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding",
        )
        # >>> Number of features in X, passed to 'enc_embedding'
        parser.add_argument(
            "--enc_in", type=int, default=7, help="encoder input size"
        )  # DLinear with --individual, use this hyperparameter as the number of channels
        # >>> Number of features in Y, passed to 'dec_embedding'
        parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
        # >>> Number of features in Y
        parser.add_argument("--c_out", type=int, default=7, help="output size")
        # >>> Used a lot to describe the dimensionality of intermediate representations
        # 1. output_dim     = seq_len * d_model in MLP y_encoder
        # 2. input_dim      = seq_len * d_model (* 2) in MLP xy_decoder
        # 3. DecoderLayer(d_model) # Width of AutoCorrelation layers, used in both y_encoders and xy_decoders (
        # Transformer-based)
        # 4. ***Embeddings(d_model) # Defines the output width
        # 5.
        parser.add_argument(
            "--d_model", type=int, default=512, help="dimension of model"
        )
        # >>> TODO:
        parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        # >>> TODO:
        parser.add_argument(
            "--e_layers", type=int, default=2, help="num of encoder layers"
        )
        # >>> TODO:
        parser.add_argument(
            "--d_layers", type=int, default=1, help="num of decoder layers"
        )
        # >>> TODO:
        parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
        # >>> TODO:
        parser.add_argument(
            "--moving_avg", type=int, default=25, help="window size of moving average"
        )
        # >>> TODO:
        parser.add_argument("--factor", type=int, default=1, help="attn factor")
        # >>> TODO:
        parser.add_argument(
            "--distil",
            action="store_false",
            help="whether to use distilling in encoder, using this argument means not using distilling",
            default=True,
        )
        # >>> TODO:
        parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
        # >>> TODO:
        parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        # >>> TODO:
        parser.add_argument("--activation", type=str, default="gelu", help="activation")
        # >>> TODO:
        parser.add_argument(
            "--output_attention",
            action="store_true",
            help="whether to output attention in ecoder",
        )
        # >>> TODO:
        parser.add_argument(
            "--do_predict",
            action="store_true",
            help="whether to predict unseen future data",
        )

        # optimization
        parser.add_argument(
            "--num_workers", type=int, default=10, help="data loader num workers"
        )  # NOTE: we will make this
        # '0' just in case
        # >>> TODO:
        parser.add_argument("--itr", type=int, default=2, help="experiments times")
        # >>> TODO:
        parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
        # >>> TODO:
        parser.add_argument(
            "--batch_size", type=int, default=32, help="batch size of train input data"
        )
        # >>> TODO:
        parser.add_argument(
            "--patience", type=int, default=3, help="early stopping patience"
        )
        # >>> TODO:
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.0001,
            help="optimizer learning rate",
        )
        # >>> TODO:
        parser.add_argument("--des", type=str, default="test", help="exp description")
        # >>> TODO:
        parser.add_argument("--loss", type=str, default="mse", help="loss function")
        # >>> TODO:
        parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )
        # >>> TODO:
        parser.add_argument(
            "--use_amp",
            action="store_true",
            help="use automatic mixed precision training",
            default=False,
        )

        # >>> TODO:
        parser.add_argument(
            "-s",
            "--ebm_samples",
            required=True,
            type=int,
            help="Samples to use for NCE",
        )
        # >>> TODO:
        parser.add_argument(
            "-ep",
            "--ebm_epochs",
            required=True,
            type=int,
            help="Number of epochs used for training the EBM",
        )
        # >>> TODO:
        parser.add_argument(
            "-mn",
            "--ebm_model_name",
            required=True,
            type=str,
            help="Name of the EBM architecture",
        )
        # >>> TODO:
        parser.add_argument(
            "-hs",
            "--ebm_hidden_size",
            required=True,
            type=int,
            help="Hidden units used for the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-nl",
            "--ebm_num_layers",
            required=True,
            type=int,
            help="Number of layers used for the " "predictive model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ndl",
            "--ebm_decoder_num_layers",
            required=True,
            type=int,
            help="Number of decoder layers used for the EBM model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ps",
            "--ebm_predictor_size",
            required=True,
            type=int,
            help="Predictor code size used in the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ds",
            "--ebm_decoder_size",
            required=True,
            type=int,
            help="Decoder code used in the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-lr",
            "--ebm_optim_lr",
            required=False,
            type=float,
            default=1e-3,
            help="Learning rate for EBM optimizer",
        )

        # >>> TODO:
        parser.add_argument(
            "-olr",
            "--ebm_inference_optim_lr",
            required=False,
            type=float,
            default=0.001,
            help="Learning rate for inference optimizer",
        )
        # >>> TODO:
        parser.add_argument(
            "-ost",
            "--ebm_inference_optim_steps",
            required=False,
            type=int,
            default=50,
            help="Step size for inference optimizer",
        )
        # >>> TODO:
        parser.add_argument(
            "-obs",
            "--ebm_inference_batch_size",
            required=False,
            type=int,
            default=32,
            help="Batch size for inference",
        )

        # >>> TODO:
        parser.add_argument(
            "-vdt",
            "--ebm_validate_during_training_step",
            required=False,
            type=int,
            default=10,
            help="Perform validation every n steps",
        )
        # >>> TODO:
        parser.add_argument(
            "-tm",
            "--ebm_training_method",
            required=False,
            type=str,
            default="nce",
            help="Training method",
        )
        # >>> TODO:
        parser.add_argument(
            "-sd",
            "--ebm_seed",
            required=False,
            type=int,
            default=ExperimentConstants.fix_seed,
            help="Seed for experiments",
        )
        # >>> TODO:
        parser.add_argument(
            "-ts",
            "--ebm_training_strategy",
            required=True,
            type=str,
            help="Training strategy to be used to train the EBM",
        )

        # Specific to Contrastive Divergence
        # >>> TODO:
        parser.add_argument(
            "-cds",
            "--ebm_cd_step_size",
            required=True,
            type=float,
            help="Step size for CD",
        )
        # >>> TODO:
        parser.add_argument(
            "-cdn",
            "--ebm_cd_num_steps",
            required=True,
            type=int,
            help="Num steps for CD",
        )
        # >>> TODO:
        parser.add_argument(
            "-cda", "--ebm_cd_alpha", required=True, type=float, help="Alpha for CD"
        )
        # >>> TODO:
        parser.add_argument(
            "-cdsr",
            "--ebm_cd_sched_rate",
            required=True,
            type=float,
            help="CD scheduling rate",
        )

        # GPU
        parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
        parser.add_argument("--gpu", type=int, default=0, help="gpu")
        parser.add_argument(
            "--use_multi_gpu",
            action="store_true",
            help="use multiple gpus",
            default=False,
        )
        parser.add_argument(
            "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
        )
        parser.add_argument(
            "--test_flop",
            action="store_true",
            default=False,
            help="See utils/tools for usage",
        )

        # >>> TODO: FIX
        parser.add_argument(
            "-o",
            "--output_parent_path",
            required=True,
            type=str,
            help="Output parent path for experiment outputs",
        )

        parser.add_argument(
            "--experiment_only_on_given_model_path",
            required=False,
            type=str,
            default="None",
            help="Special mode with no training",
        )
        parser.add_argument(
            "--only_rerun_inference", required=False, type=int, default=0
        )
        parser.add_argument(
            "--only_output_model_params", required=False, type=int, default=0
        )

        parser.add_argument(
            "--force_retrain_orig_model", required=False, type=bool, default=False
        )
        parser.add_argument(
            "--force_retrain_y_enc", required=False, type=bool, default=False
        )
        parser.add_argument(
            "--force_retrain_xy_dec", required=False, type=bool, default=False
        )

        parser.add_argument(
            "--ebm_margin_loss", required=False, type=float, default=-1.0
        )

        ####
        # FEDformer specific stuff

        # >>> TODO:
        parser.add_argument("--version", required=False, type=str, default="Fourier")
        # >>> TODO:
        parser.add_argument(
            "--mode_select",
            type=str,
            default="random",
            help="for FEDformer, there are two mode selection method, options: [random, low]",
        )
        # >>> TODO:
        parser.add_argument(
            "--modes", type=int, default=64, help="modes to be selected random 64"
        )
        # parser.add_argument('--moving_avg', default=[24], help='window size of moving average')

        ####
        # TimesNet specific stuff

        parser.add_argument(
            "--top_k", type=int, default=5, help="TimesNet 'k' parameter for TimesBlock"
        )
        parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
        return parser

    @staticmethod
    # Add this function after the imports and before the main code
    def set_enc_dec_in(args):
        if args.features == "MS":
            if args.data == "custom":
                if args.data_path == "exchange_rate.csv":
                    args.enc_in = args.dec_in = 8
                elif args.data_path == "electricity.csv":
                    args.enc_in = args.dec_in = 321
                elif args.data_path == "traffic.csv":
                    args.enc_in = args.dec_in = 862
                elif args.data_path == "weather.csv":
                    args.enc_in = args.dec_in = 21
                elif args.data_path == "national_illness.csv":
                    args.enc_in = args.dec_in = 7
            elif args.data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
                args.enc_in = args.dec_in = 7
        if args.features == "S":
            args.enc_in = args.dec_in = 1

        print(f"enc_in and dec_in set to: {args.enc_in}")

    @staticmethod
    def get_arg_parse_for_fm(is_compat_old_ebm=False):
        parser = argparse.ArgumentParser(
            description="Autoformer & Transformer family for Time Series Forecasting"
        )
        # basic config
        parser.add_argument(
            "--is_training", type=int, required=True, default=1, help="status"
        )
        parser.add_argument(
            "--model_id", type=str, required=True, default="test", help="model id"
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            default="Autoformer",
            help="model name, options: [Autoformer, Informer, Transformer, ]",
        )

        # logging configuration
        parser.add_argument(
            "--should_log",
            type=bool,
            required=False,
            default=False,
            help="should use ClearML to log",
        )
        parser.add_argument(
            "--name", type=str, required=False, default="NeoEBM experiment"
        )

        # data loader
        parser.add_argument(
            "--data", type=str, required=True, default="ETTm1", help="dataset type"
        )
        parser.add_argument(
            "--site_id",
            type=str,
            required=False,
            default="None",
            help="Site id for HeatFlex data",
        )
        parser.add_argument(
            "--target_site_id",
            type=str,
            required=False,
            default=None,
            help="Analysis targets site id for HeatFlex data",
        )
        parser.add_argument(
            "--root_path",
            type=str,
            default="./data/ETT/",
            help="root path of the data file",
        )
        parser.add_argument(
            "--data_path", type=str, default="ETTh1.csv", help="data file"
        )

        # NOTE: this can do multivariate input -> univariate predict
        parser.add_argument(
            "--features",
            type=str,
            default="M",
            help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
        )
        parser.add_argument(
            "--feature",
            type=str,
            default=None,
            help="Unused arg, guard against typos for --feature. Fails validation if used",
        )
        parser.add_argument(
            "--target", type=str, default="OT", help="target feature in S or MS task"
        )
        parser.add_argument(
            "--freq",
            type=str,
            default="h",
            help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            default="./checkpoints/",
            help="location of model checkpoints",
        )

        # forecasting task
        parser.add_argument(
            "--seq_len", type=int, default=96, help="input sequence length"
        )
        parser.add_argument(
            "--label_len", type=int, default=48, help="start token length"
        )
        parser.add_argument(
            "--pred_len", type=int, default=96, help="prediction sequence length"
        )

        # DLinear
        # >>> only for DLinear - individual channels for each feature
        parser.add_argument(
            "--individual",
            action="store_true",
            default=False,
            help="DLinear: a linear layer for each variate(channel) individually",
        )
        # Formers
        # >>> Embedding types for the transformers
        parser.add_argument(
            "--embed_type",
            type=int,
            default=0,
            help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding",
        )
        # >>> Number of features in X, passed to 'enc_embedding'
        parser.add_argument(
            "--enc_in", type=int, default=7, help="encoder input size"
        )  # DLinear with --individual, use this hyperparameter as the number of channels
        # >>> Number of features in Y, passed to 'dec_embedding'
        parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
        # >>> Number of features in Y
        parser.add_argument("--c_out", type=int, default=7, help="output size")
        # >>> Used a lot to describe the dimensionality of intermediate representations
        # 1. output_dim     = seq_len * d_model in MLP y_encoder
        # 2. input_dim      = seq_len * d_model (* 2) in MLP xy_decoder
        # 3. DecoderLayer(d_model) # Width of AutoCorrelation layers, used in both y_encoders and xy_decoders (
        # Transformer-based)
        # 4. ***Embeddings(d_model) # Defines the output width
        # 5.
        parser.add_argument(
            "--d_model", type=int, default=512, help="dimension of model"
        )
        # >>> TODO:
        parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        # >>> TODO:
        parser.add_argument(
            "--e_layers", type=int, default=2, help="num of encoder layers"
        )
        # >>> TODO:
        parser.add_argument(
            "--d_layers", type=int, default=1, help="num of decoder layers"
        )
        # >>> TODO:
        parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
        # >>> TODO:
        parser.add_argument(
            "--moving_avg", type=int, default=25, help="window size of moving average"
        )
        # >>> TODO:
        parser.add_argument("--factor", type=int, default=1, help="attn factor")
        # >>> TODO:
        parser.add_argument(
            "--distil",
            action="store_false",
            help="whether to use distilling in encoder, using this argument means not using distilling",
            default=True,
        )
        # >>> TODO:
        parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
        # >>> TODO:
        parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        # >>> TODO:
        parser.add_argument("--activation", type=str, default="gelu", help="activation")
        # >>> TODO:
        parser.add_argument(
            "--output_attention",
            action="store_true",
            help="whether to output attention in ecoder",
        )
        # >>> TODO:
        parser.add_argument(
            "--do_predict",
            action="store_true",
            help="whether to predict unseen future data",
        )

        # optimization
        parser.add_argument(
            "--num_workers", type=int, default=10, help="data loader num workers"
        )  # NOTE: we will make this
        # '0' just in case
        # >>> TODO:
        parser.add_argument("--itr", type=int, default=2, help="experiments times")
        # >>> TODO:
        parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
        # >>> TODO:
        parser.add_argument(
            "--batch_size", type=int, default=32, help="batch size of train input data"
        )
        # >>> TODO:
        parser.add_argument(
            "--patience", type=int, default=3, help="early stopping patience"
        )
        # >>> TODO:
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.0001,
            help="optimizer learning rate",
        )
        # >>> TODO:
        parser.add_argument("--des", type=str, default="test", help="exp description")
        # >>> TODO:
        parser.add_argument("--loss", type=str, default="mse", help="loss function")
        # >>> TODO:
        parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )
        # >>> TODO:
        parser.add_argument(
            "--use_amp",
            action="store_true",
            help="use automatic mixed precision training",
            default=False,
        )

        # >>> TODO:
        parser.add_argument(
            "-s",
            "--ebm_samples",
            required=True,
            type=int,
            help="Samples to use for NCE",
        )
        # >>> TODO:
        parser.add_argument(
            "-ep",
            "--ebm_epochs",
            required=True,
            type=int,
            help="Number of epochs used for training the EBM",
        )
        # >>> TODO:
        parser.add_argument(
            "-mn",
            "--ebm_model_name",
            required=True,
            type=str,
            help="Name of the EBM architecture",
        )
        # >>> TODO:
        parser.add_argument(
            "-hs",
            "--ebm_hidden_size",
            required=True,
            type=int,
            help="Hidden units used for the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-nl",
            "--ebm_num_layers",
            required=True,
            type=int,
            help="Number of layers used for the " "predictive model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ndl",
            "--ebm_decoder_num_layers",
            required=True,
            type=int,
            help="Number of decoder layers used for the EBM model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ps",
            "--ebm_predictor_size",
            required=True,
            type=int,
            help="Predictor code size used in the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-ds",
            "--ebm_decoder_size",
            required=True,
            type=int,
            help="Decoder code used in the model",
        )
        # >>> TODO:
        parser.add_argument(
            "-lr",
            "--ebm_optim_lr",
            required=False,
            type=float,
            default=1e-3,
            help="Learning rate for EBM optimizer",
        )

        # >>> TODO:
        parser.add_argument(
            "-olr",
            "--ebm_inference_optim_lr",
            required=False,
            type=float,
            default=0.001,
            help="Learning rate for inference optimizer",
        )
        # >>> TODO:
        parser.add_argument(
            "-ost",
            "--ebm_inference_optim_steps",
            required=False,
            type=int,
            default=50,
            help="Step size for inference optimizer",
        )
        # >>> TODO:
        parser.add_argument(
            "-obs",
            "--ebm_inference_batch_size",
            required=False,
            type=int,
            default=32,
            help="Batch size for inference",
        )

        # >>> TODO:
        parser.add_argument(
            "-vdt",
            "--ebm_validate_during_training_step",
            required=False,
            type=int,
            default=10,
            help="Perform validation every n steps",
        )
        # >>> TODO:
        parser.add_argument(
            "-tm",
            "--ebm_training_method",
            required=False,
            type=str,
            default="nce",
            help="Training method",
        )
        # >>> TODO:
        parser.add_argument(
            "-sd",
            "--ebm_seed",
            required=False,
            type=int,
            default=ExperimentConstants.fix_seed,
            help="Seed for experiments",
        )
        # >>> TODO:
        parser.add_argument(
            "-ts",
            "--ebm_training_strategy",
            required=True,
            type=str,
            help="Training strategy to be used to train the EBM",
        )

        # Specific to Contrastive Divergence
        # >>> TODO:
        parser.add_argument(
            "-cds",
            "--ebm_cd_step_size",
            required=True,
            type=float,
            help="Step size for CD",
        )
        # >>> TODO:
        parser.add_argument(
            "-cdn",
            "--ebm_cd_num_steps",
            required=True,
            type=int,
            help="Num steps for CD",
        )
        # >>> TODO:
        parser.add_argument(
            "-cda", "--ebm_cd_alpha", required=True, type=float, help="Alpha for CD"
        )
        # >>> TODO:
        parser.add_argument(
            "-cdsr",
            "--ebm_cd_sched_rate",
            required=True,
            type=float,
            help="CD scheduling rate",
        )

        # GPU
        parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
        parser.add_argument("--gpu", type=int, default=0, help="gpu")
        parser.add_argument(
            "--use_multi_gpu",
            action="store_true",
            help="use multiple gpus",
            default=False,
        )
        parser.add_argument(
            "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
        )
        parser.add_argument(
            "--test_flop",
            action="store_true",
            default=False,
            help="See utils/tools for usage",
        )

        # >>> TODO: FIX
        parser.add_argument(
            "-o",
            "--output_parent_path",
            required=True,
            type=str,
            help="Output parent path for experiment outputs",
        )

        parser.add_argument(
            "--experiment_only_on_given_model_path",
            required=False,
            type=str,
            default="None",
            help="Special mode with no training",
        )
        parser.add_argument(
            "--only_rerun_inference", required=False, type=int, default=0
        )
        parser.add_argument(
            "--only_output_model_params", required=False, type=int, default=0
        )

        parser.add_argument(
            "--force_retrain_orig_model", required=False, type=bool, default=False
        )
        parser.add_argument(
            "--force_retrain_y_enc", required=False, type=bool, default=False
        )
        parser.add_argument(
            "--force_retrain_xy_dec", required=False, type=bool, default=False
        )

        parser.add_argument(
            "--ebm_margin_loss", required=False, type=float, default=-1.0
        )

        ####
        # FEDformer specific stuff

        # >>> TODO:
        parser.add_argument("--version", required=False, type=str, default="Fourier")
        # >>> TODO:
        parser.add_argument(
            "--mode_select",
            type=str,
            default="random",
            help="for FEDformer, there are two mode selection method, options: [random, low]",
        )
        # >>> TODO:
        parser.add_argument(
            "--modes", type=int, default=64, help="modes to be selected random 64"
        )
        # parser.add_argument('--moving_avg', default=[24], help='window size of moving average')

        ####
        # TimesNet specific stuff

        parser.add_argument(
            "--top_k", type=int, default=5, help="TimesNet 'k' parameter for TimesBlock"
        )
        parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")

        ##########
        # Multi loader
        is_multi_required = not (is_compat_old_ebm)

        parser.add_argument(
            "--multi_data",
            type=str,
            required=is_multi_required,
            help="Comma separated 'data' values",
        )
        parser.add_argument(
            "--multi_data_path",
            type=str,
            required=is_multi_required,
            help="Comma separated 'data_path' values",
        )

        parser.add_argument(
            "--target_multi_data",
            type=str,
            required=is_multi_required,
            help="Comma separated TARGET 'data' values",
        )
        parser.add_argument(
            "--target_multi_data_path",
            type=str,
            required=is_multi_required,
            help="Comma separated TARGET 'data_path' values",
        )

        parser.add_argument(
            "--is_test_mode",
            type=int,
            default=0,
            help="1 if is test mode, 0 if not",
        )

        parser.add_argument(
            "--train_ebm_on_target",
            type=int,
            required=is_multi_required,
            help="1 if should train EBM on target datasets only, 0 if not",
        )

        parser.add_argument(
            "--train_ebm_on_subset_source",
            type=int,
            required=is_multi_required,
            help="1 if should train EBM on SUBSET SOURCE dataset only, 0 if not",
        )

        parser.add_argument(
            "--train_ebm_like_original_tem",
            type=int,
            required=False,
            default=0,
            help="1 if should train EBM on trained like original TEM",
        )
        return parser
