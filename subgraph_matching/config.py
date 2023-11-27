import argparse
from common import utils

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)
    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--n_edge_type', type=int,
                        help='number of edge type')
    enc_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    enc_parser.add_argument('--pool', type=str,
                        help='pool or seed')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--numberOfNeighK', type=int,
                        help='Number of neighbour'),
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    enc_parser.add_argument('--data_identifier', type=str,
                        help='file identifier')
    enc_parser.add_argument('--skip', type=str,
                        help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--feature_size', type=int,
                        help='Feature size +1')
    enc_parser.add_argument('--feature_type', type=str,
                        help='embeding or type')
    enc_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--tenTimes', action="store_true",
                        help='whether to use each query one times in training')
    enc_parser.add_argument('--saveBatch', action="store_true",
                        help='save batch data')
    enc_parser.add_argument('--loadBatch', action="store_true",
                        help='load batch data')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--check_data', action="store_true")
    enc_parser.add_argument('--loss_type', type=str)
    enc_parser.add_argument('--test_type', type=str)
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    enc_parser.set_defaults(n_edge_type=5,
                        conv_type='SAGE_typed',
                        method_type='order',
                        n_layers=4,
                        pool='mean',
                        batch_size=1024,
                        hidden_dim=256,
                        feature_size=37,
                        feature_type='type_basic_Proc',
                        skip="learnable",
                        dropout=0.0,
                        n_batches=1000000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                            
                        opt_decay_rate=0.9,
                        opt_decay_step=100,
                        weight_decay=1e-5,
                            
                        opt_restart=100,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=10,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=1024,
                        node_anchored=True,
                        tenTimes=False,
                        numberOfNeighK=3)

    # return enc_parser.parse_args(arg_str)

