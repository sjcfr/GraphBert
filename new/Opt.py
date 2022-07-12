import argparse

def model_opts(parser):
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input train data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--test_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Path for repository of pretrained Bert model. "
                             "Bert pre-trained model belonged in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_tokenizer", default=None, type=str, required=True,
                        help="Path for Bert tokenizer. Bert tokenizer should belong to the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_seq_length_test",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization for test set. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--pret",
                        default=False,
                        action='store_true',
                        help="Whether to pretreat the data.")
    parser.add_argument("--pretrain",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--graph_type",
                        default='event',
                        type=str,
                        required=False,
                        help="Type of graph. Should be in 'event' or 'knowledge'.")

    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=6.25e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--l2_reg',
                        type=float, default=0.01,
                        help='Margin value used in the MultiMarginLoss.')
    parser.add_argument('--gpuls', type=int, default=0,help='The gpus to use')
    parser.add_argument('--gpuid', type=int, default=0,help='The gpu to use')
    
    parser.add_argument("--start_layer",
                        default=15,
                        type=int,
                        help="The layer starting from which the graph vectors extracts information from BERT.")
                             
    parser.add_argument("--merge_layer",
                        default=18,
                        type=int,
                        help="The layer which auxilary branch merge with the stem.")
                        
    parser.add_argument("--n_layer_extractor",
                        default=2,
                        type=int,
                        help="Number of layers for extracting graph information from BERT.")
                             
    parser.add_argument("--n_layer_aa",
                        default=2,
                        type=int,
                        help="Number of layers for approximating adjacancy matrix using extracted graph vectors.")
                             
    parser.add_argument("--n_layer_gnn",
                        default=2,
                        type=int,
                        help="Number of layers for GNN to cunduct reasoning using extracted graph vectors and (approximated) adjacancy matrix.")

    parser.add_argument("--n_layer_merger",
                        default=2,
                        type=int,
                        help="Number of merger layers for merging extracted graph vectors and hidden states of BERT.")

    parser.add_argument("--method_extractor",
                        default='mean',
                        type=str,
                        help="Type of extracting graph vector from BERT.")
                        
    parser.add_argument("--method_gnn",
                        default='gat',
                        type=str,
                        help="Type of GNN for conducting graph reasoning using extracted graph vectors and (approximated) adjacancy matrix.")
                        
    parser.add_argument("--method_merger",
                        default='gat',
                        type=str,
                        help="Way for merging GNN extracted features into BERT decoder.")
                        
    parser.add_argument("--layer_norm",
                        default=False,
                        action='store_true',
                        help="If conducting layer normalization for the auxiliary graph branch.")

    parser.add_argument("--act_fn_branch",
                        default='gelu',
                        type=str,
                        help="If conducting layer normalization for the auxiliary graph branch.")
    parser.add_argument("--num_frozen_epochs",
                        default=1,
                        type=int,
                        help="The number of epochs in which the parameters of bert model is frozen.")
    parser.add_argument("--use_bert",
                        default=False,
                        action='store_true',
                        help="If use pretrained bert model.")
                        
    parser.add_argument("--loss_aa_distrib",
                        default='cat',
                        choices=['cat', 'normal'],
                        type=str,
                        help="method of calculating loss function.")

    parser.add_argument("--loss_aa_smooth",
                        default=0,
                        type=float,
                        help="A constant for smoothing the aa_loss_function")
    parser.add_argument("--loss_aa_smooth_method",
                        default='all',
                        type=str,
                        help="A constant for smoothing the aa_loss_function")
    parser.add_argument("--Lambda",
                        default=0.01,
                        type=float,
                        help="Lambda for graph loss")
                                                
    parser.add_argument("--trained_model", default=None, type=str, required=False,
                        help="The path of trained model.")
                        
    parser.add_argument("--graph_embedder_voc_path", default=None, type=str, required=False,
                        help="The path of vocabulary graph embedder.")

    parser.add_argument("--graph_embedder_path", default=None, type=str, required=False,
                        help="The path of graph embedder.")
                        
    parser.add_argument("--baseline", default=False, action='store_true',
                        help="Training baseline model or GraphBert")

    parser.add_argument("--path_gb", default=None, type=str, required=False,
                        help="Path of pertrained GraphBert")
    parser.add_argument("--path_bt", default=None, type=str, required=False,
                        help="Path of pertrained BERT")
    parser.add_argument("--path_er", default=None, type=str, required=False,
                        help="Path of pertrained ERNIE")
    parser.add_argument("--path_gt", default=None, type=str, required=False,
                        help="Path of pertrained GraphTransformer")




