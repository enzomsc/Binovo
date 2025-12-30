from collections import OrderedDict
import argparse
import numpy as np
import torch.optim as optim
import model_res

n_gpu=[0]
beta=0.5

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default="train", 
                        help="""set differnt type of model:
                                train:      train new model;
                                test:       test model accuracy(percision and recall on aa and peptide levels);
                        """)
    parser.add_argument("--spect_file",type=str,default="",help='the mass spectrum file path')
    parser.add_argument("--train_file", type=str, default="",help="the feature file path for training")
    parser.add_argument("--val_file", type=str, default="",help="the target feature file path for validation")
    parser.add_argument("--test_file", type=str, default="",help="the target feature file path for testing")
    parser.add_argument("--model_path", type=str, default="", help="the path of trained model")
    parser.add_argument("--eval_step", type=int, default=200, help="")
    parser.add_argument("--max_epochs", type=int, default=300,
                    help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                    help="minibatch size")
    parser.add_argument("--warmup", type=int, default=10000, help="scheduler warmup.")
    parser.add_argument("--scheduler_maxiters", type=int, default=100000, help="scheduler max itertions.")
    parser.add_argument('--optim', type=str, default='adam',help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam|adamw')
    parser.add_argument('--learning_rate', type=float, default=2e-4,help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,help='weight_decay')
    parser.add_argument('--checkpoint_path', type=str, default='',help='directory to store checkpointed models')

    parser.add_argument("--beam_size", type=int, default=5,
                    help="the beam size used in beam search")
    parser.add_argument('--max_length', type=int, default=30,help='Maximum length during sampling')
    
    parser.add_argument("--out", type=str, default="", help="generate sequence file")
    args = parser.parse_args()

    return args

def printParams(opt:object)->None:
    print("****************************parameters****************************")
    print(f"model_path:{opt.model_path}")
    print(f"max_epochs:{opt.max_epochs}")
    print(f"batch_size:{opt.batch_size}")
    print(f"learning_rate:{opt.learning_rate}")
    print(f"weight_decay:{opt.weight_decay}")
    print(f"scheduler warmup:{opt.warmup}")
    print(f"schedulermaxiters:{opt.scheduler_maxiters}")
    print(f"checkpoint_path:{opt.checkpoint_path}")
    print("******************************************************************")


def calMass(seq, charge=None):
        if isinstance(seq, str):
            # seq = re.split(r"(?<=.)(?=[A-Z])", seq)
            seq = list(seq)
        calc_mass = sum([model_res.mass_AA[aa] for aa in seq]) + 2*1.007825035+15.99491463
        if charge is not None:
            calc_mass = (calc_mass / charge) + 1.00727646688

        return calc_mass

def build_optimizer(params, opt):
    if opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamw':
        return optim.AdamW(params, opt.learning_rate, weight_decay=opt.weight_decay)

    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
 