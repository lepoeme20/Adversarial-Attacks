"""Set default hyper-parameter
"""
import argparse
import multiprocessing
import torch

def str2bool(s):
    """ convert str to boolean
    Arguments:
        s {str} -- "true" or "1"

    Returns:
        boolean -- if input is "ture" or "1", return True
    """
    return s.lower() in ("true", "1")

def str2float(s):
    if '/' in s:
        s1, s2 = s.split('/')
        s = float(s1)/float(s2)
    return float(s)

def default_parser_setting(parser):
    """Set default arguments
    """
    default_arg = parser.add_argument_group('Default')
    default_arg.add_argument(
        '--attack-name', type=str, default='FGSM', choices=['FGSM', 'DeepFool', 'CW', 'PGD']
        )
    default_arg.add_argument(
        '--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
        help="Dataset"
    )
    default_arg.add_argument(
        '--image-size', type=int, default=64, help="image size"
    )
    default_arg.add_argument(
        '--batch-size', type=int, default=512, help="mini-batch size for classification"
    )
    default_arg.add_argument(
        '--n-cpu', type=int, default=multiprocessing.cpu_count()-1,
        help="# of workers for dataloader"
    )
    default_arg.add_argument(
        "--classifier", type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'],
        help='pre-trained target classifier'
    )
    default_arg.add_argument(
        "--pretrained-dir", type=str, default="./resnet/pretrained_models/",
        help="path of pretrained model"
    )
    default_arg.add_argument(
        "--device-ids", type=int, nargs='*', help="number of total classes"
    )

    pgd_args = parser.add_argument_group('PGD')
    pgd_args.add_argument(
        '--pgd-iters', type=int, default=40, help="# of iteration for PGD attack"
    )
    pgd_args.add_argument(
        '--pgd-eps', type=float, default=0.3, help="For bound eta"
    )
    pgd_args.add_argument(
        '--pgd-alpha', type=str2float, default=2/255, help="Magnitude of perturbation"
    )
    pgd_args.add_argument(
        '--pgd-random-start', type=str2bool, default=False,
        help="If ture, initialize perturbation using eps"
    )

    cw_args = parser.add_argument_group("CW")
    cw_args.add_argument(
        '--cw-c', type=str2float, default=1e-4, help="loss scaler"
    )
    cw_args.add_argument(
        '--cw-kappa', type=float, default=0, help="minimum value on clamping"
    )
    cw_args.add_argument(
        '--cw-iters', type=int, default=10000, help="# of iteration for CW grdient descent"
    )
    cw_args.add_argument(
        '--cw-lr', type=float, default=0.01, help="learning rate for CW attack"
    )
    cw_args.add_argument(
        '--cw-binary-search-steps', type=int, default=9, help="# of iteration for CW optimization"
    )
    cw_args.add_argument(
        '--cw-targeted', type=str2bool, default=False, help="d"
    )

    fgsm_arg = parser.add_argument_group('FGSM')
    fgsm_arg.add_argument(
        '--fgsm-eps', type=float, default=0.003, help="Magnitude of perturbation"
    )

    deepfool_args = parser.add_argument_group('DeepFool')
    deepfool_args.add_argument(
        '--deepFool-iters', type=int, default=5, help="# of iteration for DeepFool attack"
    )

    visualization_args = parser.add_argument_group('Visualization')
    visualization_args.add_argument(
        '--vis-normal', type=str2bool, default=False,
        help="Set true if you want to save normal images"
    )
    visualization_args.add_argument(
        '--vis-n-rows', type=int, default=8,
        help="# of image rows at each saved figure"
        "default value is same as default value of nrow in torchvision.utils.save_image"
    )
    visualization_args.add_argument(
        '--vis-batch-size', type=int, default=1, help=""
    )
    visualization_args.add_argument(
        '--vis-set-idx', type=str2bool, default=False,
        help="Set True if you want to saving specific indices"
    )
    visualization_args.add_argument(
        '--vis-indices', type=int, nargs='*',
        help="Set indices for saving images if you set vis_set_idx True"
    )
    return parser


def get_config():
    """
    Returns:
        parser with set hyper-parameters
    """
    parser = argparse.ArgumentParser()
    default_parser = default_parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return args
