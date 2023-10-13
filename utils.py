import argparse

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--lam', type=int, default=-1)
    parser.add_argument('--thred', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, default='')
    
    args, _ = parser.parse_known_args()
    return args