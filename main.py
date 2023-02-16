import argparse
import os

from utils.utils import read_config
from workspaces.cdp import Workspace as cdp
from workspaces.oracle_cdp import Workspace as oracle_cdp
from workspaces.sac import Workspace as sac
from workspaces.edl import Workspace as edl

def main(args):

    cfg = read_config('config/'+args.config)

   
    if args.dir!=None:
        cfg['Logger']['log_dir'] = str(args.dir)
    if args.file_name!=None:
        cfg['Logger']['log_dir'] =cfg['Logger']['log_dir'] + str(args.file_name) + '/'
        cfg['Logger']['file_name'] = str(args.file_name)
    if args.seed !=None:
        cfg['Training']['seed'] = args.seed
        cfg['Logger']['log_dir'] = cfg['Logger']['log_dir'] + str(args.seed)+ '/'
    if args.beta !=None:
        cfg['Training']['Focus']['rts'] = args.beta
        cfg['Logger']['log_dir'] = cfg['Logger']['log_dir']+ str(args.beta)+ '/'
        cfg['Logger']['file_name'] = cfg['Logger']['file_name']+ '-'+str(args.beta)
    if args.task !=None:
        cfg['Environment']['config']['Task'] = args.task
        cfg['Logger']['log_dir'] = cfg['Logger']['log_dir']+ str(args.task) + '/'


    if args.method =='cdp':
        workspace = cdp(cfg)
    if args.method =='oracle_cdp':
        workspace = oracle_cdp(cfg)
    if args.method =='sac':
        workspace = sac(cfg)
    if args.method =='edl':
        workspace = edl(cfg)
    if args.train:
        workspace.run()
    if args.eval:
        workspace.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-method',default=None)
    parser.add_argument('-config', default=None)
    parser.add_argument('-dir',default=None)
    parser.add_argument('-file_name',default=None)
    parser.add_argument('-seed',default=None, type=int)
    parser.add_argument('-task',default=None, type=str)
    parser.add_argument('-beta',default=None, type=float)
    parser.add_argument('-train',default=False,type=bool)
    parser.add_argument('-eval',default=False,type=bool)
    args = parser.parse_args()

    if not(args.config):
        print("***config not provided***")

    main(args)