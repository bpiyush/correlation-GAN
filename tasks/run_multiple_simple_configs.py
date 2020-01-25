import os
import argparse
from termcolor import colored
import subprocess

from utils.startup import get_project_root
from utils.visualize import colored_print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpu', '--cpu_list', nargs='*', type=int, required=True, help='pass cpu indices to work on, example: 70 79')
    parser.add_argument('-c', '--configs', nargs='*', type=str, required=True)
    parser.add_argument('-gpu', '--cuda_device', type=int, required=True)
    args = parser.parse_args()

    colored_print("=> Setting CUDA device: {}".format(args.cuda_device))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda_device)

    cwd = os.getcwd()

    project_root = get_project_root()
    colored_print("=> Changing directory to project home: {} ...".format(project_root))
    os.chdir(str(project_root))

    for config in args.configs:
        colored_print("=============================== Running for config: {} ====================================".format(config), 'white')
        subprocess.call('python main.py -v {}.yml -cpu {} {}'.format(config, args.cpu_list[0], args.cpu_list[1]), shell=True)
        colored_print("=============================== xxxxxxxxxxxxxxxxxxxxxx ====================================".format(config), 'white')

    colored_print("=> Changing directory to original: {} ...".format(cwd))
    os.chdir(str(cwd))



# cuda 1; python main.py -v v1.yml -cpu 20 25