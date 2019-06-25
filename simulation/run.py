import json
import argparse
from simulator import run_simu
from fitor import run_fitting
import utils
import os



def main():
    # get param
    parser = argparse.ArgumentParser("PP simulator")
    parser.add_argument('--param-file', type=str, default='parameters', help="")
    parser.add_argument('--result', type=str, default='result', help="")
    args = parser.parse_args()

    # Load params
    #param = json.load(open(args.param_file, "rt"))
    params = utils.load_all([args.param_file])[0]

    os.makedirs(args.result, exist_ok=True)

    # Step 1 : simulation
    tauList = run_simu(args, params, 2)

    # Step 2 : fitting
    run_fitting(args, params, tauList)

main()
