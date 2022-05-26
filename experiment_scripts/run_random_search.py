import os
import json
import argparse
import sys

# Add submodule to path
root_path = os.path.dirname(os.path.dirname(os.getcwd()))

path = os.path.join(os.path.dirname(os.getcwd()), 'nc_nas_benchmarks')
path_nasbench = os.path.join(os.path.dirname(os.getcwd()), 'nc_nasbench')
path_nascar = os.path.join(root_path)

sys.path.append(path)
sys.path.append(path_nasbench)
sys.path.append(path_nascar)

from nascar.api.nasbench101_carbon import NASBench101Carbon
from nascar.nas.search_space import Carbon
from nasbench.lib import graph_util

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')

args = parser.parse_args()

if args.benchmark == "carbon":
    nasbench_carbon = NASBench101Carbon(dataset_file='/home/student/pedram-local/nascar/nascar/utils/data/tabular_benchmarks/carbon_4V9E.tfrecord')
    b = Carbon(nasbench_carbon)

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=False)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)

output_path = os.path.join(args.output_path, "random_search")
os.makedirs(os.path.join(path, 'experiments', output_path), exist_ok=True)

cs = b.get_configuration_space()

#runtime = []
#regret = []
#curr_incumbent = None
#curr_inc_value = None

#rt = 0
#X = []
for i in range(args.n_iters):
    config = cs.sample_configuration()
    b.objective_function(config)

if args.benchmark == "nas_cifar10a" or args.benchmark == "nas_cifar10b" or args.benchmark == "nas_cifar10c":
    res = b.get_results(ignore_invalid_configs=True)
else:
    res = b.get_results()

fh = open(os.path.join(path, 'experiments', output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
