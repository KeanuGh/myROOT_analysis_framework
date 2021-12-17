import argparse
import pathlib

from src.dataset import lumi_year

parser = argparse.ArgumentParser(description="Build singular dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# INPUT
parser.add_argument('test', help='test')

parser.add_argument('--datapath', '-d', type=pathlib.Path, required=True,
                    help='Path to ROOT file(s)')
parser.add_argument('--name', '-n', default='data',
                    help='Dataset name')
parser.add_argument('--TTree', '-T', required=True,
                    help='Name of default TTree')
parser.add_argument('--cutfile', '-c', type=pathlib.Path, required=True,
                    help='Path to cutfile')
parser.add_argument('--label', '-l', default='',
                    help='dataset label for plots')
parser.add_argument('--pkl_dir', type=pathlib.Path,
                    help='directory of pickle file')

lumi_year_group = parser.add_mutually_exclusive_group()
lumi_year_group.add_argument('--year', '-y', default='2015+2016', choices=lumi_year.keys(),
                             help='data-taking year')
lumi_year_group.add_argument('--luminosity', '--lumi', type=float,
                             help='data luminosity')

# PATHS
parser.add_argument('--output_root_dir', '-o',
                    help='output directory')
# parser.add_argument('output')

# OPTIONS
parser.add_argument('--reco', action='store_true',
                    help='whether dataset will contain reconstructed data (will try and guess if not given)')
parser.add_argument('--truth', action='store_true',
                    help='whether dataset will contain truth data (will try and guess if not given)')
parser.add_argument('--to_pkl', action='store_false',
                    help='whether to output to a pickle file')
parser.add_argument('--lepton', default='lepton',
                    help='name of lepton in W->lnu if applicable')
parser.add_argument('--chunksize', default=1024, type=int,
                    help='chunksize for uproot ROOT file import')
parser.add_argument('--no_validate_duplicated_events', action='store_false',
                    help='do not check for duplicated events')
parser.add_argument('--no_validate_sum_of_weights', action='store_false',
                    help='do not check if sumofweights is sum of weight_mc for DSIDs')

args = parser.parse_args()
print(args.test)
