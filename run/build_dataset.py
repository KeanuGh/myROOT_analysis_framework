import argparse

parser = argparse.ArgumentParser()

# INPUT
parser.add_argument('--datapath', '-d', help='Path to ROOT file(s)',
                    optional=False)
parser.add_argument('--name', '-n', help='Dataset name',
                    optional=True, default='data')
parser.add_argument('--TTree', '-T', help='Name of default TTree',
                    optional=False)
parser.add_argument('--cutfile', '-c', help='Path to cutfile',
                    optional=False)
parser.add_argument('--year', '-y', help='data-taking year',
                    default='2015+2016', optional=True)
parser.add_argument('--label', '-l', help='dataset label for plots',
                    optional=True, default='')
parser.add_argument('--pkl_dir', help='directory of pickle file',
                    optional=True)

# PATHS
parser.add_argument('output_root_dir', '-o', help='output directory',
                    optional=False,)
# parser.add_argument('output')

# OPTIONS
parser.add_argument('--reco', help='whether dataset will contain reconstructed data (will try and guess if not given)',
                    optional=True, action='store_true')
parser.add_argument('--truth', help='whether dataset will contain truth data (will try and guess if not given)',
                    optional=True, action='store_true')
parser.add_argument('--to_pkl', help='whether to output to a pickle file',
                    ptional=True)
parser.add_argument('--lepton', help='name of lepton in W->lnu if applicable',
                    optional=True, default='lepton')
parser.add_argument('--chunksize', help='chunksize for uproot ROOT file import',
                    optional=True, default=1024, type=int)
parser.add_argument('--no_validate_missing_events', help='do not check for missing events',
                    optional=True, action='store_false')
parser.add_argument('--no_validate_duplicated_events', help='do not check for duplicated events',
                    optional=True, action='store_false')
parser.add_argument('--no_validate_sum_of_weights', help='do not check if sumofweights is sum of weight_mc for DSIDs',
                    optional=True, action='store_false')

args = parser.parse_args()
