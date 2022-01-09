import argparse

from src.datasetbuilder import lumi_year, DatasetBuilder
from src.logger import get_logger


def main():
    # PARSER
    # ============================
    parser = argparse.ArgumentParser(description="Build singular dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input
    inputs = parser.add_argument_group(title='Inputs', description='Dataset inputs')
    inputs.add_argument('--datapath', '-d', type=str, required=True,
                        help='Path to ROOT file(s)')
    inputs.add_argument('--name', '-n', default='data',
                        help='Dataset name')
    inputs.add_argument('--TTree', '-T', required=True,
                        help='Name of default TTree')
    inputs.add_argument('--cutfile', '-c', type=str, required=True,
                        help='Path to cutfile')
    inputs.add_argument('--label', '-l', default='',
                        help='dataset label for plots')
    lumi_year_group = inputs.add_mutually_exclusive_group()
    lumi_year_group.add_argument('--year', '-y', default='2015+2016', choices=lumi_year.keys(),
                                 help='data-taking year')
    lumi_year_group.add_argument('--luminosity', '--lumi', type=float,
                                 help='data luminosity')

    # Output
    outputs = parser.add_argument_group(title='Output',
                                        description="Output paths")
    outputs.add_argument('--pkl_file', '-o', type=str, required=False,
                         help="Pickled DataFrame output file. DEFAULT: './<name>.pkl'")
    outputs.add_argument('--log_file', type=str, required=False,
                         help="Log output file. DEFAULT: './<name>.log'")
    outputs.add_argument('--latex_cutflow', '-f', type=str,
                         help='File to output a latex cutflow to')

    # Options
    # logging options
    logging_args = parser.add_argument_group(title='Logging options', description='Options for logger')
    logging_args.add_argument('--log_level', type=str.lower, default='debug', choices=['debug', 'info', 'warning'],
                              help='Logging level. Default: DEBUG')
    logging_args.add_argument('--log_out', type=str.lower, default='console', choices=['console', 'file', 'both'],
                              help="Whether to print logs to 'console', 'file', or 'both'")
    logging_args.add_argument('--timedatelog', action='store_true',
                              help='Whether to append datetime to log filename')
    logging_args.add_argument('--log_mode', type=str, choices=['w', 'w+', 'a', 'a+'], default='w',
                              help='mode for openning log file')

    # data options
    data_opt = parser.add_argument_group(title='Dataset options', description='Options for dataset building')
    data_opt.add_argument('--chunksize', default=1024, type=int,
                          help='chunksize for uproot ROOT file import')
    data_opt.add_argument('--no_validate_missing_events', action='store_false',
                          help='do not check for missing events from truth tree')
    data_opt.add_argument('--no_validate_duplicated_events', action='store_false',
                          help='do not check for duplicated events')
    data_opt.add_argument('--no_validate_sumofweights', action='store_false',
                          help='do not check if sumofweights is sum of weight_mc for DSIDs')

    # RUN
    # ========================
    args = parser.parse_args()

    # get logger
    if args.log_out in ('file', 'both'):
        args.log_file = args.name + '.log'
    log_levels = {'debug': 10, 'info': 20, 'warning': 30}
    logger = get_logger(
        name=args.name,
        log_level=log_levels[args.log_level],
        log_out=args.log_out,
        timedatelog=args.timedatelog,
        log_file=args.log_file,
        mode=args.log_mode,
    )

    builder = DatasetBuilder(
        name=args.name,
        TTree_name=args.TTree,
        logger=logger,
        chunksize=args.chunksize,
        validate_sumofweights=not args.no_validate_sumofweights,
        validate_missing_events=not args.no_validate_missing_events,
        validate_duplicated_events=not args.no_validate_duplicated_events
    )
    dataset = builder.build(data_path=args.datapath, cutfile_path=args.cutfile)

    dataset.save_pkl_file(args.pkl_file if args.pkl_file else None)

    if args.latex_cutflow:
        dataset.print_latex_table(args.latex_cutflow)


if __name__ == "__main__":
    main()
