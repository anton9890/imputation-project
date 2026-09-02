from parse_config import ConfigParser
import argparse
from utils import load_data, post_data
from imputation import Imputation
from datetime import datetime
import collections

def main(config):
    query = config["query_proccess"]
    df = load_data(query)

    imputation = Imputation(df, config)
    imputation.run()

    imputed = imputation.data

    n = input("Do you want to post data? : (Y/N)")
    if n in ["Y", "y", "yes", "Yes", "YES"]:
        while True:
            name = input("Please name the tag: ")
            if name == query['tag'] or name == "":
                print("Please give me a different name from the original tag")
            else:
                post_data(query, imputed, name)
                break
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Imputation Project')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--imputer', default="interpolated", type=str,
                      help='choose imputer (default: interpolated)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-s', '--start'], type=str, target='query_proccess;date;start'),
        CustomArgs(['-e', '--end'], type=str, target='query_proccess;date;end'),
        CustomArgs(['-sv', '--save'], type=bool, target='query_proccess;save')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)