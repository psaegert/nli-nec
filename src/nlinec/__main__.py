import argparse

from .data.load import download_data
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser()

    # Add a subcommand for downloading data
    subparsers = parser.add_subparsers(dest='command')

    # Add a subcommand for downloading data
    subparsers.add_parser('download-data')

    # Add a subcommand for training
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-m', '--model-name', type=str, required=True)
    train_parser.add_argument('-g', '--granularity', type=int, required=True, choices=[1, 2, 3])
    train_parser.add_argument('-d', '--device', type=str, default=None, choices=['cuda', 'cpu'])
    train_parser.add_argument('-n', '--negative-frac', type=float, default=0.5)
    train_parser.add_argument('-r', '--random-state', type=int, default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate command
    match args.command:
        case 'download-data':
            download_data()
        case 'train':
            train(
                model_name=args.model_name,
                granularity=args.granularity,
                device=args.device,
                negative_frac=args.negative_frac,
                random_state=args.random_state)
        case None:
            parser.print_help()
