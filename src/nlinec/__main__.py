import argparse

from .data.load import download_data


def main() -> None:
    parser = argparse.ArgumentParser()

    # Add a subcommand for downloading data
    subparsers = parser.add_subparsers(dest='command')

    # Add a subcommand for downloading data
    subparsers.add_parser('download-data')

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate command
    match args.command:
        case 'download-data':
            download_data()
        case None:
            parser.print_help()
