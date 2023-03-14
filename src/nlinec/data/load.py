import json
import os
import shutil

import pandas as pd
import urllib3
from tqdm import tqdm

from ..utils import get_data_dir
from .preprocessing import get_type, stringify_context


def download_data(target_dir: str = None) -> None:
    """
    Download the Ultra-Fine Entity Typing (ACL 2018) data from the web.
    For more information, see https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html

    Parameters
    ----------
    target_dir : str, optional
        The directory to download the data to. The default is 'data'.
    """
    DATA_URL = 'http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz'

    if target_dir is None:
        target_dir = get_data_dir()

    # Create the data directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Download the data
    http = urllib3.PoolManager()
    with http.request('GET', DATA_URL, preload_content=False) as r, open('data/ultrafine_acl18.tar.gz', 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

    # Extract the data
    shutil.unpack_archive('data/ultrafine_acl18.tar.gz', 'data')

    # Remove the tar.gz file
    os.remove('data/ultrafine_acl18.tar.gz')

    # Move the files form the 'data/release/ontonotes/' directory to the 'data' directory
    for filename in os.listdir('data/release/ontonotes/'):
        shutil.move(f'data/release/ontonotes/{filename}', f'data/{filename}')

    # Remove the 'data/release/' directory
    shutil.rmtree('data/release')


def load_data(filename: str, explode: bool = False) -> pd.DataFrame:
    """
    Load the data from the json file.

    Parameters
    ----------
    filename : str
        The name of the json file.
    explode : bool, optional
        Whether to explode the 'y_str' column to convert the list of possible types into a separate row for each type.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the data.
    """
    # Read the json file in the data directory
    instances = []
    with open(os.path.join(get_data_dir(), filename), 'r') as f:
        # Read each line of the file
        for line in tqdm(f):
            # Convert the line into a dictionary and append the dictionary to the list
            instances.append(json.loads(line))

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(instances)

    # Drop the 'annot_id', 'y_type_str', 'y', and 'y_type', 'original_y_str columns
    df.drop(['annot_id', 'y_type_str', 'y', 'y_type', 'original_y_str'], axis=1, inplace=True, errors='ignore')

    # Convert the 'left_context_token' and 'right_context_token' columns into a string
    df['left_context_token'] = df['left_context_token'].apply(stringify_context)
    df['right_context_token'] = df['right_context_token'].apply(stringify_context)

    # Reconstruct the sentence from the 'left_context_token', 'mention_span', and 'right_context_token' columns
    df['sentence'] = df['left_context_token'] + ' ' + df['mention_span'] + ' ' + df['right_context_token']

    # Drop the 'left_context_token' and 'right_context_token' columns
    df.drop(['left_context_token', 'right_context_token'], axis=1, inplace=True, errors='ignore')

    # For each instance and for each type in the 'y_str' column, create a new row
    if explode:
        df = df.explode('y_str')

    # Rename the 'y_str' column to 'full_type'
    df.rename(columns={'y_str': 'full_type'}, inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    return df


def get_all_types(granularity: int = -1) -> pd.DataFrame:
    """
    Get all the types in the training data.

    Parameters
    ----------
    granularity : int, optional
        The granularity of the type, measured from the root of the ontology, by default -1 (finest granularity)

    Returns
    -------
    all_types : pd.DataFrame
        A DataFrame containing all the full types and the types at the specified granularity.
    """
    all_types_path = get_data_dir('derived', 'all_types.csv')

    if os.path.exists(all_types_path):
        # If the file exists, load the types from the file
        all_types = pd.read_csv(all_types_path)
    else:
        # If the file doesn't exist, take the types from the training data and save them
        train_data = load_data('g_train.json')
        all_types = pd.DataFrame(train_data['full_type'].unique(), columns=['full_type'])

        # Save the types to a file
        all_types.to_csv(all_types_path, index=False)

    # Get the types at the specified granularity
    all_types['type'] = all_types['full_type'].apply(lambda x: get_type(x, granularity=granularity))

    return all_types
