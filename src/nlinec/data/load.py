import json
import os
import shutil

import numpy as np
import pandas as pd
import urllib3
from tqdm import tqdm

from ..utils import get_data_dir, get_labels
from .preprocessing import get_granularity, get_type, stringify_context


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

    print(f'Downloading the data from {DATA_URL} to {target_dir}...')

    # Create the data directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Download the data
    http = urllib3.PoolManager()
    with http.request('GET', DATA_URL, preload_content=False) as r, open('data/ultrafine_acl18.tar.gz', 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

    # Extract the data
    print('Extracting the data...')
    shutil.unpack_archive('data/ultrafine_acl18.tar.gz', 'data')

    # Remove the tar.gz file
    os.remove('data/ultrafine_acl18.tar.gz')

    # Move the files form the 'data/release/ontonotes/' directory to the 'data' directory
    for filename in os.listdir('data/release/ontonotes/'):
        shutil.move(f'data/release/ontonotes/{filename}', f'data/{filename}')

    # Remove the 'data/release/' directory
    shutil.rmtree('data/release')


def get_positive_data(filename: str, explode: bool = False) -> pd.DataFrame:
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
        for line in tqdm(f, desc=f'Loading {filename}'):
            # Convert the line into a dictionary and append the dictionary to the list
            instances.append(json.loads(line))

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(instances)

    # Drop the 'annot_id', 'y_type_str', 'y', 'y_type', and 'original_y_str' columns
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

    # Calculate the granularity of the type
    if explode:
        df['granularity'] = df['full_type'].apply(get_granularity)

    # Add the 'label' column
    labels = get_labels()
    df['label'] = labels['ENTAILMENT']
    df['label'] = df['label'].astype(int)

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
    all_types_path = os.path.join(get_data_dir(), 'derived', 'all_types.csv')
    os.makedirs(os.path.dirname(all_types_path), exist_ok=True)

    if os.path.exists(all_types_path):
        # If the file exists, load the types from the file
        all_types = pd.read_csv(all_types_path)
    else:
        # If the file doesn't exist, take the types from the training data and save them
        train_data = get_positive_data('augmented_train.json', explode=True)
        all_types = pd.DataFrame(train_data['full_type'].unique(), columns=['full_type'])

        # Save the types to a file
        all_types.to_csv(all_types_path, index=False)

    # Get the types at the specified granularity
    all_types['type'] = all_types['full_type'].apply(lambda x: get_type(x, granularity=granularity))

    return all_types


def get_ambiguity_index(filename: str = 'augmented_train.json') -> dict:
    """
    Get the ambiguity index (a dictionary that maps each entity to a dictionary of types and their counts).

    Parameters
    ----------
    filename : str, optional
        The name of the json file, by default 'augmented_train.json'

    Returns
    -------
    ambiguity_index : dict
        The ambiguity index.
    """

    index_file = os.path.join(get_data_dir(), 'derived', f'ambiguity_index_{os.path.splitext(filename)[0]}.json')

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            ambiguity_index = json.load(f)
    else:
        # Load the training data
        data = get_positive_data('augmented_train.json', explode=True)

        # Find the most ambiguous entities
        ambiguity_index = {}
        for i, row in tqdm(data.iterrows(), total=len(data), desc='Building the ambiguity index'):
            # If the entity is not in the index, add an empty dict for its types
            if row['mention_span'] not in ambiguity_index:
                ambiguity_index[row['mention_span']] = {}

            # If the entity type is not in the index, add an empty count
            if row['full_type'] not in ambiguity_index[row['mention_span']]:
                ambiguity_index[row['mention_span']][row['full_type']] = 0

            # Increment the count
            ambiguity_index[row['mention_span']][row['full_type']] += 1

        # Find the most ambiguous entities, i.e. the ones with the most types
        ambiguity_index = {k: v for k, v in sorted(ambiguity_index.items(), key=lambda item: len(item[1]), reverse=True)}

        with open(index_file, 'w') as f:
            json.dump(ambiguity_index, f, indent=4)

    return pd.DataFrame(ambiguity_index).T.fillna(0)


def get_negative_type_candidates(entity: str, ambiguity_index: pd.DataFrame, granularity: int, size: int = None) -> list | None:
    """
    Get the negative type candidates for the specified entity at the specified granularity.

    Parameters
    ----------
    entity : str
        The entity.
    ambiguity_index : pd.DataFrame
        The ambiguity index.
    granularity : int
        The granularity to sample the negative type candidates at.
    size : int, optional
        The number of negative type candidates to sample, by default None (return all the negative type candidates)

    Returns
    -------
    negative_type_candidates : list
        The negative type candidates.
    """
    try:
        # Get the types at the specified granularity
        negative_type_candidates = ambiguity_index.drop(columns=[t for t in ambiguity_index.columns if not get_granularity(t) == granularity]).loc[entity]

        # Remove the entities with count > 0
        negative_type_candidates = negative_type_candidates[negative_type_candidates == 0].index.tolist()

        if len(negative_type_candidates) == 0:
            return None

        if size is not None:
            return np.random.choice(negative_type_candidates, size=size, replace=False).tolist()[0]

        return negative_type_candidates
    except KeyError:
        return None


def get_negative_data(positive_file: str = 'augmented_train.json', random_state: int = None, step: int = 1000) -> pd.DataFrame:
    """
    Sample negative data based on a file full of positive NEC data and store it in a json file.

    Parameters
    ----------
    filename : str, optional
        The name of the positive json file, by default 'augmented_train.json'
    random_state : int, optional
        The random state, by default None
    step : int, optional
        The step size for sampling and logging, by default 1000

    Returns
    -------
    negative_data : pd.DataFrame
        A DataFrame containing the negative data in the 'full_type' column. May contain NaN values in case no negative type candidates were found.
    """

    random_state_suffix = f'_{random_state}' if random_state is not None else ''
    negative_file = os.path.join(get_data_dir(), 'derived', 'negative_data', f'{positive_file}{random_state_suffix}.csv')
    os.makedirs(os.path.dirname(negative_file), exist_ok=True)

    if os.path.exists(negative_file):
        print(f'Loading negative data from {negative_file}...')
        negative_data = pd.read_csv(negative_file)
    else:
        print(f'Generating negative data from {positive_file}...')
        data = get_positive_data(positive_file, explode=True)
        ambiguity_index = get_ambiguity_index(positive_file)

        if random_state is not None:
            np.random.seed(random_state)

        negative_data = data.copy()
        pbar = tqdm(range(0, len(negative_data), step), total=len(negative_data), desc='Sampling negative data')
        for i in range(0, len(negative_data), step):
            negative_data.loc[i:i + step, 'full_type'] = negative_data.loc[i:i + step, ['mention_span', 'granularity']].apply(lambda x: get_negative_type_candidates(x['mention_span'], ambiguity_index, x['granularity'], size=1), axis=1)
            pbar.update(step)

        print(f'Storing negative data in {negative_file}...')
        negative_data.to_csv(negative_file, index=False)

    # Add the 'label' column
    labels = get_labels()
    negative_data['label'] = labels['NEUTRAL']

    return negative_data


def combine_positive_negative_data(positive_data: pd.DataFrame, negative_data: pd.DataFrame, frac: float = 0.5, random_state: int = None) -> pd.DataFrame:
    """
    Combine the positive and negative data by randomly replacing a fraction of the positive data with negative data or adding it to the positive data.

    Parameters
    ----------
    positive_data : pd.DataFrame
        The positive data (entailment).
    negative_data : pd.DataFrame
        The negative data (not entailment = neutral).
    frac : float, optional
        The fraction of the positive data that should be replaced with negative data, by default 0.5. If < 0, the negative data is added to the positive data.
    random_state : int, optional
        The random state for the random number generator, by default None

    Returns
    -------
    combined_data : pd.DataFrame
        The combined data.
    """
    if frac > 1:
        raise ValueError(f'frac must be <= 1, but is {frac}')

    if frac < 0:
        return pd.concat([positive_data, negative_data], ignore_index=True)

    # Mask for the negative data that has a full_type
    negative_type_available = negative_data['full_type'].notna()

    # Deterministic, random mask for the replacement of the positive data
    if random_state is not None:
        np.random.seed(random_state)
    random_mask = np.random.choice([True, False], size=len(positive_data), p=[frac, frac])

    # Replace the positive data with the negative data
    combined_data = positive_data.copy()
    combined_data.loc[random_mask & negative_type_available, ['full_type', 'label']] = negative_data.loc[random_mask & negative_type_available, ['full_type', 'label']].values

    return combined_data
