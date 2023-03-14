import numpy as np
import pandas as pd
import transformers
from scipy.special import softmax as scipy_softmax
from torch.nn.functional import softmax as torch_softmax
from tqdm import tqdm

from .data.preprocessing import combine_premise_hypothesis, construct_hypothesis


def predict_heads(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, input_text: str | list[str]) -> np.ndarray:
    """
    Predicts the entailment of the input text.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model to use.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use.
    input_text : str
        The input text.

    Returns
    -------
    outputs : np.ndarray, shape (n_samples, 3)
        The entailment probabilities, ordered as follows: contradiction, neutral, entailment.
    """
    # Convert the input text to a list if it is a string
    if isinstance(input_text, str):
        input_text = [input_text]

    # If the sentence and entity are other types of iterables, convert them to lists
    if isinstance(input_text, pd.Series):
        input_text = input_text.tolist()

    # Compute the outputs of the model
    input_ids, attention_mask = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding=True).values()
    outputs = model(input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

    # Return the probabilities of the model
    return torch_softmax(outputs[0], dim=1).detach().cpu().numpy()


def predict_probabilities(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, sentence: str | list[str], entity: str | list[str], types: list[str], normalize: str | None = None, hypothesis_only: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Predicts the type of the entity in the sentence.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model to use.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use.
    sentence : str | list[str]
        The sentence in which the entity occurs.
    entity : str | list[str]
        The entity.
    types : list[str]
        A list of possible types to choose from.
    normalize : str | None , optional
        The normalization method to use to generate a probability distribution along the type axis. Can be 'mean', 'softmax', or None. If None, no normalization is performed.
    hypothesis_only : bool, optional
        Whether to only use the hypothesis (i.e. '<entity> is a <type>') in the prediction.
    verbose : bool, optional
        Display a progress bar.

    Returns
    -------
    outputs : np.ndarray, shape (len(sentence), len(types), 3)
        The entailment probabilities for each type, ordered as follows: contradiction, neutral, entailment.
    """
    # Check that the sentence and entity are of the same type
    if type(sentence) != type(entity):
        raise ValueError('sentence and entity must be of the same type (str or list[str])')

    # Convert the sentence and entity to lists if they are strings
    if isinstance(sentence, str):
        sentence = [sentence]

    if isinstance(entity, str):
        entity = [entity]

    # If the sentence and entity are other types of iterables, convert them to lists
    if isinstance(sentence, pd.Series):
        sentence = sentence.tolist()

    if isinstance(entity, pd.Series):
        entity = entity.tolist()

    # Check that the sentence and entity are of the same length
    if len(sentence) != len(entity):
        raise ValueError('sentence and entity must have the same length')

    # Check that the normalize parameter is valid
    if normalize not in ['mean', 'softmax', None]:
        raise ValueError(f'normalize must be "mean", "softmax", or None, not {normalize}')

    # Predict the contradiction, non-entailment and entailment probabilities for each sentence + entity and type
    # Construct the hypothesis for each type and combine it with the sentence before predicting the entailment
    outputs = np.empty((len(sentence), len(types), 3))
    pbar = tqdm(total=len(sentence), disable=not verbose, desc='Predicting types')
    for i, (s, e) in enumerate(zip(sentence, entity)):
        outputs[i] = predict_heads(
            model=model,
            tokenizer=tokenizer,
            input_text=[combine_premise_hypothesis(premise=s, hypothesis=construct_hypothesis(entity=e, type=t), hypothesis_only=hypothesis_only) for t in types]
        )
        pbar.update(1)

    # Normalize the outputs
    match normalize:
        case 'mean':
            outputs /= outputs.sum(axis=1, keepdims=True)
        case 'softmax':
            outputs = scipy_softmax(outputs, axis=1)
        case None:
            pass

    return outputs


def predict_type(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, sentence: str | list[str], entity: str | list[str], types: list[str], return_str: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Predicts the type of the entity in the sentence.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model to use.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use.
    sentence : str | list[str]
        The sentence in which the entity occurs.
    entity : str | list[str]
        The entity.
    types : list[str]
        A list of possible types to choose from.
    return_str : bool, optional
        Whether to return the type as a string or as an index.
    verbose : bool, optional
        Display a progress bar.

    Returns
    -------
    outputs : np.ndarray, shape (len(sentence),)
        The predicted type for each sentence.
    """
    # Get the entailment probability for each type
    entailment_probability = predict_probabilities(model=model, tokenizer=tokenizer, sentence=sentence, entity=entity, types=types, verbose=verbose)[:, :, 2]

    # Find the most probable type
    most_probable_type_index = entailment_probability.argmax(axis=1)

    # If return_str is True, return the type as a string
    if return_str:
        return np.array([types[i] for i in most_probable_type_index])

    # Otherwise, return the type as an index
    return most_probable_type_index
