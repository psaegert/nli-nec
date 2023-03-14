def stringify_context(context: list) -> str:
    """
    Convert a list of strings into a string.

    Parameters
    ----------
    context : list(str)
        A list of strings.

    Returns
    -------
    context : str
        The combined string with spaces between each word and removed mid-word spaces.
    """
    # Convert the list into a string
    context_string = ' '.join(context)

    # Remove spaces before these tokens
    no_space_before = ['.', ',', ';', ':', '!', '?', "n't", "'s"]
    for token in no_space_before:
        context_string = context_string.replace(' ' + token, token)

    return context_string


def get_granularity(type: str) -> int:
    """
    Get the granularity of the type by counting the number of '/' in the type_str.

    Parameters
    ----------
    type : str
        The type of the mention.

    Returns
    -------
    granularity : int
        The granularity of the type, measured from the root of the ontology.
    """
    # Count the number of '/' in the type
    return type.count('/')


def get_type(type: str, granularity: int = -1) -> str | None:
    """
    Get the type of the mention at the specified granularity.

    Parameters
    ----------
    type : str
        The type of the mention.
    granularity : int, optional
        The granularity of the type, measured from the root of the ontology, by default -1 (finest granularity)

    Returns
    -------
    type : str
        The type of the mention at the specified granularity.
    """
    # Split the type into a list
    type_hierarchy = type.split('/')

    # If the granularity is -1, return the finest granularity
    if granularity == -1:
        return type_hierarchy[-1]

    # Otherwise, return the type at the specified granularity
    # If the granularity is greater than the number of types, return None
    if granularity >= len(type_hierarchy):
        return None
    else:
        return type_hierarchy[granularity]


def construct_hypothesis(entity: str, type: str) -> str:
    """
    Construct the hypothesis for the entity and type.

    Parameters
    ----------
    entity : str
        The entity.
    type : str
        The type of the entity.

    Returns
    -------
    hypothesis : str
        The hypothesis for the entity and type.
    """
    return f'{entity} is a {type}.'


def combine_premise_hypothesis(premise: str, hypothesis: str) -> str:
    """
    Combine the premise and hypothesis into a single string.

    Parameters
    ----------
    premise : str
        The premise.
    hypothesis : str
        The hypothesis.

    Returns
    -------
    combined : str
        The combined premise and hypothesis.
    """
    return f'{premise}</s><s>{hypothesis}'
