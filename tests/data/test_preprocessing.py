from nlinec.data.preprocessing import stringify_context, get_granularity, get_type, construct_hypothesis, combine_premise_hypothesis


def test_stringify_context() -> None:
    context = ['This', 'is', 'a', 'test', '.']

    assert stringify_context(context) == 'This is a test.'


def test_get_granularity() -> None:
    assert get_granularity('/other') == 1
    assert get_granularity('/other/test') == 2
    assert get_granularity('/correct/horse/battery') == 3


def test_get_type() -> None:
    assert get_type('/other') == 'other'
    assert get_type('/other/test') == 'test'
    assert get_type('/correct/horse/battery') == 'battery'
    assert get_type('/correct/horse/battery', granularity=1) == 'correct'
    assert get_type('/correct/horse/battery', granularity=2) == 'horse'
    assert get_type('/correct/horse/battery', granularity=4) is None


def test_construct_hypothesis() -> None:
    entity = 'my_entity'
    type = 'my_type'

    assert construct_hypothesis(entity, type) == 'my_entity is a my_type.'


def test_combine_premise_hypothesis() -> None:
    premise = 'This is a premise.'
    hypothesis = 'This is a hypothesis.'

    assert combine_premise_hypothesis(premise, hypothesis) == 'This is a premise.</s><s>This is a hypothesis.'
