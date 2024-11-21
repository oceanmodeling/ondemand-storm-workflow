from copy import deepcopy

import pytest

from stormworkflow.main import handle_input_version, CUR_INPUT_VER


def test_no_version_specified(conf_latest):
    conf_latest.pop('input_version')
    with pytest.warns(UserWarning):
        handle_input_version(conf_latest)

    assert conf_latest['input_version'] == str(CUR_INPUT_VER)
    

def test_invalid_version_specified(conf_latest):

    invalid_1 = deepcopy(conf_latest)
    invalid_1['input_version'] = (
        f'{CUR_INPUT_VER.major}.{CUR_INPUT_VER.minor}.{CUR_INPUT_VER.micro + 1}'
    )
    with pytest.raises(ValueError) as e:
        handle_input_version(invalid_1)

    assert "max" in str(e.value).lower()


    invalid_2 = deepcopy(conf_latest)
    invalid_2['input_version'] = 'a.b.c'
    with pytest.raises(ValueError) as e:
        handle_input_version(invalid_2)
    assert "invalid version" in str(e.value).lower()


def test_v0_0_1_to_latest(conf_v0_0_1, conf_latest):
    handle_input_version(conf_v0_0_1)
    assert conf_latest == conf_v0_0_1


def test_v0_0_2_to_latest(conf_v0_0_2, conf_latest):
    handle_input_version(conf_v0_0_2)
    assert conf_latest == conf_v0_0_2


def test_v0_0_3_to_latest(conf_v0_0_3, conf_latest):
    handle_input_version(conf_v0_0_3)
    assert conf_latest == conf_v0_0_3


def test_v0_0_4_to_latest(conf_v0_0_4, conf_latest):
    handle_input_version(conf_v0_0_4)
    assert conf_latest == conf_v0_0_4


def test_v0_0_5_to_latest(conf_v0_0_5, conf_latest):
    handle_input_version(conf_v0_0_5)
    assert conf_latest == conf_v0_0_5
