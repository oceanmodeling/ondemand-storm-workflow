from copy import deepcopy
from importlib.resources import files

import pytest
import yaml
from packaging.version import Version
from yaml import Loader, Dumper

from stormworkflow.main import handle_input_version, CUR_INPUT_VER


refs = files('tests.data.refs')
input_v0_0_1 = refs.joinpath('input_v0.0.1.yaml')
input_v0_0_2 = refs.joinpath('input_v0.0.2.yaml')
input_v0_0_3 = refs.joinpath('input_v0.0.3.yaml')


def read_conf(infile):
    with open(infile, 'r') as yfile:
        conf = yaml.load(yfile, Loader=Loader)
    return conf


@pytest.fixture
def conf_v0_0_1():
    return read_conf(input_v0_0_1)


@pytest.fixture
def conf_v0_0_2():
    return read_conf(input_v0_0_2)


@pytest.fixture
def conf_v0_0_3():
    return read_conf(input_v0_0_3)


@pytest.fixture
def conf_latest(conf_v0_0_3):
    return conf_v0_0_3


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
