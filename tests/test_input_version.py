from importlib.resources import files

import pytest
import yaml
from packaging.version import Version
from yaml import Loader, Dumper

from stormworkflow.main import handle_input_version, CUR_INPUT_VER


refs = files('tests.data.refs')
input_v0_0_1 = refs.joinpath('input_v0.0.1.yaml')
input_v0_0_2 = refs.joinpath('input_v0.0.2.yaml')


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
def conf_latest(conf_v0_0_2):
    return conf_v0_0_2


def test_no_version_specified(conf_latest):
    conf_latest.pop('input_version')
    with pytest.warns(UserWarning):
        handle_input_version(conf_latest)
        
    assert conf_latest['input_version'] == CUR_INPUT_VER
    
