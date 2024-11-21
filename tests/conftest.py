from importlib.resources import files

import pytest
import yaml
from yaml import Loader


refs = files('stormworkflow.refs')
test_refs = files('data.refs')
input_v0_0_1 = test_refs.joinpath('input_v0.0.1.yaml')
input_v0_0_2 = test_refs.joinpath('input_v0.0.2.yaml')
input_v0_0_3 = test_refs.joinpath('input_v0.0.3.yaml')
input_v0_0_4 = test_refs.joinpath('input_v0.0.4.yaml')
input_v0_0_5 = test_refs.joinpath('input_v0.0.5.yaml')
input_latest = refs.joinpath('input.yaml')


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
def conf_v0_0_4():
    return read_conf(input_v0_0_4)

@pytest.fixture
def conf_v0_0_5():
    return read_conf(input_v0_0_5)

@pytest.fixture
def conf_latest():
    return read_conf(input_latest)
