from datetime import datetime

import pytest

from stormworkflow.prep.hurricane_data import trackstart_from_file

def test_leadtime_pick(leadtime_file):

    # Always picks first
    assert datetime(2049, 10, 3, 6) == trackstart_from_file(
        leadtime_file=leadtime_file,
        nhc_code="al082049",
        leadtime=48)

    assert datetime(2050, 9, 28, 0) == trackstart_from_file(
        leadtime_file=leadtime_file,
        nhc_code="al142050",
        leadtime=24)
