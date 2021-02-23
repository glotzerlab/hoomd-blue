import hoomd
import numpy as np
import pytest

def test_before_attaching():
    trigger = hoomd.trigger.Periodic(100)
    zm = hoomd.md.update.ZeroMomentum(trigger)
    assert zm.trigger is trigger

    trigger = hoomd.trigger.Periodic(10, 30)
    zm.trigger = trigger
    assert zm.trigger is trigger

