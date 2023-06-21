from hoomd import shape_plugin


def test_make_mysphere():
    intgrator = shape_plugin.integrate.MySphere()
