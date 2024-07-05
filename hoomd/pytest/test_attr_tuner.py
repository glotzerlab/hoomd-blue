# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

from hoomd.tune.attr_tuner import ManualTuneDefinition


@pytest.fixture
def attr_dict():
    return dict(y=42, x=7, target=3, domain=None)


@pytest.fixture
def attr_definition(attr_dict):
    return ManualTuneDefinition(get_y=lambda: attr_dict['y'],
                                get_x=lambda: attr_dict['x'],
                                set_x=lambda x: attr_dict.__setitem__('x', x),
                                target=attr_dict['target'],
                                domain=attr_dict['domain'])


@pytest.fixture
def alternate_definition():
    return ManualTuneDefinition(
        get_y=lambda: 46,
        get_x=lambda: 1293,
        set_x=lambda x: None,
        target='foo',
    )


class TestManualTuneDefinition:

    def test_getting_attrs(self, attr_dict, attr_definition):
        assert attr_dict['x'] == attr_definition.x
        assert attr_dict['y'] == attr_definition.y
        assert attr_dict['target'] == attr_definition.target
        assert attr_dict['domain'] == attr_definition.domain

    def test_setting_attrs(self, attr_dict, attr_definition):
        attr_definition.x = 5
        assert attr_dict['x'] == attr_definition.x
        assert attr_definition.x == 5

        attr_definition.target = 1
        assert attr_dict['target'] != attr_definition.target
        assert attr_definition.target == 1

        attr_definition.domain = (0, None)
        assert attr_dict['domain'] != attr_definition.domain
        assert attr_definition.domain == (0, None)

        with pytest.raises(AttributeError):
            attr_definition.y = 43

    def test_domain_wrapping(self, attr_definition):
        domain_clamped_pairs = [((0, None), [(1, 1), (2, 2), (-1, 0),
                                             (1000, 1000)]),
                                ((None, 5), [(-1, -1), (-1000, -1000),
                                             (4.9, 4.9), (5.01, 5)]),
                                (None, [(1000, 1000), (-1000, -1000)])]
        for domain, value_pairs in domain_clamped_pairs:
            attr_definition.domain = domain
            for x, clamped_x in value_pairs:
                assert clamped_x == attr_definition.clamp_into_domain(x)

    def test_setting_x_with_wrapping(self, attr_definition):
        attr_definition.domain = (-5, 5)
        attr_definition.x = -6
        assert attr_definition.x == -5
        attr_definition.x = 6
        assert attr_definition.x == 5

    def test_in_domain(self, attr_definition):
        domain_check_pairs = [((0, None), [(1, True), (2, True), (-1, False),
                                           (1000, True)]),
                              ((None, 5), [(-1, True), (-1000, True),
                                           (4.9, True), (5.01, False)]),
                              (None, [(1000, True), (-1000, True)])]
        for domain, check_pairs in domain_check_pairs:
            attr_definition.domain = domain
            for x, in_domain in check_pairs:
                assert in_domain == attr_definition.in_domain(x)

    def test_hash(self, attr_definition, alternate_definition):
        assert hash(attr_definition) == hash(attr_definition)
        assert hash(attr_definition) != hash(alternate_definition)

    def test_eq(self, attr_definition, alternate_definition):
        assert attr_definition == attr_definition
        assert attr_definition != alternate_definition
