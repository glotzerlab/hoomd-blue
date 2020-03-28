from hoomd.parameterdicts import TypeParameterDict, RequiredArg
from hoomd.parameterdicts import AttachedTypeParameterDict
from hoomd.pytest.dummy import DummyCppObj, DummySimulation
from hoomd.typeconverter import TypeConversionError
from pytest import fixture, raises


@fixture(scope='function')
def typedict_singleton_keys():
    return TypeParameterDict(**dict(foo=1,
                                    bar=lambda x: x,
                                    baz='hello'),
                             len_keys=1
                             )


@fixture(scope='module')
def valid_single_keys():
    return ['A', ['A', 'B']]


@fixture(scope='module')
def invalid_keys():
    return [1, dict(A=2), (dict(A=2), tuple())]


def single_keys_generator():
    yield from ['A', 'B', 'C']


def test_typeparamdict_key_validation_single(typedict_singleton_keys,
                                             valid_single_keys,
                                             valid_pair_keys,
                                             invalid_keys):
    '''Test the validation step of type parameter dictionaries.'''

    for valid_key in valid_single_keys + [single_keys_generator()]:
        typedict_singleton_keys._validate_and_split_key(valid_key)
    for invalid_key in invalid_keys:
        with raises(KeyError) as err_info:
            typedict_singleton_keys._validate_and_split_key(invalid_key)


@fixture(scope='function')
def typedict_pair_keys():
    return TypeParameterDict(**dict(foo=1,
                                    bar=lambda x: x,
                                    baz='hello'),
                             len_keys=2
                             )


@fixture(scope='module')
def valid_pair_keys():
    return [('A', 'B'), (['A', 'B'], 'B'), ('B', ['A', 'B']),
            [('A', 'B'), ('A', 'A')],
            (['A', 'C'], ['B', 'D'])
            ]


def pair_keys_generator():
    yield from [('A', 'B'), ('B', 'C'), ('C', 'D')]


def test_key_validation_pairs(typedict_pair_keys,
                              valid_single_keys,
                              valid_pair_keys,
                              invalid_keys):
    '''Test the validation step of pair keys in type parameter dictionaries.
    '''
    for valid_key in valid_pair_keys + [pair_keys_generator()]:
        typedict_pair_keys._validate_and_split_key(valid_key)
    for invalid_key in valid_single_keys + [single_keys_generator()]:
        with raises(KeyError) as err_info:
            typedict_pair_keys._validate_and_split_key(invalid_key)


@fixture(scope='module')
def expanded_single_keys():
    return [['A'], ['A', 'B']]


def test_key_expansion_single(typedict_singleton_keys,
                              valid_single_keys,
                              expanded_single_keys):
    '''Test expansion of single type keys.'''
    for expanded_keys, valid_key in zip(expanded_single_keys,
                                        valid_single_keys):
        for expected_key, given_key in zip(
                expanded_keys,
                typedict_singleton_keys._yield_keys(valid_key)
        ):
            if expected_key != given_key:
                raise Exception("Key {} != Key {}".format(expected_key,
                                                          given_key))


@fixture(scope='module')
def expanded_pair_keys():
    return [[('A', 'B')], [('A', 'B'), ('B', 'B')],
            [('A', 'B'), ('B', 'B')],
            [('A', 'B'), ('A', 'A')],
            [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
            ]


def test_key_expansion_pair(typedict_pair_keys,
                            valid_pair_keys,
                            expanded_pair_keys):
    '''Test key expansion of pair type keys.'''
    for expanded_keys, valid_key in zip(expanded_pair_keys,
                                        valid_pair_keys):
        for expected_key, given_key in zip(
                expanded_keys,
                typedict_pair_keys._yield_keys(valid_key)
        ):
            if expected_key != given_key:
                raise Exception("Key {} != Key {}".format(expected_key,
                                                          given_key))


def test_setting_dict_values(typedict_pair_keys,
                             valid_pair_keys):
    '''Test setting type parameter dicts with dict values.'''
    # Valid setting
    for ind, valid_key in enumerate(valid_pair_keys):
        typedict_pair_keys[valid_key] = dict(foo=ind)
        values = typedict_pair_keys[valid_key]
        try:
            for dict_vals in values.values():
                assert dict_vals['foo'] == ind
        # In case only a single type is returned
        except (TypeError, AttributeError):
            assert values['foo'] == ind


@fixture
def typedict_with_int():
    return TypeParameterDict(100., len_keys=1)


def test_setting_arg_values(typedict_with_int,
                            valid_single_keys):
    '''Test setting typeparam_dicts with non dict values.'''
    # Valid setting
    for ind, valid_key in enumerate(valid_single_keys):
        typedict_with_int[valid_key] = ind
        values = typedict_with_int[valid_key]
        try:
            for arg_val in values.values():
                assert int(arg_val) == ind and type(arg_val) == float
        # In case only a single type is returned
        except (TypeError, AttributeError):
            assert int(values) == ind and type(values) == float


def test_invalid_value_setting(typedict_with_int, typedict_singleton_keys):
    '''Test value validation on new dict keys and wrong types.'''
    # New dict key
    with raises(ValueError):
        typedict_singleton_keys['A'] = dict(boo=None)
    # Invalid types
    with raises(TypeConversionError):
        typedict_singleton_keys['A'] = 3.
    with raises(TypeConversionError):
        typedict_with_int['A'] = []


def test_defaults(typedict_singleton_keys):
    assert dict(foo=1, bar=RequiredArg, baz='hello') == \
        typedict_singleton_keys.default
    assert typedict_singleton_keys['FAKETYPE'] == \
        typedict_singleton_keys.default


def test_singleton_keys(typedict_singleton_keys, valid_single_keys,
                        expanded_single_keys):
    '''Test the keys function.'''
    assert list(typedict_singleton_keys.keys()) == []
    typedict_singleton_keys[valid_single_keys[-1]] = dict(bar=2)
    assert list(typedict_singleton_keys.keys()) == expanded_single_keys[-1]


def test_pair_keys(typedict_pair_keys, valid_pair_keys,
                   expanded_pair_keys):
    '''Test the keys function.'''
    assert list(typedict_pair_keys.keys()) == []
    typedict_pair_keys[valid_pair_keys[-1]] = dict(bar=2)
    assert list(typedict_pair_keys.keys()) == expanded_pair_keys[-1]


def test_changing_defaults(typedict_singleton_keys):
    typedict_singleton_keys.default = dict(bar='set')
    assert dict(foo=1, bar='set', baz='hello') == \
        typedict_singleton_keys.default


def test_attaching(typedict_singleton_keys):
    sim = DummySimulation()
    cpp_obj = DummyCppObj()
    typedict_singleton_keys['A'] = dict(bar='first')
    typedict_singleton_keys['B'] = dict(bar='second')
    return AttachedTypeParameterDict(
        cpp_obj, param_name='type_param',
        type_kind='particle_types',
        type_param_dict=typedict_singleton_keys,
        sim=sim)


@fixture(scope='function')
def attached_param_dict(typedict_singleton_keys):
    return test_attaching(typedict_singleton_keys)


def test_attached_default(attached_param_dict, typedict_singleton_keys):
    tp = typedict_singleton_keys
    assert tp._default == attached_param_dict._default
    assert tp._type_converter == attached_param_dict._type_converter


def test_attached_values(attached_param_dict):
    expected_values = dict(A=dict(foo=1, bar='first', baz='hello'),
                           B=dict(foo=1, bar='second', baz='hello'))
    for type_, expected_value in expected_values.items():
        assert attached_param_dict[type_] == expected_value


def test_attached_keys(attached_param_dict):
    assert list(attached_param_dict.keys()) == ['A', 'B']


def test_attached_type_error_raising(attached_param_dict):
    with raises(KeyError):
        attached_param_dict['C']
    with raises(KeyError):
        attached_param_dict['C'] = dict(bar='third')


def test_attached_set_error_raising(attached_param_dict):
    with raises(ValueError):
        attached_param_dict['A'] = dict(foo=2.)
    with raises(TypeConversionError):
        attached_param_dict['A'] = dict(foo='third')


def test_attached_value_setting(attached_param_dict):
    attached_param_dict['A'] = dict(bar='new')
    assert attached_param_dict['A']['bar'] == 'new'


def test_attach_dettach(attached_param_dict):
    tp = attached_param_dict.to_dettached()
    assert tp._default == attached_param_dict._default
    assert tp._type_converter == attached_param_dict._type_converter
    assert tp['A'] == attached_param_dict['A']
    assert tp['B'] == attached_param_dict['B']
    assert type(tp) == TypeParameterDict
