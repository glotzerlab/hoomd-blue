import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import pytest

@pytest.fixture(scope='session')
def dummy_integrator_args():
    sph_args_1 = {'diameter': 1, 'orientable': 0, 'ignore_statistics': 1}
    sph_args_2 = {'diameter': 9, 'orientable': 1, 'ignore_statistics': 1}
    sph_args_3 = {'diameter': 4, 'orientable': 0, 'ignore_statistics': 0}

    sph_union_args1 = {'shapes': [sph_args_1, sph_args_2],
                       'positions': [(0, 0, 0), (0, 0, 1)],
                       'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                       'overlap': [1, 1],
                       'capacity': 4,
                       'ignore_statistics': 1}

    sph_union_args2 = {'shapes': [sph_args_1, sph_args_3],
                       'positions': [(1, 0, 0), (0, 0, 1)],
                       'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                       'overlap': [1, 2],
                       'capacity': 3,
                       'ignore_statistics': 1}

    sph_union_args3 = {'shapes': [sph_args_3, sph_args_2],
                       'positions': [(1, 1, 0), (1, 0, 1)],
                       'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                       'overlap': [1, 0],
                       'capacity': 2,
                       'ignore_statistics': 0}

    sph_union_args4 = {'shapes': [sph_args_1, sph_args_2, sph_args_3],
                       'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)],
                       'orientations': [(1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 0, 1)],
                       'overlap': [1, 1, 1],
                       'capacity': 4,
                       'ignore_statistics': 1}
                       
    return sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4

def test_dict_conversions(dummy_integrator_args):
    
    (sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4) = dummy_integrator_args
    
    test_sph_union1 = _hpmc.SphereUnionParams(sph_union_args1)
    test_sph_dict1 = test_sph_union1.asDict()
    assert test_sph_dict1 == sph_union_args1

    test_sph_union2 = _hpmc.SphereUnionParams(sph_union_args2)
    test_sph_dict2 = test_sph_union2.asDict()
    assert test_sph_dict2 == sph_union_args2
    
    test_sph_union3 = _hpmc.SphereUnionParams(sph_union_args3)
    test_sph_dict3 = test_sph_union3.asDict()
    assert test_sph_dict3 == sph_union_args3

    test_sph_union4 = _hpmc.SphereUnionParams(sph_union_args4)
    test_sph_dict4 = test_sph_union4.asDict()
    assert test_sph_dict4 == sph_union_args4
'''
def test_shape_params(dummy_integrator_args):

    (sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4) = dummy_integrator_args

    mc = hoomd.hpmc.integrate.SphereUnion(23456)

    mc.shape['A'] = dict()
    assert mc.shape['A']['shapes'] is None
    assert mc.shape['A']['centers'] is None
    assert mc.shape['A']['diameters'] is None
    assert mc.shape['A']['overlap'] == 1
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['A'] = dict(shapes=sph_union_args1['shapes'])
    assert mc.shape['A']['shapes']  == sph_union_args1['shapes']
    assert mc.shape['A']['centers'] is None
    assert mc.shape['A']['diameters'] is None
    assert mc.shape['A']['overlap'] == 1
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['A'] = dict(centers=sph_union_args2['centers'],
                         ignore_statistics=True)
    assert mc.shape['A']['shapes'] is None
    assert mc.shape['A']['centers'] == sph_union_args2['centers']
    assert mc.shape['A']['diameters'] is None
    assert mc.shape['A']['overlap'] == 1
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is True
    
    mc.shape['A'] = sph_union_args3
    assert mc.shape['A']['shapes'] == sph_union_args3['shapes']
    assert mc.shape['A']['centers'] == sph_union_args3['centers']
    assert mc.shape['A']['diameters'] == sph_union_args3['diameters']
    assert mc.shape['A']['overlap'] == sph_union_args3['overlap']
    assert mc.shape['A']['capacity'] == sph_union_args3['capacity']
    assert mc.shape['A']['ignore_statistics'] == sph_union_args3['ignore_statistics']
    
    
def test_shape_params_attached(device, dummy_simulation_factory, dummy_integrator_args):

    (sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4) = dummy_integrator_args
    
    mc = hoomd.hpmc.integrate.SphereUnion(23456)
    mc.shape['A'] = sph_union_args1
    mc.shape['B'] = sph_union_args2
    mc.shape['C'] = sph_union_args3
    mc.shape['D'] = sph_union_args4

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D'])
    sim.operations.add(mc)
    sim.operations.schedule()

    assert mc.shape['A']['shapes'] == sph_union_args1['shapes']
    assert mc.shape['A']['centers'] == sph_union_args1['centers']
    assert mc.shape['A']['diameters'] == sph_union_args1['diameters']
    assert mc.shape['A']['overlap'] == sph_union_args1['overlap']
    assert mc.shape['A']['capacity'] == sph_union_args1['capacity']
    assert mc.shape['A']['ignore_statistics'] == sph_union_args1['ignore_statistics']

    assert mc.shape['B']['shapes'] == sph_union_args2['shapes']
    assert mc.shape['B']['centers'] == sph_union_args2['centers']
    assert mc.shape['B']['diameters'] == sph_union_args2['diameters']
    assert mc.shape['B']['overlap'] == sph_union_args2['overlap']
    assert mc.shape['B']['capacity'] == sph_union_args2['capacity']
    assert mc.shape['B']['ignore_statistics'] == sph_union_args2['ignore_statistics']

    assert mc.shape['C']['shapes'] == sph_union_args3['shapes']
    assert mc.shape['C']['centers'] == sph_union_args3['centers']
    assert mc.shape['C']['diameters'] == sph_union_args3['diameters']
    assert mc.shape['C']['overlap'] == sph_union_args3['overlap']
    assert mc.shape['C']['capacity'] == sph_union_args3['capacity']
    assert mc.shape['C']['ignore_statistics'] == sph_union_args3['ignore_statistics']
    
    assert mc.shape['D']['shapes'] == sph_union_args4['shapes']
    assert mc.shape['D']['centers'] == sph_union_args4['centers']
    assert mc.shape['D']['diameters'] == sph_union_args4['diameters']
    assert mc.shape['D']['overlap'] == sph_union_args4['overlap']
    assert mc.shape['D']['capacity'] == sph_union_args4['capacity']
    assert mc.shape['D']['ignore_statistics'] == sph_union_args4['ignore_statistics']


    sph_union_args1['shapes'] = 'invalid'
    sph_union_args2['shapes'] = 1
    sph_union_args3['shapes'] = [1, 2, 3]
    sph_union_args4['diameters'] = 'invalid'
    
    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args1

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args2
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4
        
    
    (sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4) = dummy_integrator_args
    
    sph_union_args1['diameters'] = 1
    sph_union_args2['centers'] = 1
    sph_union_args3['centers'] = [1, 2, 3]
    sph_union_args4['centers'] = 'invalid'
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args1

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args2
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4
        
        
    (sph_union_args1, sph_union_args2, sph_union_args3, sph_union_args4) = dummy_integrator_args
    
    sph_union_args1['overlap'] = 'invalid'
    sph_union_args2['capacity'] = 'invalid'
    sph_union_args3['capacity'] = [1, 2, 3]
    sph_union_args4['ignore_statistics'] = 'invalid'    
    
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args1

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args2
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4
'''
