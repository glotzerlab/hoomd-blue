import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import pytest

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

sph_union_args5 = sph_union_args1
sph_union_args6 = sph_union_args2
sph_union_args7 = sph_union_args3
sph_union_args8 = sph_union_args4
sph_union_args9 = sph_union_args1
sph_union_args10 = sph_union_args2
sph_union_args11 = sph_union_args3
sph_union_args12 = sph_union_args4




def test_dict_conversions():
        
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

def test_shape_params():

    mc = hoomd.hpmc.integrate.SphereUnion(23456)

    mc.shape['A'] = dict()
    print(mc.shape['A'])
    assert mc.shape['A']['shapes'] is None
    assert mc.shape['A']['positions'] is None
    assert mc.shape['A']['orientations'] is None
    assert mc.shape['A']['overlap'] == 1
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False


    

def test_shape_params_attached(device, dummy_simulation_factory):
    
    mc = hoomd.hpmc.integrate.SphereUnion(23456)
    mc.shape['A'] = sph_union_args1
    mc.shape['B'] = sph_union_args2
    mc.shape['C'] = sph_union_args3
    mc.shape['D'] = sph_union_args4

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D'])
    sim.operations.add(mc)
    sim.operations.schedule()

    assert mc.shape['A']['shapes'] == sph_union_args1['shapes']
    assert mc.shape['A']['positions'] == sph_union_args1['positions']
    assert mc.shape['A']['orientations'] == sph_union_args1['orientations']
    assert mc.shape['A']['overlap'] == sph_union_args1['overlap']
    assert mc.shape['A']['capacity'] == sph_union_args1['capacity']
    assert mc.shape['A']['ignore_statistics'] == sph_union_args1['ignore_statistics']

    assert mc.shape['B']['shapes'] == sph_union_args2['shapes']
    assert mc.shape['B']['positions'] == sph_union_args2['positions']
    assert mc.shape['B']['orientations'] == sph_union_args2['orientations']
    assert mc.shape['B']['overlap'] == sph_union_args2['overlap']
    assert mc.shape['B']['capacity'] == sph_union_args2['capacity']
    assert mc.shape['B']['ignore_statistics'] == sph_union_args2['ignore_statistics']

    assert mc.shape['C']['shapes'] == sph_union_args3['shapes']
    assert mc.shape['C']['positions'] == sph_union_args3['positions']
    assert mc.shape['C']['orientations'] == sph_union_args3['orientations']
    assert mc.shape['C']['overlap'] == sph_union_args3['overlap']
    assert mc.shape['C']['capacity'] == sph_union_args3['capacity']
    assert mc.shape['C']['ignore_statistics'] == sph_union_args3['ignore_statistics']
    
    assert mc.shape['D']['shapes'] == sph_union_args4['shapes']
    assert mc.shape['D']['positions'] == sph_union_args4['positions']
    assert mc.shape['D']['orientations'] == sph_union_args4['orientations']
    assert mc.shape['D']['overlap'] == sph_union_args4['overlap']
    assert mc.shape['D']['capacity'] == sph_union_args4['capacity']
    assert mc.shape['D']['ignore_statistics'] == sph_union_args4['ignore_statistics']


    sph_union_args1['shapes'] = 'invalid'
    sph_union_args2['shapes'] = 1
    sph_union_args3['shapes'] = [1, 2, 3]
    sph_union_args4['orientations'] = 'invalid'
    
    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args1

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args2
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4
    
    sph_union_args5['orientations'] = 1
    sph_union_args6['positions'] = 1
    sph_union_args7['positions'] = [1, 2, 3]
    sph_union_args8['positions'] = 'invalid'
        
    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args5

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args6
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args7
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args8
        
    
    sph_union_args9['overlap'] = 'invalid'
    sph_union_args10['capacity'] = 'invalid'
    sph_union_args11['capacity'] = [1, 2, 3]
    sph_union_args12['ignore_statistics'] = 'invalid'    
    
    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args9

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args10
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args11
        
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args12

