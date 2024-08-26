# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd import md
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)
import pytest
import numpy

import itertools
# Test parameters: class, class keyword arguments (blank), bond params, positions, orientations, force, energy, torques

sqrt2inv = 1/numpy.sqrt(2)


TOLERANCES = {"rtol": 1e-2, "atol": 1e-5}

# patch_test_parameters = [
#     (
#         hoomd.md.pair.aniso.JanusLJ,
#         {},
#         {"pair_params": {"epsilon": 1, "sigma": 1},
#          "envelope_params": {"alpha": numpy.pi/2,
#                              "omega": 10,
#                              "ni": (1,0,0),
#                              "nj": (1,0,0)
#                              }
#          },
#         [[0,0,0], [2,0,0]], # positions
#         [[1,0,0,0], [1, 0, 0, 0]], # orientations
#         [-8.245722889538097e-6, 0, 0],
#         -2.79291e-6, # energy
#         [[0,0,0], [0,0,0]] # todo put in right torque values
#     ),
#     (
#         hoomd.md.pair.aniso.JanusLJ,
#         {},
#         {"pair_params": {"epsilon": 1, "sigma": 1},
#          "envelope_params": {"alpha": 1.5707963267948966,
#                              "omega": 10,
#                              "ni": (1, 0, 0),
#                              "nj": (1, 0, 0)
#                              }
#          },
#         [[0, 0, 0], [0, 2, 1]],
#         [[1., 0., 0., 0.], [1., 0., 0., 0.]],
#         [0.03549087093887666, 0.0188928, 0.0094464],
#         -0.007936,
#         [[0., -0.01774543546943833, 0.03549087093887666],
#          [0., 0.01774543546943833, -0.03549087093887666]])
# ]

patch_test_parameters = [
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [-0.7774038792022364, -0.27267024474161755, -0.0015845806664157094], 0.34759769470591956, [[0.04993040659587277, -0.08648200106662383, -0.049115503752557785], [-0.05080192596240142, 0.08799151688923172, 0.21693434263119063]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [0.3228846356355689, -0.3509139637784533, 0.04755780916167393], 0.17244973958565052, [[0.10422831958202958, 0.018378264896216252, -0.2131628704093221], [-0.05270949048546891, -0.009294105311984296, -0.06958522398254269]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [-37.98029244596124, -6.562251931655476, -0.1382129529991447], 6.063749843631763, [[0.8710227363399076, -1.5086556338883903, -0.8568069717822429], [-0.8862261611698138, 1.534988738142842, 3.7843622274916076]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [25.19599436172106, -13.78023024194904, 4.148167008293145], 3.0083400936577576, [[1.8182354664412854, 0.32060397001463925, -3.718569892145524], [-0.919503119718846, -0.16213320885203827, -1.2138958268989883]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-0.01465432099787841, -0.027666666640927814, -0.020211692080024782], 0.011388283504224106, [[0.0016358613260541192, -0.0028333949308627323, -0.0016091628043164504], [-0.012752291970067747, 0.022087617605109952, -0.01668700041083728]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-0.009251658478899707, -0.05788074562922323, 0.0034007106684135454], 0.02378390349398191, [[0.01437494947939701, 0.0025346914395387955, -0.02939897242232397], [-0.010690998624191522, -0.0018851115081620442, 0.05047716252095921]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-0.3009628877978324, -1.84897812268115, -1.7629381114491105], 0.1986655934999705, [[0.028537168143350694, -0.0494278251284194, -0.028071419493466945], [-0.22246036040275277, 0.38531264688765143, -0.29110030841114953]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-1.2161923797379242, -2.727720178368872, 0.2966224905673739], 0.41490390554690204, [[0.25076719145578286, 0.044217021730139756, -0.5128572978011914], [-0.18650164320150397, -0.0328852716428886, 0.8805607488346467]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [-0.20401280991597603, -0.36176772675347946, -0.05694203866421653], 0.15352090296938442, [[0.022052393393207265, -0.03819586578553122, -0.021692481281465906], [-0.053370514658526345, 0.0924404430146672, -0.21073051906639378]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [0.022692364724683625, -0.012307094216380321, -0.008094853121204672], 0.006895655948872544, [[0.0041677223386576544, 0.0007348818964067269, -0.008523630241185349], [-0.01293678386324083, -0.002281104038683247, -0.013707933214014117]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [-4.62095340121715, -23.948817933906497, -4.966694015884345], 2.67813154561453, [[0.3846981698279066, -0.6663167757206949, -0.3784195982362874], [-0.9310345115751846, 1.6125990776482935, -3.676138166331529]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1., 1.], [1., 0., 0.]], [[1., 1., 1.], [1., 0., 0.]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [1.8606640164588026, -0.4005866881449048, -0.706062857946379], 0.12029289410877733, [[0.07270481382426026, 0.012819820304602875, -0.14869247503465216], [-0.2256787726805616, -0.039793256596755436, -0.2391312691337512]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [-0.03805968504495193, -0.06456499884225689, 0.03008727777126025], 0.02801228293641711, [[0., 0., 0.008427510855661064], [0.01654800277419314, -0.028661981568693252, -0.04900110619291556]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [0.004738801022916131, -0.004307629284897535, 0.006994233628431241], 0.002174698659917098, [[0., 0., -0.001989089530795274], [0.007576773054023575, 0.0013359895161292323, -0.0023215860823638385]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [-0.9159415058546319, -4.243782735080522, 2.624323012773965], 0.4886668665025968, [[0., 0., 0.14701569777801365], [0.2886755314051362, -0.5000006872956408, -0.8548113366127427]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.4480736161291701, 0., 0., -0.8939966636005579]], [0.37591752537110823, -0.1635185256558075, 0.6100627782731048], 0.037937042908685145, [[0., 0., -0.03469914074526231], [0.13217480185085467, 0.023305983736100832, -0.040499455141155355]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-0.22122614255103878, -0.024932958611667055, 0.057889843743764734], 0.08761098072205219, [[0., 0., 0.026357812134987807], [0.03183941405907059, -0.05514748283353311, 0.0715647331639529]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [0.12732524637729745, -0.17726950238751302, 0.04481708806670525], 0.08444761711985735, [[0., 0., -0.07724006742162585], [0.04854983737507447, 0.008560646238233868, -0.02682913269351473]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [-11.77816503494904, 2.165770753686239, 5.049365060463138], 1.5283503853592224, [[0., 0., 0.4598050609835814], [0.5554301566509452, -0.9620332513753777, 1.2484278409832568]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.9843766433940419, 0., 0., 0.17607561994858706]], [9.652760368011261, -7.221635485842196, 3.9091112354245268], 1.473166344037004, [[0., 0., -1.3474325459673837], [0.8469390714473051, 0.14933820931233416, -0.46802712346154485]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.9526279441628825, 0.55, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [-0.030186311835459993, 0.0064703551456309506, 0.015661951909555192], 0.00983514006853119, [[0., 0., 0.0029589073437045316], [0.008614073550255355, -0.014920013049177498, 0.019807405286184607]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [0.015375001171817795, -0.12504025632410554, 0.03980692058525533], 0.0540169790327473, [[0., 0., -0.04940666468398036], [0.0431223804174908, 0.007603639139977987, 0.05663541620200005]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [0.1905255888325765, 0.11, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [-1.7889971593891718, 1.051630994939726, 1.3660930421716653], 0.17157141707486956, [[0., 0., 0.05161735597207821], [0.15027023463888314, -0.2602756812598423, 0.3455349461062102]]),
    (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 0.778, "kappa": 1.42}, "envelope_params": {"alpha": 0.6981317007977318, "omega": 2}}, [[1., 1.2, 0.]], [[-0.8, -1.3, -1.02]], [[0, 0, 0], [-0.03820259908672467, 0.21665770566268577, 0]], [[1., 0., 0., 0.], [0.6333192030862997, 0., 0., 0.7738906815578891]], [0.41164150491537477, -5.635453277135799, 3.4721060028680535], 0.9423119115919261, [[0., 0., -0.8618861972180705], [0.7522585203990312, 0.13264347361417836, 0.9879898554259575]])]



@pytest.fixture(scope='session')
def patchy_snapshot_factory(device):

    def make_snapshot(position_i = numpy.array([0,0,0]),
                      position_j = numpy.array([2,0,0]),
                      orientation_i = (1,0,0,0),
                      orientation_j = (1,0,0,0),
                      dimensions = 3,
                      L=20                      
                      ):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            N = 2
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.position[:] = [position_i, position_j]
            snapshot.particles.orientation[:] = [orientation_i, orientation_j]
            snapshot.particles.types = ['A', 'B']
            snapshot.particles.typeid[:] = [0, 1]
            snapshot.particles.moment_inertia[:] = [(1,1,1)]*N
            snapshot.particles.angmom[:] = [(0,0,0,0)]*N
        return snapshot

    return make_snapshot


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_before_attaching(patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])
    for i,patch in enumerate(patches_A):
        # only normalized after attaching
        assert potential.patches['A'][i] == pytest.approx(patch)
    for i,patch in enumerate(patches_B):
        assert potential.patches['B'][i] == pytest.approx(patch)


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_after_attaching(patchy_snapshot_factory, simulation_factory,
                         patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):
    sim = simulation_factory(patchy_snapshot_factory())
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.05,
                                                    forces = [potential],
                                                    integrate_rotational_dof = True)
    sim.run(0)
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])
    for i,patch in enumerate(patches_A):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = nn / numpy.linalg.norm(nn)
        assert potential.patches['A'][i] == pytest.approx(patch)
    for i,patch in enumerate(patches_B):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = tuple(nn / numpy.linalg.norm(nn))
        assert potential.patches['B'][i] == pytest.approx(patch)


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_forces_energies_torques(patchy_snapshot_factory, simulation_factory,
                                 patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):

    snapshot = patchy_snapshot_factory(position_i = positions[0],
                                       position_j = positions[1],
                                       orientation_i = orientations[0],
                                       orientation_j = orientations[1])
    sim = simulation_factory(snapshot)

    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.005,
                                                    forces = [potential],
                                                    integrate_rotational_dof = True)
    sim.run(0)

    sim_forces = potential.forces
    sim_energy = potential.energy
    sim_torques = potential.torques
    if sim.device.communicator.rank == 0:

        sim_orientations = snapshot.particles.orientation

        numpy.testing.assert_allclose(sim_orientations, orientations, **TOLERANCES)
        
        numpy.testing.assert_allclose(sim_energy, energy, **TOLERANCES)

        numpy.testing.assert_allclose(sim_forces[0], force, **TOLERANCES)

        numpy.testing.assert_allclose(sim_forces[1],  [-force[0], -force[1], -force[2]], **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[0], torques[0], **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[1], torques[1], **TOLERANCES)
        

# Move this to validation
# @pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
#                          patch_test_parameters)
# def test_energy_drift(patchy_snapshot_factory, simulation_factory,
#                       patch_cls, patch_args, params, positions, orientations, force, energy, torques):
