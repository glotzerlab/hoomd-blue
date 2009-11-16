# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_random_tests (unittest.TestCase):
    def setUp(self):
        print
    
    # tests basic creation of the random initializer
    def test(self):
        init.create_random(N=100, phi_p=0.05);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
    
    # tests creation with a few more arugments specified
    def test_moreargs(self):
        init.create_random(name="B", min_dist=0.1, N=100, phi_p=0.05);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        
    # checks for an error if initialized twice
    def test_inittwice(self):
        init.create_random(N=100, phi_p=0.05);
        self.assertRaises(RuntimeError, init.create_random, N=100, phi_p=0.05);
    
    def tearDown(self):
        globals._clear();
        
# unit tests for init.read_xml
class init_read_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        f = open("test.xml", "w");
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<hoomd_xml version="1.0">
<configuration time_step="0">
<box units="sigma"  lx="8" ly="8" lz="8"/>
<position units="sigma">
-1 2 3
2 1 -3
3 -2 1
</position>
<type>
A B C
</type>
</configuration>
</hoomd_xml>
''');
        

    # tests basic creation of the random initializer
    def test(self):
        init.read_xml('test.xml');
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getN(), 3);
    
    # tests creation with a few more arugments specified
    def test_moreargs(self):
        init.read_xml('test.xml', time_step=100);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getN(), 3);
        
    # checks for an error if initialized twice
    def test_inittwice(self):
        init.read_xml('test.xml');
        self.assertRaises(RuntimeError, init.read_xml, 'test.xml');
    
    def tearDown(self):
        os.remove("test.xml");
        globals._clear();

# unit tests for init.create_random_polymers
class init_create_random_polymer_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
    
    # tests basic creation of the random initializer
    def test(self):
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
    
    # checks for an error if initialized twice
    def test_create_random_inittwice(self):
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assertRaises(RuntimeError, 
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=self.separation);
    
    # checks that invalid arguments are detected
    def test_bad_polymers(self):
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymer1,
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=5,
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers="polymers",
                          separation=self.separation);
        
        bad_polymer1 = dict(bond_len=1.2, bond="linear", count=10)
        bad_polymer2 = dict(type=['B']*4, bond="linear", count=10)
        bad_polymer3 = dict(bond_len=1.2, type=['B']*4, count=10)
        bad_polymer4 = dict(bond_len=1.2, type=['B']*4, bond="linear")
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer1],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer2],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer3],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer4],
                          separation=self.separation);
        
    def test_bad_separation(self):
        bad_separation1 = dict(A=0.35)
        bad_separation2 = dict(B=0.35)
        bad_separation3 = dict(C=0.35)
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation1);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation2);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation3);
        
    def tearDown(self):
        globals._clear();

# unit tests for init.reset
class init_reset_tests (unittest.TestCase):
    def setUp(self):
        print
    
    # tests basic creation of the random initializer
    def test_works(self):
        init.create_random(N=100, phi_p=0.05);
        init.reset()
    
    # tests creation with a few more arugments specified
    def test_error(self):
        init.create_random(N=100, phi_p=0.05);
        lj = pair.lj(r_cut=3.0)
        self.assertRaises(RuntimeError, init.reset);
        
# unit tests for analyze.imd
#class analyze_imd_tests (unittest.TestCase):
#    def setUp(self):
#        print
#        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the analyzer
#    def test(self):
#        analyze.imd(port = 12345, period = 100);
#        run(100);
    
    # test enable/disable
#    def test_enable_disable(self):
#        ana = analyze.imd(port = 12346, period = 100);
#        ana.disable();
#        self.assert_(not ana.enabled);
#        ana.disable();
#        self.assert_(not ana.enabled);
#        ana.enable();
#        self.assert_(ana.enabled);
#        ana.enable();
#        self.assert_(ana.enabled);
        
    # test set_period
#    def test_set_period(self):
#        ana = analyze.imd(port = 12347, period = 100);
#        ana.set_period(10);
#        ana.disable();
#        self.assertEqual(10, ana.prev_period);
#        ana.set_period(50);
#        self.assertEqual(50, ana.prev_period);
#        ana.enable();
    
#    def tearDown(self):
#        globals._clear();

# unit tests for analyze.log
class analyze_log_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the analyzer
    def test(self):
        analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename="test.log");
        run(100);
    
    # test set_params
    def test_set_params(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename="test.log");
        ana.set_params(quantities = ['test1']);
        run(100);
        ana.set_params(delimiter = ' ');
        run(100);
        ana.set_params(quantities = ['test2', 'test3'], delimiter=',')
        run(100);

    # test variable period
    def test_variable(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = lambda n: n*10, filename="test.log");
        run(100);        
    
    def tearDown(self):
        globals._clear();
        os.remove("test.log");
        
# unit tests for analyze.msd
class analyze_msd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the analyzer
    def test(self):
        analyze.msd(period = 10, filename="test.log", groups=[group.all()]);
        run(100);

    # test variable period
    def test_variable(self):
        analyze.msd(period = lambda n: n*10, filename="test.log", groups=[group.all()]);
        run(100);
    
    # test error if no groups defined
    def test_no_gropus(self):
        self.assertRaises(RuntimeError, analyze.msd, period=10, filename="test.log", groups=[]);
    
    # test set_params
    def test_set_params(self):
        ana = analyze.msd(period = 10, filename="test.log", groups=[group.all()]);
        ana.set_params(delimiter = ' ');
        run(100);
    
    def tearDown(self):
        globals._clear();
        os.remove("test.log");
        
# unit tests for dump.xml
class dmp_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.xml(filename="dump_xml", period=100);

    # test variable period
    def test_variable(self):
        dump.xml(filename="dump_xml", period=lambda n: n*100);
    
    # test set_params
    def test_set_params(self):
        xml = dump.xml(filename="dump_xml", period=100);
        xml.set_params(position=True);
        xml.set_params(velocity=True);
        xml.set_params(mass=False);
        xml.set_params(diameter=False);
        xml.set_params(type=True);
        xml.set_params(wall=True);
        xml.set_params(bond=True);
        xml.set_params(image=True);
        xml.set_params(all=True);
    
    def tearDown(self):
        globals._clear();

# unit tests for dump.mol2
class dmp_mol2_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.mol2(filename="dump_mol2", period=100);
        run(101)
        os.remove("dump_mol2.0000000000.mol2")
    
    # tests variable periods
    def test_variable(self):
        dump.mol2(filename="dump_mol2", period=lambda n: n*100);
        run(100);
        os.remove("dump_mol2.0000000000.mol2")
    
    def tearDown(self):
        globals._clear();
        
# unit tests for dump.pdb
class dmp_pdb_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.pdb(filename="dump_pdb", period=100);
        run(101)
        os.remove("dump_pdb.0000000000.pdb")
    
    # tests variable periods
    def test_variable(self):
        dump.pdb(filename="dump_pdb", period=lambda n: n*100);
        run(101);
        os.remove("dump_pdb.0000000000.pdb")
    
    def tearDown(self):
        globals._clear();


# unit tests for dump.dcd
class dmp_dcd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.dcd(filename="dump_dcd", period=100);
        run(100)
        os.remove('dump_dcd')
            
    # tests variable periods
    def test_variable(self):
        dump.dcd(filename="dump_dcd", period=lambda n: n*100);
        run(100)
        os.remove('dump_dcd')
            
    # test disable/enable
    def test_enable_disable(self):
        dcd = dump.dcd(filename="dump_dcd", period=100);
        dcd.disable()
        self.assertRaises(RuntimeError, dcd.enable)

    # test set_period
    def test_set_period(self):
        dcd = dump.dcd(filename="dump_dcd", period=100);
        self.assertRaises(RuntimeError, dcd.set_period, 10)    
    
    def tearDown(self):
        globals._clear();
        
# unit tests for integrate.nvt
class integrate_nvt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nvt(all, T=1.2, tau=0.5);
        run(100);
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        nvt = integrate.nvt(all, T=1.2, tau=0.5);
        nvt.set_params(T=1.3);
        nvt.set_params(tau=0.6);
    
    def tearDown(self):
        globals._clear();

# unit tests for integrate.bdnvt
class integrate_bdnvt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        
    # tests basic creation of the integration method
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        bd = integrate.bdnvt(all, T=1.2, limit=0.1, seed=52);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2, limit=0.1);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2, gamma_diam=True);
        bd.disable();
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_params(T=1.3);

    # test set_gamma
    def test_set_gamma(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_gamma('A', 0.5);
        bd.set_gamma('B', 1.0);
    
    def tearDown(self):
        globals._clear();
        
        
# unit tests for integrate.npt
class integrate_npt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5);
        run(100);
    
    # test set_params
    def test_set_params(self):
        integrate.mode_standard(dt=0.005);
        all = group.all();
        npt = integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5);
        npt.set_params(T=1.3);
        npt.set_params(tau=0.6);
        npt.set_params(P=0.5);
        npt.set_params(tauP=0.6);
        npt.set_params(partial_scale=True);
        run(100);
    
    def tearDown(self):
        globals._clear();

# unit tests for integrate.nve
class integrate_nve_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
                
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
    
    # tests creation of the method with options
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all, limit=0.01, zero_force=True);
        run(100);
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        mode = integrate.mode_standard(dt=0.005);
        mode.set_params(dt=0.001);
        nve = integrate.nve(all);
        nve.set_params(limit=False);
        nve.set_params(limit=0.1);
        nve.set_params(zero_force=False);
        
    
    def tearDown(self):
        globals._clear();

# pair.nlist testing
class pair_nlist_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        #indirectly create the neighbor list by creating a pair.lj
        pair.lj(r_cut=3.0);
        

    # test set_params
    def test_set_params(self):
        globals.neighbor_list.set_params(r_buff=0.6);
        globals.neighbor_list.set_params(check_period = 20);
    
    # test reset_exclusions
    def test_reset_exclusions_works(self):
        globals.neighbor_list.reset_exclusions(exclusions = ['1-2']);
        globals.neighbor_list.reset_exclusions(exclusions = ['1-3']);
        globals.neighbor_list.reset_exclusions(exclusions = ['1-4']);
        globals.neighbor_list.reset_exclusions(exclusions = ['bond']);
        globals.neighbor_list.reset_exclusions(exclusions = ['angle']);
        globals.neighbor_list.reset_exclusions(exclusions = ['dihedral']);
        globals.neighbor_list.reset_exclusions(exclusions = ['bond', 'angle']);
    
    # test reset_exclusions error messages
    def test_reset_exclusions_nowork(self):
        self.assertRaises(RuntimeError,
                          globals.neighbor_list.reset_exclusions,
                          exclusions = ['bond', 'angle', 'invalid']);
    
    def tearDown(self):
        globals._clear();
    
# pair.lj
class pair_lj_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        lj = pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        lj = pair.lj(r_cut=3.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);
    
    # test set params
    def test_set_params(self):
        lj = pair.lj(r_cut=3.0);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        lj.set_params(mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");
    
    def tearDown(self):
        globals._clear();

# pair.table
class pair_table_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        table = pair.table(width=1000);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        table = pair.table(width=1000);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0);
        self.assertRaises(RuntimeError, table.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        table = pair.table(width=1000);
        self.assertRaises(RuntimeError, table.update_coeffs);
    
    def tearDown(self):
        globals._clear();

# pair.gauss
class pair_gauss_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        gauss.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        gauss = pair.gauss(r_cut=3.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);
    
    # test set params
    def test_set_params(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.set_params(mode="no_shift");
        gauss.set_params(mode="shift");
        self.assertRaises(RuntimeError, gauss.set_params, mode="blah");
    
    def tearDown(self):
        globals._clear();

# pair.cgcmm
class pair_cgcmm_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        cgcmm.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        cgcmm.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        cgcmm.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, cgcmm.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        self.assertRaises(RuntimeError, cgcmm.update_coeffs);
    
    def tearDown(self):
        globals._clear();
        
# tests force.constant
class force_constant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
    
    # test to see that se can create a force.constant
    def test_create(self):
        force.constant(fx=1.0, fy=0.5, fz=0.74);
        
    # test changing the force
    def test_change_force(self):
        const = force.constant(fx=1.0, fy=0.5, fz=0.74);
        const.set_force(fx=1.45, fy=0.25, fz=-0.1);
    
    def tearDown(self):
        globals._clear();

# tests wall.lj
class wall_lj_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
    
    # test to see that se can create a wall.lj
    def test_create(self):
        wall.lj(r_cut=3.0);
        
    # test setting coefficients
    def test_set_coeff(self):
        lj_wall = wall.lj(r_cut=3.0);
        lj_wall.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        lj_wall = wall.lj(r_cut=3.0);
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();
        
# tests bond.harmonic
class bond_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
    
    # test to see that se can create a force.constant
    def test_create(self):
        bond.harmonic();
        
    # test setting coefficients
    def test_set_coeff(self):
        harmonic = bond.harmonic();
        harmonic.set_coeff('polymer', k=1.0, r0=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = bond.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();

# tests angle.harmonic
class angle_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a few angles to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        
        angle_data = globals.system_definition.getAngleData();
        angle_data.addAngle(hoomd.Angle(0, 0, 1, 2));
    
    # test to see that se can create an angle.harmonic
    def test_create(self):
        angle.harmonic();
        
    # test setting coefficients
    def test_set_coeff(self):
        harmonic = angle.harmonic();
        harmonic.set_coeff('angleA', k=1.0, t0=0.78125)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = angle.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();
        
# tests angle.cgcmm
class angle_cgcmm_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a few angles to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        
        angle_data = globals.system_definition.getAngleData();
        angle_data.addAngle(hoomd.Angle(0, 0, 1, 2));
    
    # test to see that se can create an angle.cgcmm
    def test_create(self):
        angle.cgcmm();
        
    # test setting coefficients
    def test_set_coeff(self):
        cgcmm = angle.cgcmm();
        cgcmm.set_coeff('angleA', k=3.0, t0=0.7851, exponents=126, epsilon=1.0, sigma=0.53)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        cgcmm = angle.cgcmm();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();

        
# tests dihedral.harmonic
class dihedral_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a dihedral to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        
        dihedral_data = globals.system_definition.getDihedralData();
        dihedral_data.addDihedral(hoomd.Dihedral(0, 0, 1, 2, 3));
    
    # test to see that se can create an angle.harmonic
    def test_create(self):
        dihedral.harmonic();
        
    # test setting coefficients
    def test_set_coeff(self):
        harmonic = dihedral.harmonic();
        harmonic.set_coeff('dihedralA', k=1.0, d=1, n=4)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = dihedral.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();
        
# tests improper.harmonic
class improper_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a dihedral to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        
        improper_data = globals.system_definition.getImproperData();
        improper_data.addDihedral(hoomd.Dihedral(0, 0, 1, 2, 3));
    
    # test to see that se can create an angle.harmonic
    def test_create(self):
        improper.harmonic();
        
    # test setting coefficients
    def test_set_coeff(self):
        harmonic = improper.harmonic();
        harmonic.set_coeff('dihedralA', k=30.0, chi=1.57)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = improper.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();
        
# tests bond.fene
class bond_fene_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
    
    # test to see that se can create a force.constant
    def test_create(self):
        bond.fene();
        
    # test setting coefficients
    def test_set_coeff(self):
        fene = bond.fene();
        fene.set_coeff('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon=2.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        fene = bond.fene();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        globals._clear();
        
# tests for update.box_resize
class update_box_resize_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the updater
    def test(self):
        update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]))
        run(100);

    # test the setting of more args
    def test_moreargs(self):
        update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]),
                        Ly = variant.linear_interp([(0, 40), (1e6, 80)]))
        run(100);
    
    # test the setting of more args
    def test_evenmoreargs(self):
        update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]),
                        Ly = variant.linear_interp([(0, 40), (1e6, 80)]),
                        Lz = variant.linear_interp([(0, 40), (1e6, 80)]),
                        period=10);
        run(100);
    
    # test set_params
    def test_set_params(self):
        upd = update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]))
        upd.set_params(scale_particles = False);
    
    def tearDown(self):
        globals._clear();


# tests for update.rescale_temp
class update_rescale_temp_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the updater
    def test(self):
        update.rescale_temp(T=1.0)
        run(100);

    # test variable periods
    def test_variable(self):
        update.rescale_temp(T=1.0, period=lambda n: n*10)
        run(100);
    
    # test enable/disable
    def test_enable_disable(self):
        upd = update.rescale_temp(T=1.0)
        upd.disable();
        self.assert_(not upd.enabled);
        upd.disable();
        self.assert_(not upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);
        
    # test set_period
    def test_set_period(self):
        upd = update.rescale_temp(T=1.0)
        upd.set_period(10);
        upd.disable();
        self.assertEqual(10, upd.prev_period);
        upd.set_period(50);
        self.assertEqual(50, upd.prev_period);
        upd.enable();
        
    # test set_params
    def test_set_params(self):
        upd = update.rescale_temp(T=1.0);
        upd.set_params(T=1.2);
    
    def tearDown(self):
        globals._clear();
        
# tests for update.sorter
class update_sorter_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # test set_params
    def test_set_params(self):
        import __main__;
        __main__.sorter.set_params(bin_width=2.0);
    
    def tearDown(self):
        globals._clear();
        
# tests for update.zero_momentum
class update_zero_momentum_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the updater
    def test(self):
        update.zero_momentum()
        run(100);
    
    # test variable periods
    def test_variable(self):
        update.zero_momentum(period = lambda n: n*100);
        run(100);
    
    # test enable/disable
    def test_enable_disable(self):
        upd = update.rescale_temp(T=1.0)
        upd.disable();
        self.assert_(not upd.enabled);
        upd.disable();
        self.assert_(not upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);
        
    # test set_period
    def test_set_period(self):
        upd = update.rescale_temp(T=1.0)
        upd.set_period(10);
        upd.disable();
        self.assertEqual(10, upd.prev_period);
        upd.set_period(50);
        self.assertEqual(50, upd.prev_period);
        upd.enable();
        
    def tearDown(self):
        globals._clear();

# tests for variant types
class variant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests creation of the constant variant
    def test_const(self):
        v = variant._constant(5)
        self.assertEqual(5.0, v.cpp_variant.getValue(0))
        self.assertEqual(5.0, v.cpp_variant.getValue(100000))
        self.assertEqual(5.0, v.cpp_variant.getValue(5000))
        self.assertEqual(5.0, v.cpp_variant.getValue(40))
        self.assertEqual(5.0, v.cpp_variant.getValue(50))

    # tests a simple linear variant
    def test_linear_interp(self):
        v = variant.linear_interp(points = [(0, 10), (100, 20)]);
        self.assertEqual(15.0, v.cpp_variant.getValue(50));
        self.assertEqual(10.0, v.cpp_variant.getValue(0));
        self.assertEqual(20.0, v.cpp_variant.getValue(100));
        self.assertEqual(20.0, v.cpp_variant.getValue(1000));

    # test the setup helper
    def setup_variant_input_test(self):
        v = variant._setup_variant_input(55);
        self.assertEqual(55.0, v.cpp_variant.getValue(0));

        v = variant._setup_variant_input(variant.linear_interp(points = [(0, 10), (100, 20)]));
        self.assertEqual(15.0, v.cpp_variant.getValue(50));

    def tearDown(self):
        globals._clear();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

