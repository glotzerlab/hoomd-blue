# $Id:$
# $URL:$

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
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=self.polymers, separation=self.separation);
	
	# checks that invalid arguments are detected
	def test_bad_polymers(self):
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=[], separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=self.polymer1, separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=5, separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers="polymers", separation=self.separation);
		
		bad_polymer1 = dict(bond_len=1.2, bond="linear", count=10)
		bad_polymer2 = dict(type=['B']*4, bond="linear", count=10)
		bad_polymer3 = dict(bond_len=1.2, type=['B']*4, count=10)
		bad_polymer4 = dict(bond_len=1.2, type=['B']*4, bond="linear")
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=[bad_polymer1], separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=[bad_polymer2], separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=[bad_polymer3], separation=self.separation);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=[bad_polymer4], separation=self.separation);
		
	def test_bad_separation(self):
		bad_separation1 = dict(A=0.35)
		bad_separation2 = dict(B=0.35)
		bad_separation3 = dict(C=0.35)
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=self.polymers, separation=bad_separation1);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=self.polymers, separation=bad_separation2);
		self.assertRaises(RuntimeError, init.create_random_polymers, box=self.box, polymers=self.polymers, separation=bad_separation3);
		
	def tearDown(self):
		globals._clear();

# unit tests for analyze.imd
#class analyze_imd_tests (unittest.TestCase):
#	def setUp(self):
#		print
#		init.create_random(N=100, phi_p=0.05);

	# tests basic creation of the analyzer
#	def test(self):
#		analyze.imd(port = 12345, period = 100);
#		run(100);
	
	# test enable/disable
#	def test_enable_disable(self):
#		ana = analyze.imd(port = 12346, period = 100);
#		ana.disable();
#		self.assert_(not ana.enabled);
#		ana.disable();
#		self.assert_(not ana.enabled);
#		ana.enable();
#		self.assert_(ana.enabled);
#		ana.enable();
#		self.assert_(ana.enabled);
		
	# test set_period
#	def test_set_period(self):
#		ana = analyze.imd(port = 12347, period = 100);
#		ana.set_period(10);
#		ana.disable();
#		self.assertEqual(10, ana.prev_period);
#		ana.set_period(50);
#		self.assertEqual(50, ana.prev_period);
#		ana.enable();
	
#	def tearDown(self):
#		globals._clear();

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
	
	# tests variable periods
	def test_variable(self):
		dump.mol2(filename="dump_mol2", period=lambda n: n*100);
	
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
	
	# tests variable periods
	def test_variable(self):
		dump.dcd(filename="dump_dcd", period=lambda n: n*100);
	
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
		integrate.nvt(dt=0.005, T=1.2, tau=0.5);
		run(100);
	
	# test set_params
	def test_set_params(self):
		nvt = integrate.nvt(dt=0.005, T=1.2, tau=0.5);
		nvt.set_params(dt=0.001);
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
		
	# tests basic creation of the dump
	def test(self):
		integrate.bdnvt(dt=0.005, T=1.2, limit=0.1, seed=52);
		run(100);
		integrate.bdnvt(dt=0.005, T=1.2, limit=0.1);
		integrate.bdnvt(dt=0.005, T=1.2);
	
	# test set_params
	def test_set_params(self):
		bd = integrate.bdnvt(dt=0.005, T=1.2);
		bd.set_params(dt=0.001);
		bd.set_params(T=1.3);

	# test set_gamma
	def test_set_gamma(self):
		bd = integrate.bdnvt(dt=0.005, T=1.2);
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
		integrate.npt(dt=0.005, T=1.2, tau=0.5, P=1.0, tauP=0.5);
		run(100);
	
	# test set_params
	def test_set_params(self):
		npt = integrate.npt(dt=0.005, T=1.2, tau=0.5, P=1.0, tauP=0.5);
		npt.set_params(dt=0.001);
		npt.set_params(T=1.3);
		npt.set_params(tau=0.6);
		npt.set_params(P=0.5);
		npt.set_params(tauP=0.6);
	
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
		integrate.nve(dt=0.005);
		run(100);
	
	# test set_params
	def test_set_params(self):
		nve = integrate.nve(dt=0.005);
		nve.set_params(dt=0.001);
	
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
		lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0);
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
		lj.set_params(fraction=0.5);
		lj.set_params(mode="no_shift");
		lj.set_params(mode="shift");
		lj.set_params(mode="xplor");
		self.assertRaises(RuntimeError, lj.set_params, mode="blah");
	
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
		gauss = pair.lj(r_cut=3.0);
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
		integrate.nve(dt=0.005);
		run(100);
		
	# test coefficient not set checking
	def test_set_coeff_fail(self):
		lj_wall = wall.lj(r_cut=3.0);
		integrate.nve(dt=0.005);
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
		integrate.nve(dt=0.005);
		run(100);
		
	# test coefficient not set checking
	def test_set_coeff_fail(self):
		harmonic = bond.harmonic();
		integrate.nve(dt=0.005);
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
		integrate.nve(dt=0.005);
		run(100);
		
	# test coefficient not set checking
	def test_set_coeff_fail(self):
		fene = bond.fene();
		integrate.nve(dt=0.005);
		self.assertRaises(RuntimeError, run, 100);
	
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
