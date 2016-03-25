## \package hpmc.integrate
# \brief HPMC integration modes

from . import _hpmc
from . import integrate

from hoomd_script.analyze import _analyzer
from hoomd_script import util, globals, init
import hoomd

## Compute the scale distribution function
#
# analyze.sdf computes a distribution function of scale parameters \f$ x \f$ . For each particle, it finds the smallest
# scale factor \f$ 1+x \f$ that would cause the particle to touch one of its neighbors and records that in the histogram
# \f$ s(x) \f$. The histogram is discreet and \f$ s(x_i) = s[i] \f$ where \f$ x_i = i \cdot dx + dx/2 \f$.
#
# In an NVT simulation, the extrapolation of \f$ s(x) \f$ to \f$ x = 0 \f$, \f$ s(0+) \f$ is related to the pressure.
# \f[
#     \frac{P}{kT} = \rho \left(1 + \frac{s(0+)}{2d} \right)
# \f], where \f$d\f$ is the dimensionality of the system.
#
# Extrapolating \f$ s(0+) \f$ is not trivial. Here are some suggested parameters, but they may not work in all cases.
#   - *xmax* = 0.02
#   - *dx* = 1e-4
#   - Polynomial curve fit of degree 5.
#
# In systems near densest packings, *dx*=1e-5 may be needed along with either a smaller xmax or a smaller region to fit.
# A good rule of thumb might be to fit a region where numpy.sum(s[0:n]*dx) ~ 0.5 - but this needs further testing to
# confirm.
#
# The current version of analyze.sdf is in beta. It averages *navg* histograms together before writing them out to a
# text file in a plain format: "timestep bin_0 bin_1 bin_2 .... bin_n". Future versions may upgrade to using HDF5 files
# and may even do the extrapolation to pressure automatically.
#
# analyze.sdf works well with restartable jobs. Just ensure that navg*period is a fraction 1/k of the restart period.
# Then analyze.sdf will have written the final output to its file just before the restart gets written. The new data
# needed for the next line of values is entirely collected after the restart.
#
# analyze.sdf does not compute correct pressures for simulations with concave particles.
#
# \par Numpy extrapolation code
#
# \code
# def extrapolate(s, dx, xmax, degree=5):
#   # determine the number of values to fit
#   n_fit = int(math.ceil(xmax/dx));
#   s_fit = s[0:n_fit];
#   # construct the x coordinates
#   x_fit = numpy.arange(0,xmax,dx)
#   x_fit += dx/2;
#   # perform the fit and extrapolation
#   p = numpy.polyfit(x_fit, s_fit, degree);
#   return numpy.polyval(p, 0.0);
# \endcode
class sdf(_analyzer):

    ## Specifies the SDF analysis to perform
    # \param mc MC integrator (don't specify a new integrator later, sdf will continue to use the old one)
    # \param filename Output file name
    # \param xmax Maximum *x* value at the right hand side of the rightmost bin
    # \param dx Bin width
    # \param navg Number of times to average before writing the histogram to the file
    # \param period Number of timesteps between histogram evaluations
    # \param overwrite Set to True to overwrite \a filename instead of appending to it
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236)
    # analyze.sdf(mc=mc, filename='sdf.dat', xmax=0.02, dx=1e-4, navg=100, period=100)
    # analyze.sdf(mc=mc, filename='sdf.dat', xmax=0.002, dx=1e-5, navg=100, period=100)
    # ~~~~~~~~~~~~~
    def __init__(self, mc, filename, xmax, dx, navg, period, overwrite=False, phase=-1):
        util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        # create the c++ mirror class
        cls = None;
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.AnalyzerSDFSphere;
        elif isinstance(mc, integrate.convex_polygon):
            cls = _hpmc.AnalyzerSDFConvexPolygon;
        elif isinstance(mc, integrate.simple_polygon):
            cls = _hpmc.AnalyzerSDFSimplePolygon;
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = integrate._get_sized_entry('AnalyzerSDFConvexPolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = integrate._get_sized_entry('AnalyzerSDFSpheropolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.ellipsoid):
            cls = _hpmc.AnalyzerSDFEllipsoid;
        elif isinstance(mc, integrate.convex_spheropolygon):
            cls =_hpmc.AnalyzerSDFSpheropolygon;
        else:
            globals.msg.error("analyze.sdf: Unsupported integrator.\n");
            raise runtime_error("Error initializing analyze.sdf");

        self.cpp_analyzer = cls(globals.system_definition,
                                mc.cpp_integrator,
                                xmax,
                                dx,
                                navg,
                                filename,
                                overwrite);

        self.setupAnalyzer(period, phase);

        # meta data
        self.filename = filename
        self.xmax = xmax
        self.dx = dx
        self.navg = navg
        self.period = period
        self.overwrite = overwrite
        self.metadata_fields = ['filename', 'xmax', 'dx', 'navg', 'period', 'overwrite']
