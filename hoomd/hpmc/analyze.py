# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Compute properties of hard particle configurations.
"""

from . import _hpmc
from . import integrate

from hoomd.analyze import _analyzer
import hoomd

class sdf(_analyzer):
    R""" Compute the scale distribution function.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        filename (str): Output file name.
        xmax (float): Maximum *x* value at the right hand side of the rightmost bin (distance units).
        dx (float): Bin width (distance units).
        navg (int): Number of times to average before writing the histogram to the file.
        period (int): Number of timesteps between histogram evaluations.
        overwrite (bool): Set to True to overwrite *filename* instead of appending to it.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.

    :py:class:`sdf` computes a distribution function of scale parameters :math:`x`. For each particle, it finds the smallest
    scale factor :math:`1+x` that would cause the particle to touch one of its neighbors and records that in the histogram
    :math:`s(x)`. The histogram is discrete and :math:`s(x_i) = s[i]` where :math:`x_i = i \cdot dx + dx/2`.

    In an NVT simulation, the extrapolation of :math:`s(x)` to :math:`x = 0`, :math:`s(0+)` is related to the pressure.

    .. math::
        \frac{P}{kT} = \rho \left(1 + \frac{s(0+)}{2d} \right)

    where :math:`d` is the dimensionality of the system and :math:`\rho` is the number density.

    Extrapolating :math:`s(0+)` is not trivial. Here are some suggested parameters, but they may not work in all cases.

      * *xmax* = 0.02
      * *dx* = 1e-4
      * Polynomial curve fit of degree 5.

    In systems near densest packings, ``dx=1e-5`` may be needed along with either a smaller xmax or a smaller region to fit.
    A good rule of thumb might be to fit a region where ``numpy.sum(s[0:n]*dx)`` ~ 0.5 - but this needs further testing to
    confirm.

    :py:class:`sdf` averages *navg* histograms together before writing them out to a
    text file in a plain format: "timestep bin_0 bin_1 bin_2 .... bin_n".

    :py:class:`sdf` works well with restartable jobs. Ensure that ``navg*period`` is an integer fraction :math:`1/k` of the
    restart period. Then :py:class:`sdf` will have written the final output to its file just before the restart gets
    written. The new data needed for the next line of values is entirely collected after the restart.

    Warning:
        :py:class:`sdf` does not compute correct pressures for simulations with concave particles.

    Numpy extrapolation code::

        def extrapolate(s, dx, xmax, degree=5):
          # determine the number of values to fit
          n_fit = int(math.ceil(xmax/dx));
          s_fit = s[0:n_fit];
          # construct the x coordinates
          x_fit = numpy.arange(0,xmax,dx)
          x_fit += dx/2;
          # perform the fit and extrapolation
          p = numpy.polyfit(x_fit, s_fit, degree);
          return numpy.polyval(p, 0.0);

    Examples::

        mc = hpmc.integrate.sphere(seed=415236)
        analyze.sdf(mc=mc, filename='sdf.dat', xmax=0.02, dx=1e-4, navg=100, period=100)
        analyze.sdf(mc=mc, filename='sdf.dat', xmax=0.002, dx=1e-5, navg=100, period=100)
    """
    def __init__(self, mc, filename, xmax, dx, navg, period, overwrite=False, phase=0):
        hoomd.util.print_status_line();

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
            cls = _hpmc.AnalyzerSDFConvexPolyhedron;
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = _hpmc.AnalyzerSDFSpheropolyhedron;
        elif isinstance(mc, integrate.ellipsoid):
            cls = _hpmc.AnalyzerSDFEllipsoid;
        elif isinstance(mc, integrate.convex_spheropolygon):
            cls =_hpmc.AnalyzerSDFSpheropolygon;
        else:
            hoomd.context.msg.error("analyze.sdf: Unsupported integrator.\n");
            raise runtime_error("Error initializing analyze.sdf");

        self.cpp_analyzer = cls(hoomd.context.current.system_definition,
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
