// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _ANALYZER_SDF_H_
#define _ANALYZER_SDF_H_

/*! \file AnalyzerSDF.h
    \brief Declaration of AnalyzerSDF
*/


#include "hoomd/Analyzer.h"
#include "hoomd/Filesystem.h"
#include "IntegratorHPMCMono.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

namespace detail
{

//! Local helper function to test overlap of two particles with scale
template < class Shape >
bool test_scaled_overlap(const vec3<Scalar>& r_ij,
                         const quat<Scalar>& orientation_i,
                         const quat<Scalar>& orientation_j,
                         const typename Shape::param_type& params_i,
                         const typename Shape::param_type& params_j,
                         Scalar lambda)
    {
    // need a dummy error counter
    unsigned int dummy = 0;

    // instantiate the shapes
    Shape shape_i(orientation_i, params_i);
    Shape shape_j(orientation_j, params_j);

    vec3<Scalar> r_ij_scaled = r_ij * (Scalar(1.0) - lambda);
    return check_circumsphere_overlap(r_ij_scaled, shape_i, shape_j) && test_overlap(r_ij_scaled, shape_i, shape_j, dummy);
    }

}

//! SDF analysis
/*! **Overview** <br>

    AnalyzeSDF computes \f$ s(\lambda)/N \f$, averages it over a defined number of configurations, and writes it out to
    a file. \f$ s(\lambda) \f$ is a distribution function like *g(r)*, except that \f$\lambda\f$ is the smallest scale factor
    that causes a particle to just just touch the closest of its neighbors. The output of AnalyzeSDF \f$ s(\lambda)/N \f$
    is raw data, with the only normalization being a division by the number of particles to
    compute a count of the average number overlapping particle pairs in a given \f$\lambda\f$ histogram bin.

    \f$ s(\lambda)/N \f$ extrapolated out to \f$ \lambda=0 \f$ is directly related by a scale factor to the pressure in
    an NVT system of hard particles. Performing this extrapolation only needs data very near zero
    (e.g. up to 0.02 for disks).

    AnalyzeSDF is implemented as an analyzer in HPMC (and not in freud) due to the need for high performance evaluation
    at very small periods. A future module from freud does not conflict with this code, as that would be more general
    and extend to larger values of \f$\lambda\f$, whereas this code is optimized only for small values.

    \b Computing \f$ \lambda \f$ <br>

    In the initial version of the code, a completely general way of computing *\f$ \lambda \f$* is implemented.
    It uses a binary search tree and the existing test_overlap code to find which bin a given pair of particles sits in.
    Future versions of the code may use shape specific data to compute *\f$ \lambda \f$* directly.

    Outside of that AnalyzerSDF is a pretty basic histogramming code. The only other notable features in the design
    are:
      - Suitably chosen navg results in the average being written out just before a restart - enabling full restart
        capabilities.
      - Fully uses the MPI domain decomposition to compute the SDF fast in large jobs.

    \b Storage <br>

    Bin counts are stored in a basic std::vector<unsigned int>. Bin 0 counts \f$ \lambda \f$ values from
    \f$ \lambda [0,d\lambda) \f$, bin n counts from \f$ [d\lambda \cdot n,d\lambda \cdot (n+1)) \f$. Up to a value of
    *lmax* for the right hand side of the last bin (a total of *lmax/dl* bins)

    The position of the bin centers needs to be taken into account carefully (see the MPMC paper), but AnalyzerSDF
    currently doesn't do that. It just writes out the raw histogram counts.

    \b File format <br>

    The initial implementation has a dirt simple file format. Simply output the timestep and then all of the normalized
    bin counts after that on a single line. This is suitable for processing and plotting in matlab or python. Once
    the code is tested completely, final use cases may dictate a different format. For now, we need the full information
    for testing.

    \b Connection to an integrator <br>

    In MPI, the ghost layer width needs to be increased slightly. This is done by passing the MC integrator into the
    analyzer which then calls a method to set up the extra ghost width. This connection is also used to get the maximum
    particle diameter for an input into the cell list size.

    \ingroup hpmc_analyzers
*/
template < class Shape >
class AnalyzerSDF : public Analyzer
    {
    public:
        //! Shape parameter time (shorthand)
        typedef typename Shape::param_type param_type;

        //! Constructor
        AnalyzerSDF(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                    double lmax,
                    double dl,
                    unsigned int navg,
                    const std::string& fname,
                    bool overwrite);

        //! Destructor
        virtual ~AnalyzerSDF()
            {
            m_exec_conf->msg->notice(5) << "Destroying AnalyzerSDF" << std::endl;
            }


        //! Analyze the system configuration on the given time step
        virtual void analyze(unsigned int timestep);

    protected:
        std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc; //!< The integrator
        double m_lmax;                          //!< Maximum lambda value
        double m_dl;                            //!< Histogram step size
        unsigned int m_navg;                    //!< Number of samples to average before writing out to the file
        std::string m_filename;                 //!< File name to write out to

        std::ofstream m_file;                   //!< Output file
        bool m_is_initialized;                  //!< Bool indicating if we have initialized the file yet
        bool m_appending;                       //!< Flag indicating this file is being appended to
        std::vector<unsigned int> m_hist;       //!< Raw histogram data

        unsigned int m_iavg;                    //!< Current count of the number of steps averaged
        Scalar m_last_max_diam;                 //!< Last recorded maximum diameter

        //! Helper function to open the output file
        void openOutputFile();

        //! Write current histogram to the file
        void writeOutput(unsigned int timestep);

        //! Zero the histogram counts
        void zeroHistogram();

        //! Add to histogram counts
        void countHistogram(unsigned int timestep);

        //! Determine the s bin of a given particle pair
        int computeBin(const vec3<Scalar>& r_ij,
                       const quat<Scalar>& orientation_i,
                       const quat<Scalar>& orientation_j,
                       const typename Shape::param_type& params_i,
                       const typename Shape::param_type& params_j);
    };


/*! \param sysdef System definition
    \param mc The MC integrator
    \param lmax Right hand side of the last histogram bin
    \param dl Bin size
    \param navg Number of samples to average before writing to the file
    \param fname File name to write to
    \param overwrite Set to true to overwrite instead of append to the file

    Construct the SDF analyzer and initialize histogram memory to 0
*/
template < class Shape >
AnalyzerSDF<Shape>::AnalyzerSDF(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                                double lmax,
                                double dl,
                                unsigned int navg,
                                const std::string& fname,
                                bool overwrite)
    : Analyzer(sysdef), m_mc(mc), m_lmax(lmax), m_dl(dl), m_navg(navg), m_filename(fname), m_is_initialized(false),
      m_appending(!overwrite), m_iavg(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing AnalyzerSDF: " << fname << " " << lmax << " " << dl << " " << navg << std::endl;

    m_hist.resize(lmax / dl);
    zeroHistogram();

    Scalar max_diam = m_mc->getMaxCoreDiameter();
    m_last_max_diam = max_diam;
    Scalar extra = lmax * max_diam;
    m_mc->setExtraGhostWidth(extra);
    }

/*! \param timestep Current time step

    Main analysis driver. Manages the state and calls the appropriate helper functions.
*/
template < class Shape >
void AnalyzerSDF<Shape>::analyze(unsigned int timestep)
    {
    m_exec_conf->msg->notice(8) << "Analyzing sdf at step " << timestep << std::endl;

    // kludge to update the max diameter dynamically if it changes
    Scalar max_diam = m_mc->getMaxCoreDiameter();
    if (max_diam != m_last_max_diam)
        {
        m_last_max_diam = max_diam;
        Scalar extra = m_lmax * max_diam;
        m_mc->setExtraGhostWidth(extra);

        // this forces an extra communication of ghosts, but only when the maximum diameter changes
        m_mc->communicate(false);
        }

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "SDF");

    // open output file for writing
    if (!m_is_initialized)
        {
        openOutputFile();
        m_is_initialized = true;
        }

    countHistogram(timestep);
    m_iavg++;

    if (m_iavg == m_navg)
        {
        writeOutput(timestep);
        m_iavg = 0;
        zeroHistogram();
        }

    if (this->m_prof) this->m_prof->pop();
    }

template < class Shape >
void AnalyzerSDF<Shape>::openOutputFile()
    {
    // open the output file for writing or appending, based on the existence of the file and any user options

#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_comm)
        if (! m_exec_conf->isRoot())
            return;
#endif
    // open the file
    if (filesystem::exists(m_filename) && m_appending)
        {
        m_exec_conf->msg->notice(3) << "analyze.sdf: Appending to existing data file \"" << m_filename << "\"" << std::endl;
        m_file.open(m_filename.c_str(), std::ios_base::in | std::ios_base::out | std::ios_base::ate);
        }
    else
        {
        m_exec_conf->msg->notice(3) << "analyze.sdf: Creating new data file \"" << m_filename << "\"" << std::endl;
        m_file.open(m_filename.c_str(), std::ios_base::out);
        m_appending = false;
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.sdf: Error opening data file " << m_filename << std::endl;
        throw std::runtime_error("Error initializing analyze.sdf");
        }
    }

/*! \param timestep Current time step

    Write the output to the file.
*/
template < class Shape >
void AnalyzerSDF<Shape>::writeOutput(unsigned int timestep)
    {
    std::vector<unsigned int> hist_total(m_hist);

    // in MPI, we need to total up all of the histogram bins from all nodes to the root node
#ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Reduce(&m_hist[0], &hist_total[0], m_hist.size(), MPI_UNSIGNED, MPI_SUM, 0, m_exec_conf->getMPICommunicator());

        // then all ranks but root stop here
        if (! m_exec_conf->isRoot())
            return;
        }
#endif

    // write out the normalized histogram bin values on one line
    m_file << std::setprecision(16) << timestep << " ";
    for (unsigned int i = 0; i < m_hist.size(); i++)
        {
        m_file << double(hist_total[i]) / double(m_navg*m_pdata->getNGlobal()*m_dl) << " ";
        }

    m_file << std::endl;

    // check that the file handle is still OK
    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.sdf: I/O error while writing data file" << std::endl;
        throw std::runtime_error("Error writing data file");
        }
    }

template < class Shape >
void AnalyzerSDF<Shape>::zeroHistogram()
    {
    // Zero the histogram
    for (unsigned int i = 0; i < m_hist.size(); i++)
        {
        m_hist[i] = 0;
        }
    }

/*! \param timestep current timestep

    countHistogram() loops through all particle pairs *i,j* where *i* is on the local rank, computes the bin in which
    that pair should be and adds 1 to the bin. countHistogram() can be called multiple times to increment the counters
    for averaging, and it operates without any communication
      - The integrator performs the ghost exchange (with the ghost width extra that we add)
      - Only on writeOutput() do we need to sum the per-rank histograms into a global histogram
*/
template < class Shape >
void AnalyzerSDF<Shape>::countHistogram(unsigned int timestep)
    {
    // update the aabb tree
    const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();
    // update the image list
    const std::vector<vec3<Scalar> >&image_list = m_mc->updateImageList();

    Scalar extra_width = m_lmax / (1 - m_lmax) * m_mc->getMaxCoreDiameter();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

    const std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();

    // loop through N particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        int min_bin = m_hist.size();

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        Shape shape_i(quat<Scalar>(orientation_i), params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // construct the AABB around the particle's circumsphere
        // pad with enough extra width so that when scaled by lmax, found particles might touch
        detail::AABB aabb_i_local(vec3<Scalar>(0,0,0), shape_i.getCircumsphereDiameter()/Scalar(2) + extra_width);

        const unsigned int n_images = image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;


                            int bin = computeBin(r_ij,
                                                 quat<Scalar>(orientation_i),
                                                 quat<Scalar>(orientation_j),
                                                 params[__scalar_as_int(postype_i.w)],
                                                 params[__scalar_as_int(postype_j.w)]);

                            if (bin >= 0)
                                min_bin = std::min(min_bin, bin);
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                    }
                } // end loop over AABB nodes
            } // end loop over images

        // record the minimum bin
        if ((unsigned int)min_bin < m_hist.size())
            m_hist[min_bin]++;

        } // end loop over all particles
    }

/*! \param r_ij Vector pointing from particle i to j (already wrapped into the box)
    \param orientation_i Orientation of the particle i
    \param orientation_j Orientation of particle j
    \param params_i Parameters for particle i
    \param params_j Parameters for particle j

    \returns s bin index

    In the first general version, computeBin uses a binary search tree to determine
    the bin. In this way, only a test_overlap method is needed, no extra math. The
    binary search works by first ensuring that the particle does not overlap at the
    left boundary and does overlap a the right. Then it picks a new point halfway between
    the left and right, ensuring that the same assumption holds. Once right=left+1, the
    correct bin has been found.
*/
template < class Shape >
int AnalyzerSDF<Shape>:: computeBin(const vec3<Scalar>& r_ij,
                             const quat<Scalar>& orientation_i,
                             const quat<Scalar>& orientation_j,
                             const typename Shape::param_type& params_i,
                             const typename Shape::param_type& params_j)
    {
    unsigned int L=0;
    unsigned int R=m_hist.size();

    // if the particles already overlap a the left boundary, return an out of range value
    if (detail::test_scaled_overlap<Shape>(r_ij, orientation_i, orientation_j, params_i, params_j, L*m_dl))
        return -1;

    // if the particles do not overlap a the right boundary, return an out of range value
    if (!detail::test_scaled_overlap<Shape>(r_ij, orientation_i, orientation_j, params_i, params_j, R*m_dl))
        return m_hist.size();

    // progressively narrow the search window by halves
    do
        {
        unsigned int m = (L+R)/2;

        if (detail::test_scaled_overlap<Shape>(r_ij, orientation_i, orientation_j, params_i, params_j, m*m_dl))
            R = m;
        else
            L = m;
        } while ((R-L) > 1);

    return L;
    }

//! Export the AnalyzerSDF class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of AnalyzerSDF<Shape> will be exported
*/
template < class Shape > void export_AnalyzerSDF(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< AnalyzerSDF<Shape>, std::shared_ptr< AnalyzerSDF<Shape> > >(m, name.c_str(), pybind11::base<Analyzer>())
          .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr< IntegratorHPMCMono<Shape> >, double, double, unsigned int, const std::string&, bool>())
          ;
    }

} // end namespace hpmc

#endif // _ANALYZER_SDF_H_
