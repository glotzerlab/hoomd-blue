// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COMPUTE_SDF__H__
#define __COMPUTE_SDF__H__

#include "hoomd/Autotuner.h"
#include "hoomd/CellList.h"
#include "hoomd/Compute.h"

#include "HPMCCounters.h"
#include "HPMCPrecisionSetup.h"
#include "IntegratorHPMCMono.h"
#include "hoomd/RNGIdentifiers.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

/*! \file ComputeSDF.h
    \brief Defines the template class for an sdf compute
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Local helper function to test overlap of two particles with scale
template<class Shape>
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
    return check_circumsphere_overlap(r_ij_scaled, shape_i, shape_j)
           && test_overlap(r_ij_scaled, shape_i, shape_j, dummy);
    }

    } // namespace detail

//! SDF analysis
/*! **Overview** <br>

    ComputeSDF computes \f$ s(\lambda)/N \f$. \f$ s(\lambda) \f$ is a distribution function like
    *g(r)*, except that \f$\lambda\f$ is the smallest scale factor that causes a particle to just
    just touch the closest of its neighbors. The output of ComputeSDF \f$ s(\lambda)/N \f$ is raw
    data, with the only normalization being a division by the number of particles to compute a count
    of the average number overlapping particle pairs in a given \f$\lambda\f$ histogram bin.

    \f$ s(\lambda)/N \f$ extrapolated out to \f$ \lambda=0 \f$ is directly related by a scale factor
    to the pressure in an NVT system of hard particles. Performing this extrapolation only needs
    data very near zero (e.g. up to 0.02 for disks).

    ComputeSDF is implemented as a compute in HPMC (and not in freud) due to the need for high
    performance evaluation at very small periods. A future module from freud does not conflict with
    this code, as that would be more general and extend to larger values of \f$\lambda\f$, whereas
    this code is optimized only for small values.

    \b Computing \f$ \lambda \f$ <br>

    In the initial version of the code, a completely general way of computing *\f$ \lambda \f$* is
    implemented. It uses a binary search tree and the existing test_overlap code to find which bin a
    given pair of particles sits in. Future versions of the code may use shape specific data to
    compute *\f$ \lambda \f$* directly.

    Outside of that ComputeSDF is a pretty basic histogramming code. The only other notable feature
    in the design is the full use of the MPI domain decomposition to compute the SDF fast in large
    jobs.

    \b Storage <br>

    Bin counts are stored in a basic std::vector<unsigned int>. Bin 0 counts \f$ \lambda \f$ values
    from \f$ \lambda [0,d\lambda) \f$, bin n counts from \f$ [d\lambda \cdot n,d\lambda \cdot (n+1))
    \f$. Up to a value of *xmax* for the right hand side of the last bin (a total of *xmax/dx* bins)

    The position of the bin centers needs to be taken into account carefully (see the MPMC paper),
    but ComputeSDF currently doesn't do that. It just writes out the raw histogram counts.

    \b Connection to an integrator <br>

    In MPI, the ghost layer width needs to be increased slightly. This is done by passing the MC
    integrator into the compute which then calls a method to set up the extra ghost width. This
    connection is also used to get the maximum particle diameter for an input into the cell list
    size.

    \ingroup hpmc_computes
*/
template<class Shape> class ComputeSDF : public Compute
    {
    public:
    //! Shape parameter time (shorthand)
    typedef typename Shape::param_type param_type;

    //! Construct the integrator
    ComputeSDF(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
               double xmax,
               double dx,
               int mode,
               unsigned int num_random_samples,
               double beta);
    //! Destructor
    virtual ~ComputeSDF() {};

    //! Get the maximum value in the rightmost histogram bin
    double getXMax()
        {
        return m_xmax;
        }

    //! Set the maximum value in the rightmost histogram bin
    //! \param xmax maximum value in the rightmost histogram bin
    void setXMax(double xmax)
        {
        m_xmax = xmax;
        }

    //! Get the histogram bin width
    double getDx()
        {
        return m_dx;
        }

    //! Set the histogram bin width
    //! \param dx histogram bin width
    void setDx(double dx)
        {
        m_dx = dx;
        }

    //! Analyze the current configuration
    virtual void compute(uint64_t timestep);

    //! Return an sdf
    virtual pybind11::array_t<double> getSDF();

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< The parent integrator
    double m_xmax;                                   //!< Maximum lambda value
    double m_dx;                                     //!< Histogram step size
    int m_mode;                        //!< Sampling mode; 0 = binary search, 1 = random sampling
    unsigned int m_num_random_samples; //!< Number of random samples to take if m_mode == 1
    double m_beta;                     //!< Inverse temperature

    std::vector<double> m_hist; //!< Raw histogram data
    std::vector<double> m_sdf;  //!< Computed SDF

    Scalar m_last_max_diam; //!< Last recorded maximum diameter
    unsigned int m_instance = 0;

    //! Zero the histogram counts
    void zeroHistogram();

    //! Add to histogram counts
    void countHistogram(uint64_t timestep);

    //! Determine the s bin of a given particle pair
    size_t computeBin(const vec3<Scalar>& r_ij,
                      const quat<Scalar>& orientation_i,
                      const quat<Scalar>& orientation_j,
                      const typename Shape::param_type& params_i,
                      const typename Shape::param_type& params_j);

    //! Return the sdf
    virtual void computeSDF(uint64_t timestep);
    }; // end class ComputeSDF

template<class Shape>
ComputeSDF<Shape>::ComputeSDF(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                              double xmax,
                              double dx,
                              int mode,
                              unsigned int num_random_samples,
                              double beta)
    : Compute(sysdef), m_mc(mc), m_xmax(xmax), m_dx(dx), m_mode(mode),
      m_num_random_samples(num_random_samples), m_beta(beta)
    {
    m_exec_conf->msg->notice(5) << "Constructing ComputeSDF: " << xmax << " " << dx << std::endl;

    zeroHistogram();

    Scalar max_diam = m_mc->getMaxCoreDiameter();
    m_last_max_diam = max_diam;
    Scalar extra = xmax * max_diam;
    m_mc->setExtraGhostWidth(extra);
    }

template<class Shape> void ComputeSDF<Shape>::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    if (!shouldCompute(timestep))
        return;

    // kludge to update the max diameter dynamically if it changes
    Scalar max_diam = m_mc->getMaxCoreDiameter();
    if (max_diam != m_last_max_diam)
        {
        m_last_max_diam = max_diam;
        Scalar extra = m_xmax * max_diam;
        m_mc->setExtraGhostWidth(extra);
        }

    // update ghost layers
    m_mc->communicate(false);

    this->computeSDF(timestep);
    }

/*! \return the current sdf histogram
 */
template<class Shape> void ComputeSDF<Shape>::computeSDF(uint64_t timestep)
    {
    zeroHistogram();

    countHistogram(timestep);

    std::vector<double> hist_total(m_hist);

// in MPI, total up all of the histogram bins from all nodes to the root node
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Reduce(m_hist.data(),
                   hist_total.data(),
                   (unsigned int)m_hist.size(),
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   m_exec_conf->getMPICommunicator());
        }
#endif

    // compute the probability density
    m_sdf.resize(m_hist.size());
    for (size_t i = 0; i < m_hist.size(); i++)
        {
        m_sdf[i] = hist_total[i] / (m_pdata->getNGlobal() * m_dx);
        }
    }

// \return the sdf histogram
template<class Shape> pybind11::array_t<double> ComputeSDF<Shape>::getSDF()
    {
#ifdef ENABLE_MPI
    if (!m_exec_conf->isRoot())
        return pybind11::none();
#endif

    return pybind11::array_t<double>(m_sdf.size(), m_sdf.data());
    }

template<class Shape> void ComputeSDF<Shape>::zeroHistogram()
    {
    // resize the histogram
    m_hist.resize((size_t)(m_xmax / m_dx));
    // Zero the histogram
    for (size_t i = 0; i < m_hist.size(); i++)
        {
        m_hist[i] = 0.0;
        }
    }

/*! \param timestep current timestep

    countHistogram() loops through all particle pairs *i,j* where *i* is on the local rank, computes
    the bin in which that pair should be and adds 1 to the bin. countHistogram() can be called
    multiple times to increment the counters for averaging, and it operates without any
    communication.
      - The integrator performs the ghost exchange (with the ghost width extra that we add)
      - Only on writeOutput() do we need to sum the per-rank histograms into a global histogram
*/
template<class Shape> void ComputeSDF<Shape>::countHistogram(uint64_t timestep)
    {
    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();
    // update the image list
    const std::vector<vec3<Scalar>>& image_list = m_mc->updateImageList();

    Scalar extra_width = m_xmax / (1 - m_xmax) * m_mc->getMaxCoreDiameter();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    const std::vector<param_type, hoomd::detail::managed_allocator<param_type>>& params
        = m_mc->getParams();

    const uint16_t seed = m_sysdef->getSeed();
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCComputeSDF, timestep, seed),
                               hoomd::Counter(m_instance));

    // loop through N particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        size_t min_bin = m_hist.size();
        double hist_weight = 0.0;

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // TODO: account for patch r_cut?

        // construct the AABB around the particle's circumsphere
        // pad with enough extra width so that when scaled by xmax, found particles might touch
        hoomd::detail::AABB aabb_i_local(vec3<Scalar>(0, 0, 0),
                                         shape_i.getCircumsphereDiameter() / Scalar(2)
                                             + extra_width);

        size_t n_images = image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes();
                 cur_node_idx++)
                {
                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0;
                             cur_p < aabb_tree.getNodeNumParticles(cur_node_idx);
                             cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];
                            int typ_j = __scalar_as_int(postype_j.w);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            if (m_mode == 0)
                                {
                                size_t bin = computeBin(r_ij,
                                                        quat<Scalar>(orientation_i),
                                                        quat<Scalar>(orientation_j),
                                                        params[__scalar_as_int(postype_i.w)],
                                                        params[__scalar_as_int(postype_j.w)]);

                                if (bin >= 0)
                                    {
                                    min_bin = std::min(min_bin, bin);
                                    hist_weight = 1.0;
                                    }
                                } // end binary search path

                            else if (m_mode == 1) // do the random sampling
                                {
                                // first evaluate the patch energy to calculate u_ij_0
                                // for m_num_random_samples, pick a random bin to sample from 0 to
                                // N_bins - 1 do the scaling, then check overlap if there is an
                                // overlap, set min_bin to the sample bin and continue if there is
                                // no overlap, compute the patch energy at the scaled value if no
                                // overlap or u_ij_new - u_ij_0 == 0, then do nothing go on if there
                                // is an overlap or change in energy, then set min_bin to the bin
                                // that was just evaluated, and continue in the loop, but this time
                                // drawing random bins from 0 to min_bin - 1
                                double u_ij_0
                                    = 0.0; // energy of pair interaction in unperturbed state
                                if (m_mc->m_patch)
                                    {
                                    u_ij_0 = m_mc->m_patch->energy(r_ij,
                                                                   typ_i,
                                                                   quat<float>(shape_i.orientation),
                                                                   float(h_diameter.data[i]),
                                                                   float(h_charge.data[i]),
                                                                   typ_j,
                                                                   quat<float>(orientation_j),
                                                                   float(h_diameter.data[j]),
                                                                   float(h_charge.data[j]));
                                    }
                                for (size_t i = 0; i < m_num_random_samples; i++)
                                    {
                                    uint32_t bin_to_sample
                                        = hoomd::UniformIntDistribution((uint32_t)min_bin)(rng);
                                    double scale_factor = m_dx * double(bin_to_sample + 1);
                                    // first check for any "hard" overlaps
                                    // if there is one for a given scale value, there is no need to
                                    // check for any "soft" overlaps from m_mc.m_patch
                                    if (detail::test_scaled_overlap<Shape>(
                                            r_ij,
                                            quat<Scalar>(orientation_i),
                                            quat<Scalar>(orientation_j),
                                            params[__scalar_as_int(postype_i.w)],
                                            params[__scalar_as_int(postype_j.w)],
                                            scale_factor))
                                        {
                                        min_bin = std::min((size_t)bin_to_sample, min_bin);
                                        if (min_bin == bin_to_sample)
                                            {
                                            hist_weight
                                                = 1.0; // hist_weight = 1 - epx(-beta*du) = 1 - 0
                                            }
                                        continue;
                                        } // end if overlap
                                    if (m_mc->m_patch)
                                        // if no hard overlap, check for a soft overla
                                        //
                                        // the energy at this scale factor and seeing if it has
                                        // changed compared to the current value at the scaling
                                        // corresponding to min_bin
                                        {
                                        vec3<Scalar> r_ij_scaled = r_ij * scale_factor;
                                        double u_ij_new = m_mc->m_patch->energy(
                                            r_ij_scaled,
                                            typ_i,
                                            quat<float>(shape_i.orientation),
                                            float(h_diameter.data[i]),
                                            float(h_charge.data[i]),
                                            typ_j,
                                            quat<float>(orientation_j),
                                            float(h_diameter.data[j]),
                                            float(h_charge.data[j]));
                                        if (u_ij_new != u_ij_0)
                                            {
                                            min_bin = std::min((size_t)bin_to_sample, min_bin);
                                            if (min_bin == bin_to_sample)
                                                {
                                                hist_weight
                                                    = 1.0
                                                      - fast::exp(-m_beta * (u_ij_new - u_ij_0));
                                                }
                                            }
                                        } // end if (m_mc->m_patch)
                                    }     // end loop over m_num_random_samples
                                }         // end random sampling path
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                    }
                } // end loop over AABB nodes
            }     // end loop over images

        // record the minimum bin
        if ((unsigned int)min_bin < m_hist.size())
            {
            m_exec_conf->msg->notice(2)
                << "adding " << 1.0 * hist_weight << " to bin " << min_bin << std::endl;
            m_hist[min_bin] += 1.0 * hist_weight;
            }

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
    left boundary and does overlap at the right. Then it picks a new point halfway between
    the left and right, ensuring that the same assumption holds. Once right=left+1, the
    correct bin has been found.
*/
template<class Shape>
size_t ComputeSDF<Shape>::computeBin(const vec3<Scalar>& r_ij,
                                     const quat<Scalar>& orientation_i,
                                     const quat<Scalar>& orientation_j,
                                     const typename Shape::param_type& params_i,
                                     const typename Shape::param_type& params_j)
    {
    size_t L = 0;
    size_t R = m_hist.size();

    // if the particles already overlap a the left boundary, return an out of range value
    if (detail::test_scaled_overlap<Shape>(r_ij,
                                           orientation_i,
                                           orientation_j,
                                           params_i,
                                           params_j,
                                           double(L) * m_dx))
        return -1;

    // if the particles do not overlap a the right boundary, return an out of range value
    if (!detail::test_scaled_overlap<Shape>(r_ij,
                                            orientation_i,
                                            orientation_j,
                                            params_i,
                                            params_j,
                                            double(R) * m_dx))
        return m_hist.size();

    // progressively narrow the search window by halves
    do
        {
        size_t m = (L + R) / 2;

        if (detail::test_scaled_overlap<Shape>(r_ij,
                                               orientation_i,
                                               orientation_j,
                                               params_i,
                                               params_j,
                                               double(m) * m_dx))
            R = m;
        else
            L = m;
        } while ((R - L) > 1);

    return L;
    }

namespace detail
    {
//! Export this hpmc compute to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of ComputeSDFe<Shape> will be exported
*/
template<class Shape> void export_ComputeSDF(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ComputeSDF<Shape>, Compute, std::shared_ptr<ComputeSDF<Shape>>>(m,
                                                                                     name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            double,
                            double,
                            int,
                            unsigned int,
                            double>())
        .def_property("xmax", &ComputeSDF<Shape>::getXMax, &ComputeSDF<Shape>::setXMax)
        .def_property("dx", &ComputeSDF<Shape>::getDx, &ComputeSDF<Shape>::setDx)
        .def_property_readonly("sdf", &ComputeSDF<Shape>::getSDF);
    }

    } // end namespace detail
    } // end namespace hpmc

    } // end namespace hoomd

#endif // __COMPUTE_SDF__H__
