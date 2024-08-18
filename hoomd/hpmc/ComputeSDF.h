// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COMPUTE_SDF__H__
#define __COMPUTE_SDF__H__

#include "hoomd/Autotuner.h"
#include "hoomd/CellList.h"
#include "hoomd/Compute.h"

#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"

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
    //! Shape parameters
    typedef typename Shape::param_type param_type;

    //! Construct the integrator
    ComputeSDF(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
               double xmax,
               double dx);

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

    /// Get the number of bins
    size_t getNumBins()
        {
        assert(m_hist_compression.size() == m_hist_expansion.size());
        return m_hist_compression.size();
        }

    //! Analyze the current configuration
    virtual void compute(uint64_t timestep);

    //! Return an sdf
    virtual pybind11::object getSDFCompression();
    virtual pybind11::object getSDFExpansion();

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< The parent integrator
    double m_xmax;                                   //!< Maximum lambda value
    double m_dx;                                     //!< Histogram step size

    bool m_shape_requires_expansion_moves;  //!< If expansion moves are required
    std::vector<double> m_hist_compression; //!< Raw histogram data
    std::vector<double> m_sdf_compression;  //!< Computed SDF
    std::vector<double> m_hist_expansion;   //!< Raw histogram data
    std::vector<double> m_sdf_expansion;    //!< Computed SDF

    //! Find the maximum particle separation beyond which all interactions are zero
    Scalar getMaxInteractionDiameter();
    Scalar m_last_max_diam; //!< Last recorded maximum diameter

    //! Zero the histogram counts
    void zeroHistogram();

    //! Add to histogram counts
    void countHistogram(uint64_t timestep);
    void countHistogramBinarySearch(uint64_t timestep);
    void countHistogramLinearSearch(uint64_t timestep);

    //! Determine the s bin of a given particle pair; only used for the binary search
    size_t computeBin(const vec3<Scalar>& r_ij,
                      const quat<Scalar>& orientation_i,
                      const quat<Scalar>& orientation_j,
                      const typename Shape::param_type& params_i,
                      const typename Shape::param_type& params_j);

    //! Return the sdf
    virtual void computeSDF(uint64_t timestep);
    };

template<class Shape>
ComputeSDF<Shape>::ComputeSDF(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                              double xmax,
                              double dx)
    : Compute(sysdef), m_mc(mc), m_xmax(xmax), m_dx(dx)
    {
    m_exec_conf->msg->notice(5) << "Constructing ComputeSDF: " << xmax << " " << dx << std::endl;

    zeroHistogram();

    Scalar max_diam = this->getMaxInteractionDiameter();
    m_last_max_diam = max_diam;
    Scalar extra = xmax * max_diam;
    m_mc->setExtraGhostWidth(extra);

    // default to requiring expansive moves then check if only compressions are required
    m_shape_requires_expansion_moves = true;
    if constexpr (std::is_same<Shape, ShapeSphere>() || std::is_same<Shape, ShapeConvexPolygon>()
                  || std::is_same<Shape, ShapeConvexPolyhedron>()
                  || std::is_same<Shape, ShapeEllipsoid>()
                  || std::is_same<Shape, ShapeFacetedEllipsoid>()
                  || std::is_same<Shape, ShapeSpheropolyhedron>()
                  || std::is_same<Shape, ShapeSpheropolygon>())
        {
        m_shape_requires_expansion_moves = false;
        }
    }

template<class Shape> void ComputeSDF<Shape>::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    if (!shouldCompute(timestep))
        return;

    // kludge to update the max diameter dynamically if it changes
    Scalar max_diam = this->getMaxInteractionDiameter();
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

    std::vector<double> hist_total(m_hist_compression);
    std::vector<double> hist_total_expansion(m_hist_expansion);

// in MPI, total up all of the histogram bins from all nodes to the root node
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Reduce(m_hist_compression.data(),
                   hist_total.data(),
                   (unsigned int)m_hist_compression.size(),
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   m_exec_conf->getMPICommunicator());
        MPI_Reduce(m_hist_expansion.data(),
                   hist_total_expansion.data(),
                   (unsigned int)m_hist_expansion.size(),
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   m_exec_conf->getMPICommunicator());
        }
#endif

    // compute the probability density
    m_sdf_compression.resize(m_hist_compression.size());
    m_sdf_expansion.resize(m_hist_expansion.size());
    for (size_t i = 0; i < m_hist_compression.size(); i++)
        {
        m_sdf_compression[i] = hist_total[i] / (m_pdata->getNGlobal() * m_dx);
        }
    for (size_t i = 0; i < m_hist_expansion.size(); i++)
        {
        m_sdf_expansion[i] = hist_total_expansion[i] / (m_pdata->getNGlobal() * m_dx);
        }
    }

// \return the sdf histogram
template<class Shape> pybind11::object ComputeSDF<Shape>::getSDFCompression()
    {
#ifdef ENABLE_MPI
    if (!m_exec_conf->isRoot())
        return pybind11::none();
#endif

    return pybind11::array_t<double>(m_sdf_compression.size(), m_sdf_compression.data());
    }

// \return the sdf histogram for expansion moves
template<class Shape> pybind11::object ComputeSDF<Shape>::getSDFExpansion()
    {
#ifdef ENABLE_MPI
    if (!m_exec_conf->isRoot())
        return pybind11::none();
#endif

    return pybind11::array_t<double>(m_sdf_expansion.size(), m_sdf_expansion.data());
    }

template<class Shape> void ComputeSDF<Shape>::zeroHistogram()
    {
    // resize the histogram
    m_hist_compression.resize((size_t)(m_xmax / m_dx));
    m_hist_expansion.resize((size_t)(m_xmax / m_dx));
    // Zero the histogram
    for (size_t i = 0; i < m_hist_compression.size(); i++)
        {
        m_hist_compression[i] = 0.0;
        m_hist_expansion[i] = 0.0;
        }
    }

template<class Shape> Scalar ComputeSDF<Shape>::getMaxInteractionDiameter()
    {
    const Scalar max_core_diameter = m_mc->getMaxCoreDiameter();
    Scalar max_r_cut_patch = 0.0;

    for (unsigned int typ_i = 0; typ_i < m_pdata->getNTypes(); typ_i++)
        {
        const Scalar r_cut_patch_i = m_mc->getMaxPairEnergyRCutNonAdditive()
                                     + m_mc->getMaxPairInteractionAdditiveRCut(typ_i);
        max_r_cut_patch = std::max(max_r_cut_patch, r_cut_patch_i);
        }

    return std::max(max_core_diameter, max_r_cut_patch);
    }

/*! \param timestep current timestep

    countHistogram() loops through all particle pairs *i,j* where *i* is on the local rank, computes
    the bin corresponding to the scale factor of the vector r_ij that induces the first overlap
    between particles *i* and *j*, and then adds s_ij to the bin, where s_ij = 1 - e^{\beta\Delta
    U_{ij}}. Note that this definition allows for the consideration of soft overlaps where the pair
    potential changes value, and reduces to 1 for hard particle overlaps.  countHistogram() operates
    without any communication.
      - The integrator performs the ghost exchange (with the ghost width extra that we add)

    This function is a wrapper that calls the appropriate method depending on whether a binary or
    linear search is required.
*/
template<class Shape> void ComputeSDF<Shape>::countHistogram(uint64_t timestep)
    {
    if (m_mc->hasPairInteractions() || m_shape_requires_expansion_moves)
        {
        countHistogramLinearSearch(timestep);
        }
    else
        {
        countHistogramBinarySearch(timestep);
        }
    } // end countHistogram()

template<class Shape> void ComputeSDF<Shape>::countHistogramBinarySearch(uint64_t timestep)
    {
    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = m_mc->updateImageList();

    Scalar extra_width = m_xmax / (1 - m_xmax) * m_last_max_diam;

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    const std::vector<param_type, hoomd::detail::managed_allocator<param_type>>& params
        = m_mc->getParams();

    // loop through N particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        size_t min_bin = m_hist_compression.size();
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        const quat<LongReal> orientation_i(h_orientation.data[i]);
        Shape shape_i(orientation_i, params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

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
                if (aabb.overlaps(aabb_tree.getNodeAABB(cur_node_idx)))
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
                            const quat<LongReal> orientation_j(h_orientation.data[j]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            size_t bin = computeBin(r_ij,
                                                    orientation_i,
                                                    orientation_j,
                                                    params[__scalar_as_int(postype_i.w)],
                                                    params[__scalar_as_int(postype_j.w)]);

                            if (bin >= 0)
                                {
                                min_bin = std::min(min_bin, bin);
                                }
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
        if (min_bin < m_hist_compression.size())
            {
            m_hist_compression[min_bin]++;
            }
        } // end loop over all particles
    } // end countHistogramBinarySearch()

template<class Shape> void ComputeSDF<Shape>::countHistogramLinearSearch(uint64_t timestep)
    {
    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = m_mc->updateImageList();

    // Note - If needed for future simulations with a large disparity in additive cutoffs, compute
    // extra_width_i with knowledge of the additive cutoff of type i and half the largest additive
    // cutoff.
    Scalar extra_width = m_xmax / (1 - m_xmax) * m_last_max_diam;

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
    const auto& params = m_mc->getParams();

    // precompute constants used many times in the loop
    const LongReal min_core_radius = m_mc->getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = m_mc->getPairEnergySearchRadius();

    Scalar kT = m_mc->getTimestepkT(timestep); // use const?

    // loop through N particles
    // At the top of this loop, we initialize min_bin to the size of the sdf histogram
    // For each of particle i's neighbors, we find the scaling that produces the first overlap.
    // For each neighbor, we do a brute force search from the scaling corresponding to bin 0
    // up to the minimum bin that we've already found for particle i.
    // Then we add to m_hist_compression[min_bin] the negative Mayer-function corresponding to the
    // type of overlap corresponding to particle i's first overlap.
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        size_t min_bin_compression = m_hist_compression.size();
        size_t min_bin_expansion = m_hist_expansion.size();
        double hist_weight_ptl_i_compression = 2.0;
        double hist_weight_ptl_i_expansion = 2.0;

        // read in the current position and orientation
        const Scalar4 postype_i = h_postype.data[i];
        const quat<LongReal> orientation_i(h_orientation.data[i]);
        const int typ_i = __scalar_as_int(postype_i.w);
        const Shape shape_i(orientation_i, params[__scalar_as_int(postype_i.w)]);
        const vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // construct the AABB around the particle's circumsphere
        // pad with enough extra width so that when scaled by xmax, found particles might touch
        const LongReal R_query = std::max(shape_i.getCircumsphereDiameter() * LongReal(0.5),
                                          pair_energy_search_radius[typ_i] - min_core_radius);
        hoomd::detail::AABB aabb_i_local(vec3<Scalar>(0, 0, 0), R_query + extra_width);

        const size_t n_images = image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes();
                 cur_node_idx++)
                {
                if (aabb.overlaps(aabb_tree.getNodeAABB(cur_node_idx)))
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
                                {
                                continue;
                                }

                            const Scalar4 postype_j = h_postype.data[j];
                            const quat<LongReal> orientation_j(h_orientation.data[j]);
                            const int typ_j = __scalar_as_int(postype_j.w);

                            // put particles in coordinate system of particle i
                            const vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            double u_ij_0 = 0.0; // energy of pair interaction in unperturbed state
                            u_ij_0 = m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
                                                                r_ij,
                                                                typ_i,
                                                                shape_i.orientation,
                                                                h_diameter.data[i],
                                                                h_charge.data[i],
                                                                typ_j,
                                                                orientation_j,
                                                                h_diameter.data[j],
                                                                h_charge.data[j]);

                            // first do compressions
                            for (size_t bin_to_sample = 0; bin_to_sample < min_bin_compression;
                                 bin_to_sample++)
                                {
                                const double scale_factor
                                    = m_dx * static_cast<double>(bin_to_sample + 1);

                                // check for hard overlaps
                                // if there is one for a given scale value, there is no need to
                                // check for any soft overlaps from m_mc.m_patch
                                bool hard_overlap = detail::test_scaled_overlap<Shape>(
                                    r_ij,
                                    orientation_i,
                                    orientation_j,
                                    params[__scalar_as_int(postype_i.w)],
                                    params[__scalar_as_int(postype_j.w)],
                                    scale_factor);
                                if (hard_overlap)
                                    {
                                    hist_weight_ptl_i_compression = 1.0; // = 1-e^(-\infty)
                                    min_bin_compression = bin_to_sample;
                                    } // end if (hard_overlap)

                                // if no hard overlap, check for a soft overlap if we have
                                // patches
                                if (!hard_overlap)
                                    {
                                    // compute the energy at this size of the perturbation and
                                    // compare to the energy in the unperturbed state
                                    const vec3<Scalar> r_ij_scaled
                                        = r_ij * (Scalar(1.0) - scale_factor);
                                    double u_ij_new
                                        = m_mc->computeOnePairEnergy(dot(r_ij_scaled, r_ij_scaled),
                                                                     r_ij_scaled,
                                                                     typ_i,
                                                                     shape_i.orientation,
                                                                     h_diameter.data[i],
                                                                     h_charge.data[i],
                                                                     typ_j,
                                                                     orientation_j,
                                                                     h_diameter.data[j],
                                                                     h_charge.data[j]);
                                    // if energy has changed, there is a new soft overlap
                                    // add the appropriate weight to the appropriate bin of the
                                    // histogram and break out of the loop over bins
                                    if (u_ij_new != u_ij_0)
                                        {
                                        min_bin_compression = bin_to_sample;
                                        if (u_ij_new < u_ij_0)
                                            {
                                            hist_weight_ptl_i_compression = 0;
                                            }
                                        else
                                            {
                                            hist_weight_ptl_i_compression
                                                = 1.0 - fast::exp(-(u_ij_new - u_ij_0) / kT);
                                            }
                                        }
                                    } // end if (!hard_overlap)
                                } // end loop over bins for compression

                            // do expansions
                            for (size_t bin_to_sample = 0; bin_to_sample < min_bin_expansion;
                                 bin_to_sample++)
                                {
                                const double scale_factor
                                    = -m_dx * static_cast<double>(bin_to_sample + 1);

                                // check for hard overlaps
                                // if there is one for a given scale value, there is no need to
                                // check for any soft overlaps from m_mc.m_patch
                                bool hard_overlap = detail::test_scaled_overlap<Shape>(
                                    r_ij,
                                    orientation_i,
                                    orientation_j,
                                    params[__scalar_as_int(postype_i.w)],
                                    params[__scalar_as_int(postype_j.w)],
                                    scale_factor);
                                if (hard_overlap)
                                    {
                                    hist_weight_ptl_i_expansion = 1.0; // = 1-e^(-\infty)
                                    min_bin_expansion = bin_to_sample;
                                    } // end if (hard_overlap)

                                // if no hard overlap, check for a soft overlap if necessary
                                if (!hard_overlap)
                                    {
                                    // compute the energy at this size of the perturbation and
                                    // compare to the energy in the unperturbed state
                                    const vec3<Scalar> r_ij_scaled
                                        = r_ij * (Scalar(1.0) - scale_factor);
                                    double u_ij_new
                                        = m_mc->computeOnePairEnergy(dot(r_ij_scaled, r_ij_scaled),
                                                                     r_ij_scaled,
                                                                     typ_i,
                                                                     shape_i.orientation,
                                                                     h_diameter.data[i],
                                                                     h_charge.data[i],
                                                                     typ_j,
                                                                     orientation_j,
                                                                     h_diameter.data[j],
                                                                     h_charge.data[j]);
                                    // if energy has changed, there is a new soft overlap
                                    // add the appropriate weight to the appropriate bin of the
                                    // histogram and break out of the loop over bins
                                    if (u_ij_new != u_ij_0)
                                        {
                                        min_bin_expansion = bin_to_sample;
                                        if (u_ij_new < u_ij_0)
                                            {
                                            hist_weight_ptl_i_expansion = 0;
                                            }
                                        else
                                            {
                                            hist_weight_ptl_i_expansion
                                                = 1.0 - fast::exp(-(u_ij_new - u_ij_0) / kT);
                                            }
                                        }
                                    } // end if (!hard_overlap)
                                } // end loop over histogram bins for expansions
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
        if (min_bin_compression < m_hist_compression.size() && hist_weight_ptl_i_compression <= 1.0)
            {
            m_hist_compression[min_bin_compression] += hist_weight_ptl_i_compression;
            }
        if (min_bin_expansion < m_hist_expansion.size() && hist_weight_ptl_i_expansion <= 1.0)
            {
            m_hist_expansion[min_bin_expansion] += hist_weight_ptl_i_expansion;
            }
        } // end loop over all particles
    } // end countHistogramLinearSearch()

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
    size_t R = m_hist_compression.size();

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
        return m_hist_compression.size();

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
                            double>())
        .def_property("xmax", &ComputeSDF<Shape>::getXMax, &ComputeSDF<Shape>::setXMax)
        .def_property("dx", &ComputeSDF<Shape>::getDx, &ComputeSDF<Shape>::setDx)
        .def_property_readonly("sdf_compression", &ComputeSDF<Shape>::getSDFCompression)
        .def_property_readonly("sdf_expansion", &ComputeSDF<Shape>::getSDFExpansion)
        .def_property_readonly("num_bins", &ComputeSDF<Shape>::getNumBins);
    }

    } // end namespace detail
    } // end namespace hpmc

    } // end namespace hoomd

#endif // __COMPUTE_SDF__H__
