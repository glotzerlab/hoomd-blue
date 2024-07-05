// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancer.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#pragma once
#include "Trigger.h"
#include "Tuner.h"

#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace hoomd
    {
//! Updates domain decompositions to balance the load
/*!
 * Adjusts the boundaries of the processor domains to distribute the load close to evenly between
 * them. The load imbalance is defined as the number of particles owned by a rank divided by the
 * average number of particles per rank if the particles had a uniform distribution.
 *
 * At each load balancing step, we attempt to rescale the domain size by the inverse of the load
 * balance, subject to the following constraints that are imposed to both maintain a stable
 * balancing and to keep communication isolated to the 26 nearest neighbors of a cell:
 *  1. No domain may move more than half its neighboring domains.
 *  2. No domain may be smaller than the minimum size set by the ghost layer.
 *  3. A domain should change size by at most approximately 5% in a single rescaling.
 *
 * Constraints are satisfied by solving a least-squares problem with box constraints, where the cost
 * function is the deviation of the domain sizes from the proposed rescaled width.
 *
 * \ingroup updaters
 */
class PYBIND11_EXPORT LoadBalancer : public Tuner
    {
    public:
    //! Constructor
    LoadBalancer(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);
    //! Destructor
    virtual ~LoadBalancer();

    //! Get the tolerance for load balancing
    Scalar getTolerance() const
        {
        return m_tolerance;
        }

    //! Get the maximum number of iterations to attempt in a single rebalancing step
    unsigned int getMaxIterations() const
        {
        return m_maxiter;
        }

    //! Set the tolerance for load balancing
    /*!
     * \param tolerance Load imbalance below which no balancing is attempted (<= 1.0 forces
     * rebalancing)
     */
    void setTolerance(Scalar tolerance)
        {
        m_tolerance = tolerance;
        }

    //! Set the maximum number of iterations to attempt in a single rebalancing step
    /*!
     * \param maxiter Maximum number of times to attempt to balance in a single update()
     *
     * If the load imbalance is reduced below the tolerance, fewer iterations than \a maxiter are
     * performed.
     */
    void setMaxIterations(unsigned int maxiter)
        {
        m_maxiter = maxiter;
        }

    //! Enable / disable load balancing along a dimension
    /*!
     * \param dim Dimension along which to balance
     * \param enable Flag to balance (true) or not balance (false)
     */
    void enableDimension(unsigned int dim, bool enable)
        {
        if (dim == 0)
            m_enable_x = enable;
        else if (dim == 1)
            m_enable_y = enable;
        else if (dim == 2)
            m_enable_z = enable;
        else
            {
            throw std::runtime_error("LoadBalancer: requested direction does not exist");
            }
        }

    /// Set m_enable_x
    void setEnableX(bool enable)
        {
        m_enable_x = enable;
        }

    /// Get value of m_enable_x
    bool getEnableX()
        {
        return m_enable_x;
        }

    /// Set m_enable_y
    void setEnableY(bool enable)
        {
        m_enable_y = enable;
        }

    /// Get value of m_enable_y
    bool getEnableY()
        {
        return m_enable_y;
        }

    /// Set m_enable_z
    void setEnableZ(bool enable)
        {
        m_enable_z = enable;
        }

    /// Get value of m_enable_z
    bool getEnableZ()
        {
        return m_enable_z;
        }

    //! Take one timestep forward
    virtual void update(uint64_t timestep);

    //! Reset the counters for the run
    virtual void resetStats();

    protected:
    std::shared_ptr<DomainDecomposition> m_decomposition; //!< The domain decomposition to balance
    std::shared_ptr<Trigger> m_trigger;

#ifdef ENABLE_MPI
    const MPI_Comm m_mpi_comm; //!< MPI communicator for all ranks

    /// The systems's communicator.
    std::shared_ptr<Communicator> m_comm;

    //! Computes the maximum imbalance factor
    Scalar getMaxImbalance();

    //! Reduce the particle numbers per rank down to one dimension
    bool reduce(std::vector<unsigned int>& N_i, unsigned int dim, unsigned int reduce_root);

    //! Set flags within the class that a resize has been performed
    void signalResize()
        {
        m_recompute_max_imbalance = true;
        m_needs_migrate = true;
        m_needs_recount = true;
        }

    //! Adjust the partitioning along a single dimension
    bool adjust(std::vector<Scalar>& cum_frac_i,
                const std::vector<unsigned int>& N_i,
                Scalar L_i,
                Scalar min_domain_frac);

    //! Compute the number of particles on each rank after an adjustment
    void computeOwnedParticles();

    //! Count the number of particles that have gone off the rank
    virtual void countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts);

    //! Gets the number of owned particles, updating if necessary
    unsigned int getNOwn()
        {
        computeOwnedParticles();
        return m_N_own;
        }

    //! Force a reset of the number of owned particles without counting
    /*!
     * \param N number of particles owned by the rank
     */
    void resetNOwn(unsigned int N)
        {
        m_N_own = N;
        m_recompute_max_imbalance = true;
        m_needs_recount = false;
        }
#endif // ENABLE_MPI

    Scalar m_max_imbalance;         //!< Maximum imbalance
    bool m_recompute_max_imbalance; //!< Flag if maximum imbalance needs to be computed

    bool m_needs_migrate; //!< Flag to signal that migration is necessary
    bool m_needs_recount; //!< Flag if a particle change needs to be computed

    Scalar m_tolerance;     //!< Load imbalance to tolerate
    unsigned int m_maxiter; //!< Maximum number of iterations to attempt
    bool m_enable_x;        //!< Flag to enable balancing in x
    bool m_enable_y;        //!< Flag to enable balancing in y
    bool m_enable_z;        //!< Flag to enable balancing z

    const Scalar m_max_scale; //!< Maximum fraction to rescale either direction (5%)

    private:
    unsigned int m_N_own; //!< Number of particles owned by this rank

    Scalar m_max_max_imbalance;   //!< The maximum imbalance of any check
    double m_total_max_imbalance; //!< The average imbalance over checks
    uint64_t m_n_calls;           //!< The number of times the updater was called
    uint64_t m_n_iterations;      //!< The actual number of balancing iterations performed
    uint64_t m_n_rebalances;      //!< The actual number of rebalances (migrations) performed
    };

namespace detail
    {
//! Export the LoadBalancer to python
void export_LoadBalancer(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
