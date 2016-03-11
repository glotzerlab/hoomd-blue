/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: mphoward

/*! \file LoadBalancer.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_MPI

#ifndef __LOADBALANCER_H__
#define __LOADBALANCER_H__

#include "Updater.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <map>

//! Updates domain decompositions to balance the load
/*!
 * Adjusts the boundaries of the processor domains to distribute the load close to evenly between them. The load imbalance
 * is defined as the number of particles owned by a rank divided by the average number of particles per rank if the
 * particles had a uniform distribution.
 *
 * At each load balancing step, we attempt to rescale the domain size by the inverse of the load balance, subject to the
 * following constraints that are imposed to both maintain a stable balancing and to keep communication isolated to the
 * 26 nearest neighbors of a cell:
 *  1. No domain may move more than half its neighboring domains.
 *  2. No domain may be smaller than the minimum size set by the ghost layer.
 *  3. A domain should change size by at most approximately 5% in a single rescaling.
 *
 * Constraints are satisfied by solving a least-squares problem with box constraints, where the cost function is the
 * deviation of the domain sizes from the proposed rescaled width.
 *
 * \ingroup updaters
 */
class LoadBalancer : public Updater
    {
    public:
        //! Constructor
        LoadBalancer(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<DomainDecomposition> decomposition);
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
         * \param tolerance Load imbalance below which no balancing is attempted (<= 1.0 forces rebalancing)
         */
        void setTolerance(Scalar tolerance)
            {
            m_tolerance = tolerance;
            }

        //! Set the maximum number of iterations to attempt in a single rebalancing step
        /*!
         * \param maxiter Maximum number of times to attempt to balance in a single update()
         *
         * If the load imbalance is reduced below the tolerance, fewer iterations than \a maxiter are performed.
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
            if (dim == 0) m_enable_x = enable;
            else if (dim == 1) m_enable_y = enable;
            else if (dim == 2) m_enable_z = enable;
            else
                {
                m_exec_conf->msg->error() << "comm: requested direction does not exist" << std::endl;
                throw std::runtime_error("comm: requested direction does not exist");
                }
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Print load balancer counters
        virtual void printStats();

        //! Reset the counters for the run
        virtual void resetStats();

    protected:
        boost::shared_ptr<DomainDecomposition> m_decomposition; //!< The domain decomposition to balance

        const MPI_Comm m_mpi_comm;  //!< MPI communicator for all ranks

        //! Computes the maximum imbalance factor
        Scalar getMaxImbalance();
        Scalar m_max_imbalance;             //!< Maximum imbalance
        bool m_recompute_max_imbalance;     //!< Flag if maximum imbalance needs to be computed

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
        bool m_needs_migrate;   //!< Flag to signal that migration is necessary

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
        bool m_needs_recount;   //!< Flag if a particle change needs to be computed

        Scalar m_tolerance;     //!< Load imbalance to tolerate
        unsigned int m_maxiter; //!< Maximum number of iterations to attempt
        bool m_enable_x;        //!< Flag to enable balancing in x
        bool m_enable_y;        //!< Flag to enable balancing in y
        bool m_enable_z;        //!< Flag to enable balancing z

        const Scalar m_max_scale;   //!< Maximum fraction to rescale either direction (5%)

    private:
        unsigned int m_N_own;               //!< Number of particles owned by this rank

        Scalar m_max_max_imbalance;     //!< The maximum imbalance of any check
        double m_total_max_imbalance;   //!< The average imbalance over checks
        uint64_t m_n_calls;             //!< The number of times the updater was called
        uint64_t m_n_iterations;        //!< The actual number of balancing iterations performed
        uint64_t m_n_rebalances;        //!< The actual number of rebalances (migrations) performed
    };

//! Export the LoadBalancer to python
void export_LoadBalancer();

#endif // __LOADBALANCER_H__
#endif // ENABLE_MPI
