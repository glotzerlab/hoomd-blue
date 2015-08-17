/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

/*! \file BalancedDomainDecomposition.cc
    \brief Implements the BalancedDomainDecomposition class
*/
#ifdef ENABLE_MPI

#include "BalancedDomainDecomposition.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#include <boost/python.hpp>
using namespace boost::python;

BalancedDomainDecomposition::BalancedDomainDecomposition(boost::shared_ptr<ExecutionConfiguration> exec_conf,
                                                         Scalar3 L,
                                                         const std::vector<Scalar>& fxs,
                                                         const std::vector<Scalar>& fys,
                                                         const std::vector<Scalar>& fzs)
    : DomainDecomposition(exec_conf, L, fxs.size(), fys.size(), fzs.size()), m_uniform(false)
    {

    // everyone needs to just resize their vectors, rank 0 will fill them in
    m_frac_x = std::vector<Scalar>(m_nx, 0.0);
    m_frac_y = std::vector<Scalar>(m_ny, 0.0);
    m_frac_z = std::vector<Scalar>(m_nz, 0.0);

    m_cum_frac_x = std::vector<Scalar>(m_nx+1, 0.0);
    m_cum_frac_y = std::vector<Scalar>(m_ny+1, 0.0);
    m_cum_frac_z = std::vector<Scalar>(m_nz+1, 0.0);

    unsigned int rank = m_exec_conf->getRank();
    if (rank == 0)
        {
        m_frac_x = fxs;
        m_frac_y = fys;
        m_frac_z = fzs;

        // the DomainDecomposition constructor should respect our choice if we specify all three dimensions
        // if it doesn't, then an incorrect grid was chosen by the user and we need to warn them and fall back to the uniform case
        if ((m_frac_x.size() > 0 && m_frac_x.size() != m_nx) || 
            (m_frac_y.size() > 0 && m_frac_y.size() != m_ny) ||
            (m_frac_z.size() > 0 && m_frac_z.size() != m_nz))
            {
            m_exec_conf->msg->warning() << "Domain decomposition grid does not match specification, defaulting to uniform spacing" << std::endl;

            m_frac_x = std::vector<Scalar>(m_nx, Scalar(1.0) / Scalar(m_nx));
            m_frac_y = std::vector<Scalar>(m_nx, Scalar(1.0) / Scalar(m_ny));
            m_frac_z = std::vector<Scalar>(m_nx, Scalar(1.0) / Scalar(m_nz));
            m_uniform = true;
            }
            
        if (m_frac_x.size() == 0) m_frac_x = std::vector<Scalar>(m_nx, Scalar(1.0) / Scalar(m_nx));
        if (m_frac_y.size() == 0) m_frac_y = std::vector<Scalar>(m_ny, Scalar(1.0) / Scalar(m_ny));
        if (m_frac_z.size() == 0) m_frac_z = std::vector<Scalar>(m_nz, Scalar(1.0) / Scalar(m_nz));

        // perform an exclusive sum on the fractions to accumulate the fractions (last entry should sum to 1.0)
        // this stores the cumulative fraction of space below the current cartesian index
        // so, the first cut has 0% below it, the second cut has cum_frac_x[1] below it, and so on
        // we pad the right end with the total sum so that we can validate summation to 1.0, and also for placing particles
        std::partial_sum(m_frac_x.begin(), m_frac_x.end(), m_cum_frac_x.begin() + 1);
        std::partial_sum(m_frac_y.begin(), m_frac_y.end(), m_cum_frac_y.begin() + 1);
        std::partial_sum(m_frac_z.begin(), m_frac_z.end(), m_cum_frac_z.begin() + 1);
        }

    // broadcast the adjusted boxes
    bcast(m_uniform, 0, m_mpi_comm);

    MPI_Bcast(&m_frac_x[0], m_nx, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_frac_y[0], m_ny, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_frac_z[0], m_nz, MPI_HOOMD_SCALAR, 0, m_mpi_comm);

    MPI_Bcast(&m_cum_frac_x[0], m_nx+1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_cum_frac_y[0], m_ny+1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_cum_frac_z[0], m_nz+1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);

    // validate the fractional cuts
    Scalar tol(1e-5);
    if (std::fabs(m_cum_frac_x.back() - Scalar(1.0)) > tol ||
        std::fabs(m_cum_frac_y.back() - Scalar(1.0)) > tol ||
        std::fabs(m_cum_frac_z.back() - Scalar(1.0)) > tol)
        {
        m_exec_conf->msg->error() << "comm: domain decomposition fractions do not sum to 1.0" << std::endl;
        throw std::runtime_error("comm: domain decomposition fractions do not sum to 1.0");
        }
    }

const BoxDim BalancedDomainDecomposition::calculateLocalBox(const BoxDim & global_box)
    {
    // use the simpler method if we have a uniform decomposition
    if (m_uniform)
        {
        return DomainDecomposition::calculateLocalBox(global_box);
        }


    // initialize local box with all properties of global box
    BoxDim box = global_box;

    // calculate the local box dimensions using the fractions of the current rank
    ArrayHandle<unsigned int> h_cart_ranks_inv(m_cart_ranks_inv, access_location::host, access_mode::read);

    Scalar3 L = global_box.getL();
    Scalar3 cur_frac = make_scalar3(m_frac_x[m_grid_pos.x], m_frac_y[m_grid_pos.y], m_frac_z[m_grid_pos.z]);
    Scalar3 L_local = L * cur_frac;

    // position of this domain in the grid
    Scalar3 cum_frac = make_scalar3(m_cum_frac_x[m_grid_pos.x], m_cum_frac_y[m_grid_pos.y], m_cum_frac_z[m_grid_pos.z]);
    Scalar3 lo = global_box.getLo() + cum_frac * L;
    Scalar3 hi = lo + L_local;

    // set periodic flags
    // we are periodic in a direction along which there is only one box
    uchar3 periodic = make_uchar3(m_nx == 1 ? 1 : 0,
                                  m_ny == 1 ? 1 : 0,
                                  m_nz == 1 ? 1 : 0);

    box.setLoHi(lo,hi);
    box.setPeriodic(periodic);
    return box;
    }

unsigned int BalancedDomainDecomposition::placeParticle(const BoxDim& global_box, Scalar3 pos)
    {
    // use the simpler method if we have a uniform decomposition
    if (m_uniform)
        {
        return DomainDecomposition::placeParticle(global_box, pos);
        }

    // get fractional coordinates in the global box
    Scalar3 f = global_box.makeFraction(pos);

    Scalar tol(1e-5);
    // check user input
    if (f.x < -tol|| f.x >= 1.0+tol || f.y < -tol || f.y >= 1.0+tol || f.z < -tol|| f.z >= 1.0+tol)
        {
        m_exec_conf->msg->error() << "Particle coordinates outside box." << std::endl;
        m_exec_conf->msg->error() << "f.x = " << f.x << " f.y = " << f.y << " f.z = " << f.z << std::endl;
        throw std::runtime_error("Error placing particle");
        }

    // compute the box the particle should be placed into
    std::vector<Scalar>::iterator it;
    it = std::lower_bound(m_cum_frac_x.begin(), m_cum_frac_x.end(), f.x);
    unsigned ix = (it >= m_cum_frac_x.end()-1) ? (it - 1 - m_cum_frac_x.begin()) : 0;

    it = std::lower_bound(m_cum_frac_y.begin(), m_cum_frac_y.end(), f.y);
    unsigned iy = (it >= m_cum_frac_y.end()-1) ? (it - 1 - m_cum_frac_y.begin()) : 0;
    
    it = std::lower_bound(m_cum_frac_z.begin(), m_cum_frac_z.end(), f.z);
    unsigned iz = (it >= m_cum_frac_z.end()-1) ? (it - 1 - m_cum_frac_z.begin()) : 0;

    ArrayHandle<unsigned int> h_cart_ranks(m_cart_ranks, access_location::host, access_mode::read);
    unsigned int rank = h_cart_ranks.data[m_index(ix, iy, iz)];

    // synchronize with rank zero
    bcast(rank, 0, m_exec_conf->getMPICommunicator());
    return rank;
    }

//! Export BalancedDomainDecomposition class to python
void export_BalancedDomainDecomposition()
    {
    class_<BalancedDomainDecomposition, boost::shared_ptr<BalancedDomainDecomposition>, bases<DomainDecomposition>, boost::noncopyable>
        ("BalancedDomainDecomposition", init< boost::shared_ptr<ExecutionConfiguration>, Scalar3, const std::vector<Scalar>&, const std::vector<Scalar>&, const std::vector<Scalar>&>())
    .def("getFractions", &BalancedDomainDecomposition::getFractions)
    ;
    }

#endif
