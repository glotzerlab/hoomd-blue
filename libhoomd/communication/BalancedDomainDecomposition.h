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

/*! \file BalancedDomainDecomposition.h
    \brief Defines the BalancedDomainDecomposition class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __BALANCED_DOMAIN_DECOMPOSITION_H__
#define __BALANCED_DOMAIN_DECOMPOSITION_H__

#include "DomainDecomposition.h"
#include <vector>

/*! \ingroup communication
*/

//! Class that initializes every processor using spatial domain-decomposition with balanced domain size
/*!
 * This class is used to divide the global simulation box into sub-domains and to assign a box to every processor.
 *
 * <b>Implementation details</b>
 *
 * Unlike the standard DomainDecomposition, the BalancedDomainDecomposition allows the user to specify the fractional
 * volume of each domain cut along the Cartesian axes. This is advantageous for simulations with non-homogeneous
 * particle distributions, e.g., a vapor-liquid interface. The specified fractions must (a) create a grid commensurate
 * with the number of ranks and (b) sum to 1.0 so as to cover the entire simulation box. If the specified number of
 * ranks does not match the number that is available, behavior is reverted to the DomainDecomposition default with
 * uniform cuts along each dimension. An error is raised if condition (b) is not satisfied.
 *
 * The initialization of the domain decomposition scheme is performed in the constructor.
 */
class BalancedDomainDecomposition : public DomainDecomposition
    {
#ifdef ENABLE_MPI
    public:
        //! Constructor
        /*!
         * \param exec_conf The execution configuration
         * \param L Box lengths of global box to sub-divide
         * \param fxs Array of fractions to decompose box in x
         * \param fys Array of fractions to decompose box in y
         * \param fzs Array of fractions to decompose box in z
         */
        BalancedDomainDecomposition(boost::shared_ptr<ExecutionConfiguration> exec_conf,
                                    Scalar3 L,
                                    const std::vector<Scalar>& fxs,
                                    const std::vector<Scalar>& fys,
                                    const std::vector<Scalar>& fzs);


        //! Get the box fractions along each dimension
        std::vector<Scalar> getFractions(unsigned int dir) const
            {
            if (dir == 0) return m_frac_x;
            else if (dir == 1) return m_frac_y;
            else if (dir == 2) return m_frac_z;
            else
                {
                m_exec_conf->msg->error() << "comm: requested direction does not exist" << std::endl;
                throw std::runtime_error("comm: requested direction does not exist");
                }
            }
        
        //! Get the cumulative box fractions along each dimension
        std::vector<Scalar> getCumulativeFractions(unsigned int dir) const
            {
            if (dir == 0) return m_cum_frac_x;
            else if (dir == 1) return m_cum_frac_y;
            else if (dir == 2) return m_cum_frac_z;
            else
                {
                m_exec_conf->msg->error() << "comm: requested direction does not exist" << std::endl;
                throw std::runtime_error("comm: requested direction does not exist");
                }
            }
        
        //! Set the cumulative fractions along a dimension
        void setCumulativeFractions(unsigned int dir, const std::vector<Scalar>& cum_frac, unsigned int root);

        //! Get the dimensions of the local simulation box
        virtual const BoxDim calculateLocalBox(const BoxDim& global_box);

        //! Get the rank for a particle to be placed
        /*!
         * \param global_box Global simulation box
         * \param pos Particle position
         * \returns the rank of the processor that should receive the particle
         */
        virtual unsigned int placeParticle(const BoxDim& global_box, Scalar3 pos);

    protected:
        std::vector<Scalar> m_frac_x;       //!< Fractional divisions in x per cut plane
        std::vector<Scalar> m_frac_y;       //!< Fractional divisions in y per cut plane
        std::vector<Scalar> m_frac_z;       //!< Fractional divisions in z per cut plane
        
        std::vector<Scalar> m_cum_frac_x;   //!< Cumulative fractions in x below cut plane index
        std::vector<Scalar> m_cum_frac_y;   //!< Cumulative fractions in y below cut plane index
        std::vector<Scalar> m_cum_frac_z;   //!< Cumulative fractions in z below cut plane index

        bool m_uniform;                     //!< Flag to fall back to uniform domain decomposition if true
#endif // ENABLE_MPI
    };

#ifdef ENABLE_MPI
//! Export the balanced domain decomposition information
void export_BalancedDomainDecomposition();
#endif

#endif // __BALANCED_DOMAIN_DECOMPOSITION_H__
