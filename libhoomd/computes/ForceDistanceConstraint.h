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

// Maintainer: jglaser

#include "ForceConstraint.h"

/*! \file ForceDistanceConstraint.h
    \brief Declares a class to implement pairwise distance constraint
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ForceDistanceConstraint_H__
#define __ForceDistanceConstraint_H__

#include "GPUVector.h"

/*! Implements a pairwise distance constraint using the algorithm of

    [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, “A Non-Iterative Matrix Method for Constraint Molecular Dynamics Simulations,” Mol. Simul., vol. 13, no. 6, pp. 395–405, 1994.
    [2] M. Yoneya, “A Generalized Non-iterative Matrix Method for Constraint Molecular Dynamics Simulations,” J. Comput. Phys., vol. 172, no. 1, pp. 188–197, Sep. 2001.

    See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class ForceDistanceConstraint : public ForceConstraint
    {
    public:
        //! Constructs the compute
        ForceDistanceConstraint(boost::shared_ptr<SystemDefinition> sysdef);

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved()
            {
            return m_cdata->getNGlobal();
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif


    protected:
        boost::shared_ptr<ConstraintData> m_cdata; //! The constraint data

        GPUVector<double> m_cmatrix;                //!< The matrix for the constraint force equation (column-major)
        GPUVector<double> m_cvec;                   //!< The vector on the RHS of the constraint equation
        GPUVector<double> m_lagrange;               //!< The solution for the lagrange multipliers

        //! Compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Populate the quantities in the constraint-force equatino
        virtual void fillMatrixVector(unsigned int timestep);

        //! Solve the linear matrix-vector equation
        virtual void computeConstraintForces(unsigned int timestep);
    };

//! Exports the ForceDistanceConstraint to python
void export_ForceDistanceConstraint();

#endif
