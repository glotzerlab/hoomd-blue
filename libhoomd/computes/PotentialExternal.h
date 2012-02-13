/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include "ForceCompute.h"

/*! \file PotentialExternal.h
    \brief Declares a class for computing an external force field
*/

#ifndef __POTENTIAL_EXTERNAL_H__
#define __POTENTIAL_EXTERNAL_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
template<class evaluator>
class PotentialExternal: public ForceCompute
    {
    public:
        //! Constructs the compute
        PotentialExternal<evaluator>(boost::shared_ptr<SystemDefinition> sysdef);

        //! type of external potential parameters
        typedef typename evaluator::param_type param_type;

        //! Sets parameters of the evaluator
        void setParams(unsigned int type, param_type params);

    protected:

        GPUArray<param_type> m_params;        //!< Array of per-type parameters

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator>
PotentialExternal<evaluator>::PotentialExternal(boost::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    GPUArray<param_type> params(m_pdata->getNTypes(), exec_conf);
    m_params.swap(params);
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator>
void PotentialExternal<evaluator>::computeForces(unsigned int timestep)
    {

    if (m_prof) m_prof->push("PotentialExternal");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;

    unsigned int nparticles = m_pdata->getN();

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    // for each of the particles
    for (unsigned int idx = 0; idx < nparticles; idx++)
        {
        // get the current particle properties
        Scalar3 X = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        Scalar3 F;
        Scalar energy;
        Scalar virial[6];

        param_type params = h_params.data[type];
        evaluator eval(X, Lx, Ly, Lz, params);
        eval.evalForceEnergyAndVirial(F, energy, virial);

        // apply the constraint force
        h_force.data[idx].x = F.x;
        h_force.data[idx].y = F.y;
        h_force.data[idx].z = F.z;
        h_force.data[idx].w = energy;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*m_virial_pitch+idx]  = virial[k];
        }


    if (m_prof)
        m_prof->pop();
    }

//! Set the parameters for this potential
/*! \param type type for which to set parameters
    \param params value of parameters
*/
template<class evaluator>
void PotentialExternal<evaluator>::setParams(unsigned int type, param_type params)
    {
    if (type >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set external potential params for a non existant type! "
                  << type << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialExternal");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = params;
    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternal class template.
*/
template < class T >
void export_PotentialExternal(const std::string& name)
    {
    boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition> >())
                  .def("setParams", &T::setParams)
                  ;
    }

#endif

