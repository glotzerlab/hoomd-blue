// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Updater class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _REMOVE_DRIFT_UPDATER_H_
#define _REMOVE_DRIFT_UPDATER_H_

/*! \file ExampleUpdater.h
    \brief Declaration of ExampleUpdater
*/

// First, hoomd.h should be included

#include "hoomd/Updater.h"
#include "ExternalFieldLattice.h"
#include "IntegratorHPMCMono.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc {
// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND ONLY IF
// hoomd_config.h is included first)
// For example:
//
// #include "hoomd/Updater.h"

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a template here, there are
// no restrictions on what a template can do

//! A nonsense particle updater written to demonstrate how to write a plugin
/*! This updater simply sets all of the particle's velocities to 0 when update() is called.
*/
template<class Shape>
class RemoveDriftUpdater : public Updater
    {
    public:
        //! Constructor
        RemoveDriftUpdater( std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ExternalFieldLattice<Shape> > externalLattice,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > mc
                          ) : Updater(sysdef), m_externalLattice(externalLattice), m_mc(mc)
            {
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep)
            {
            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar3> h_r0(m_externalLattice->getReferenceLatticePositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);
            const BoxDim& box = this->m_pdata->getGlobalBox();
            vec3<Scalar> origin(this->m_pdata->getOrigin());
            vec3<Scalar> rshift;
            rshift.x=rshift.y=rshift.z=0.0f;

            for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
                {
                unsigned int tag_i = h_tag.data[i];
                // read in the current position and orientation
                vec3<Scalar> postype_i = vec3<Scalar>(h_postype.data[i]) - origin;
                int3 tmp_image = make_int3(0, 0, 0);
                box.wrap(postype_i, tmp_image);
                vec3<Scalar> dr = postype_i - vec3<Scalar>(h_r0.data[tag_i]);
                rshift += vec3<Scalar>(box.minImage(vec_to_scalar3(dr)));
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                Scalar r[3] = {rshift.x, rshift.y, rshift.z};
                MPI_Allreduce(MPI_IN_PLACE, &r[0], 3, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                rshift.x = r[0];
                rshift.y = r[1];
                rshift.z = r[2];
                }
            #endif

            rshift/=Scalar(this->m_pdata->getNGlobal());

            for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                vec3<Scalar> r_i = vec3<Scalar>(postype_i);
                h_postype.data[i] = vec_to_scalar4(r_i - rshift, postype_i.w);
                box.wrap(h_postype.data[i], h_image.data[i]);
                }

            m_mc->invalidateAABBTree();
            // migrate and exchange particles
            m_mc->communicate(true);

            }
    protected:
                std::shared_ptr<ExternalFieldLattice<Shape> > m_externalLattice;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;
    };

//! Export the ExampleUpdater class to python
template <class Shape>
void export_RemoveDriftUpdater(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<RemoveDriftUpdater<Shape>, std::shared_ptr<RemoveDriftUpdater<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLattice<Shape> >,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > >())
    ;
    }
}

#endif // _REMOVE_DRIFT_UPDATER_H_
