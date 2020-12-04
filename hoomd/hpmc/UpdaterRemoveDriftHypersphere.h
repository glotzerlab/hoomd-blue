// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Updater class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _REMOVE_DRIFT_UPDATER_HYPERSPHERE_H_
#define _REMOVE_DRIFT_UPDATER_HYPERSPHERE_H_

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
class RemoveDriftUpdaterHypersphere : public Updater
    {
    public:
        //! Constructor
        RemoveDriftUpdaterHypersphere( std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> > externalLattice,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > mc
                          ) : Updater(sysdef), m_externalLattice(externalLattice), m_mc(mc)
            {
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep)
            {
            ArrayHandle<Scalar4> h_quat_l(m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_quat_r(m_pdata->getRightQuaternionArray(), access_location::host, access_mode::readwrite);

            ArrayHandle<Scalar4> h_quat_l0(m_externalLattice->getReferenceLatticeQuat_l(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_quat_r0(m_externalLattice->getReferenceLatticeQuat_r(), access_location::host, access_mode::readwrite);

            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

            quat<Scalar> rshift_l;
            quat<Scalar> rshift_r;
            rshift_l.v.x=rshift_l.v.y=rshift_l.v.z=rshift_l.s=0.0f;
            rshift_r.v.x=rshift_r.v.y=rshift_r.v.z=rshift_r.s=0.0f;

            for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
                {
                unsigned int tag_i = h_tag.data[i];
                // read in the current position and orientation
                quat<Scalar> ql_i(h_quat_l.data[i]);
                quat<Scalar> qr_i(h_quat_r.data[i]);
                quat<Scalar> pos_i = ql_i*qr_i;

                quat<Scalar> ref_l(h_quat_l0.data[tag_i]);
                quat<Scalar> ref_r(h_quat_r0.data[tag_i]);
                quat<Scalar> ref = ref_l*ref_r + pos_i;

                ref *= fast::rsqrt(norm2(ref));
                pos_i = conj(pos_i);

                rshift_l += ref*pos_i;
                rshift_r += pos_i*ref;
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                Scalar r_l[4] = {rshift_l.s, rshift_l.v.x, rshift_l.v.y, rshift_l.v.z};
                MPI_Allreduce(MPI_IN_PLACE, &r_l[0], 4, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                rshift_l.s = r_l[0];
                rshift_l.v.x = r_l[1];
                rshift_l.v.y = r_l[2];
                rshift_l.v.z = r_l[3];

                Scalar r_r[4] = {rshift_r.s, rshift_r.v.x, rshift_r.v.y, rshift_r.v.z};
                MPI_Allreduce(MPI_IN_PLACE, &r_r[0], 4, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
                rshift_r.s = r_r[0];
                rshift_r.v.x = r_r[1];
                rshift_r.v.y = r_r[2];
                rshift_r.v.z = r_r[3];
                }
            #endif

            Scalar norm_l_inv = fast::rsqrt(norm2(rshift_l));
            rshift_l *= norm_l_inv;
            Scalar norm_r_inv = fast::rsqrt(norm2(rshift_r));
            rshift_r *= norm_r_inv;

            for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                quat<Scalar> ql_i(h_quat_l.data[i]);
                quat<Scalar> qr_i(h_quat_r.data[i]);

                ql_i = rshift_l*ql_i;
                norm_l_inv = fast::rsqrt(norm2(ql_i));
                ql_i *= norm_l_inv;

                qr_i = qr_i*rshift_r;
                Scalar norm_r_inv = fast::rsqrt(norm2(qr_i));
                qr_i *= norm_r_inv;

                h_quat_l.data[i] = quat_to_scalar4(ql_i);
                h_quat_r.data[i] = quat_to_scalar4(qr_i);
                }

            m_mc->invalidateAABBTree();
            // migrate and exchange particles
            m_mc->communicate(true);

            }
    protected:
                std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> > m_externalLattice;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;
    };

//! Export the ExampleUpdater class to python
template <class Shape>
void export_RemoveDriftUpdaterHypersphere(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<RemoveDriftUpdaterHypersphere<Shape>, std::shared_ptr<RemoveDriftUpdaterHypersphere<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> >,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > >())
    ;
    }
}

#endif // _REMOVE_DRIFT_UPDATER_HYPERSPHERE_H_
