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
#include "hoomd/RNGIdentifiers.h"

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
                            std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                            unsigned int seed
                          ) : Updater(sysdef), m_externalLattice(externalLattice), m_mc(mc), m_seed(seed)
            {

            unsigned int MaxN = m_pdata->getMaxN();
            GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_quat_l_backup);
            GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_quat_r_backup);
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep)
            {

            //Scalar energy_old = m_externalLattice->getLogValue("lattice_energy",timestep);

            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterRemoveDrift, m_seed, 0, m_exec_conf->getRank(), timestep);

            unsigned int N_backup = m_pdata->getN();
            ArrayHandle<Scalar4> h_quat_l(this->m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_quat_r(this->m_pdata->getRightQuaternionArray(), access_location::host, access_mode::readwrite);
            {
                ArrayHandle<Scalar4> h_quat_l_backup(m_quat_l_backup, access_location::host, access_mode::overwrite);
                memcpy(h_quat_l_backup.data, h_quat_l.data, sizeof(Scalar4) * N_backup);

                ArrayHandle<Scalar4> h_quat_r_backup(m_quat_r_backup, access_location::host, access_mode::overwrite);
                memcpy(h_quat_r_backup.data, h_quat_r.data, sizeof(Scalar4) * N_backup);
            }

            const Hypersphere& hypersphere = m_pdata->getHypersphere();

            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

            ArrayHandle<Scalar> h_d(m_mc->getDArray(), access_location::host, access_mode::read);

            Scalar maxD = 0;

            for (unsigned int i = 0; i < N_backup; i++)
                {
                    Scalar4 postype_i = h_postype.data[i];
                    int typ_i = __scalar_as_int(postype_i.w);
                    maxD += h_d.data[typ_i];
                }

            maxD /= N_backup;

            hoomd::UniformDistribution<Scalar> uniform(Scalar(-1.0),Scalar(1.0));

            Scalar phi = maxD*uniform(rng_i)/hypersphere.getR();

            //! Generate a direction (3d unit vector) for the translation
            Scalar theta = hoomd::UniformDistribution<Scalar>(Scalar(0.0),Scalar(2.0*M_PI))(rng_i);
            Scalar z = uniform(rng_i);
            vec3<Scalar> b = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // the transformation quaternion
            quat<Scalar> rshift(fast::cos(0.5*phi),fast::sin(0.5*phi)*b);
            Scalar norm_l_inv = fast::rsqrt(norm2(rshift));
            rshift *= norm_l_inv;

            ArrayHandle<Scalar4> h_quat_l_backup(m_quat_l_backup, access_location::host, access_mode::readwrite);

            ArrayHandle<Scalar4> h_quat_r_backup(m_quat_r_backup, access_location::host, access_mode::readwrite);

            for (unsigned int i = 0; i < N_backup; i++)
                {
                // read in the current position and orientation
                quat<Scalar> ql_i(h_quat_l_backup.data[i]);
                quat<Scalar> qr_i(h_quat_r_backup.data[i]);

                ql_i = rshift*ql_i;
                norm_l_inv = fast::rsqrt(norm2(ql_i));
                ql_i *= norm_l_inv;

                qr_i = qr_i*rshift;
                Scalar norm_r_inv = fast::rsqrt(norm2(qr_i));
                qr_i *= norm_r_inv;

                h_quat_l_backup.data[i] = quat_to_scalar4(ql_i);
                h_quat_r_backup.data[i] = quat_to_scalar4(qr_i);
                }


            Scalar deltaE = -m_externalLattice->calculateDeltaEHypersphere(h_quat_l_backup.data, h_quat_r_backup.data, &hypersphere);

            double p = hoomd::detail::generate_canonical<double>(rng_i);

            if (p < fast::exp(-deltaE))
                {
                std::cout << "Accepted" << std::endl;
                for (unsigned int i = 0; i < N_backup; i++)
                    {
                    // read in the current position and orientation
                    h_quat_l.data[i] = h_quat_l_backup.data[i];
                    h_quat_r.data[i] = h_quat_r_backup.data[i];
                    }

                }

            m_mc->invalidateAABBTree();
            // migrate and exchange particles
            m_mc->communicate(true);

            }
    protected:
                std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> > m_externalLattice;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;

                unsigned int m_seed;
                GPUArray<Scalar4> m_quat_l_backup;             //!< hold backup copy of particle positions
                GPUArray<Scalar4> m_quat_r_backup;             //!< hold backup copy of particle positions
    };

//! Export the ExampleUpdater class to python
template <class Shape>
void export_RemoveDriftUpdaterHypersphere(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<RemoveDriftUpdaterHypersphere<Shape>, std::shared_ptr<RemoveDriftUpdaterHypersphere<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> >,
                            std::shared_ptr<IntegratorHPMCMono<Shape> >,
                            unsigned int >())
    ;
    }
}

#endif // _REMOVE_DRIFT_UPDATER_HYPERSPHERE_H_
