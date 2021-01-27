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
class RemoveDriftUpdater : public Updater
    {
    public:
        //! Constructor
        RemoveDriftUpdater( std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ExternalFieldLattice<Shape> > externalLattice,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                            unsigned int seed
                          ) : Updater(sysdef), m_externalLattice(externalLattice), m_mc(mc), m_seed(seed)
            {
            unsigned int MaxN = m_pdata->getMaxN();
            GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_position_backup);
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep)
            {


            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterRemoveDrift, m_seed, 0, m_exec_conf->getRank(), timestep);

            unsigned int N_backup = m_pdata->getN();
            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            {
                ArrayHandle<Scalar4> h_postype_backup(m_position_backup, access_location::host, access_mode::overwrite);
                memcpy(h_postype_backup.data, h_postype.data, sizeof(Scalar4) * N_backup);
            }
            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);
            const BoxDim& box = this->m_pdata->getGlobalBox();
            vec3<Scalar> origin(this->m_pdata->getOrigin());

            ArrayHandle<Scalar> h_d(m_mc->getDArray(), access_location::host, access_mode::read);

            Scalar maxD = 0;

            for (unsigned int i = 0; i < N_backup; i++)
                {
                    Scalar4 postype_i = h_postype.data[i];
                    int typ_i = __scalar_as_int(postype_i.w);
                    maxD += h_d.data[typ_i];
                }

            maxD /= N_backup;

    	    hoomd::UniformDistribution<Scalar> uniform(-maxD, maxD);

    	    // Generate a random vector inside a sphere of radius d
    	    vec3<Scalar> rshift(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    	    do
    	        {
    	        rshift.x = uniform(rng_i);
    	        rshift.y = uniform(rng_i);
    	        rshift.z = uniform(rng_i);
    	        } while(dot(rshift,rshift) > maxD*maxD);


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

    	    // apply the move vector
    	    //
            ArrayHandle<Scalar4> h_postype_backup(m_position_backup, access_location::host, access_mode::readwrite);


            for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                Scalar4 postype_i = h_postype_backup.data[i];
                vec3<Scalar> r_i = vec3<Scalar>(postype_i);
                h_postype_backup.data[i] = vec_to_scalar4(r_i - rshift, postype_i.w);
                }

            Scalar deltaE = m_externalLattice->calculateDeltaE(h_postype_backup.data, NULL, &box);//here new and old is switched for calculation. Hence, missing "-" sign!

            double p = hoomd::detail::generate_canonical<double>(rng_i);

            if (p < fast::exp(deltaE))
                {
                for (unsigned int i = 0; i < N_backup; i++)
                    {
                    // read in the current position and orientation
                    h_postype.data[i] = h_postype_backup.data[i];
                    box.wrap(h_postype.data[i], h_image.data[i]);
                    }
		}

            m_mc->invalidateAABBTree();
            // migrate and exchange particles
            m_mc->communicate(true);

            }
    protected:
                std::shared_ptr<ExternalFieldLattice<Shape> > m_externalLattice;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;

                unsigned int m_seed;
                GPUArray<Scalar4> m_position_backup;             //!< hold backup copy of particle positions
    };

//! Export the ExampleUpdater class to python
template <class Shape>
void export_RemoveDriftUpdater(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<RemoveDriftUpdater<Shape>, std::shared_ptr<RemoveDriftUpdater<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLattice<Shape> >,
                            std::shared_ptr<IntegratorHPMCMono<Shape> >,
                            unsigned int >())
    ;
    }
}

#endif // _REMOVE_DRIFT_UPDATER_H_
