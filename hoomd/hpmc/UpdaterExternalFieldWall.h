// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _UPDATER_EXTERNAL_FIELD_H_
#define _UPDATER_EXTERNAL_FIELD_H_

/*! \file UpdaterExternalField.h
    \brief Updates ExternalField base class
*/
#include "hoomd/Updater.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RNGIdentifiers.h"

#include "IntegratorHPMCMono.h"
#include "ExternalField.h"
#include "ExternalFieldWall.h" // do we need anything else?

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

template< class Shape >
class __attribute__ ((visibility ("hidden"))) UpdaterExternalFieldWall : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator object
            \param py_updater Python call back for wall update. Actually UPDATES the wall.
            \param move_ratio Probability of attempting wall update move
            \param seed PRNG seed
        */
        UpdaterExternalFieldWall( std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                      std::shared_ptr< ExternalFieldWall<Shape> > external,
                      pybind11::object py_updater,
                      Scalar move_ratio,
                      unsigned int seed) : Updater(sysdef), m_mc(mc), m_external(external), m_py_updater(py_updater), m_move_ratio(move_ratio), m_seed(seed)
                      {
                      // broadcast the seed from rank 0 to all other ranks.
                      #ifdef ENABLE_MPI
                          if(this->m_pdata->getDomainDecomposition())
                              bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
                      #endif

                      // set m_count_total, m_count_accepted equal to zero
                      m_count_total_rel = 0;
                      m_count_total_tot = 0;
                      m_count_accepted_rel = 0;
                      m_count_accepted_tot = 0;

                      // get copies of the external field's current walls
                      m_CurrSpheres = m_external->GetSphereWalls();
                      m_CurrCylinders = m_external->GetCylinderWalls();
                      m_CurrPlanes = m_external->GetPlaneWalls();
                      }

        //! Destructor
        virtual ~UpdaterExternalFieldWall(){}

        //! Sets parameters
        /*! \param move_ratio Probability of attempting external field update move
        */
        void setMoveRatio(Scalar move_ratio)
            {
            m_move_ratio = move_ratio;
            };

        //! Get move_ratio parameter
        /*! \returns move_ratio parameter
        */
        Scalar getMoveRatio()
            {
            return m_move_ratio;
            }

        //! Print statistics
        void printStats()
            {
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities()
            {
            // start with the updater provided quantities
            std::vector< std::string > result = Updater::getProvidedLogQuantities();

            // then add ours
            result.push_back("hpmc_wall_acceptance_ratio");
            return result;
            }

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if (quantity == "hpmc_wall_acceptance_ratio")
                {
                return m_count_total_rel ? Scalar(m_count_accepted_rel)/Scalar(m_count_total_rel) : 0;
                }
            else
                {
                    m_exec_conf->msg->error() << "update.wall: " << quantity << " is not a valid log quantity" << std::endl;
                    throw std::runtime_error("Error getting log value");
                }
            }

        //! Reset statistics counters
        void resetStats()
            {
            m_count_total_rel = 0;
            m_count_accepted_rel = 0;
            }

        //! Get accepted count
        //! If mode!=0, return absolute quantities. If mode=0, return quantities relative to start of run.
        Scalar getAcceptedCount(unsigned int mode)
            {
            if (mode == int(0)) { return m_count_accepted_rel; }
            else { return m_count_accepted_tot; }
            }

        //! Get total count
        //! If mode!=0, return absolute quantities. If mode=0, return quantities relative to start of run.
        Scalar getTotalCount(unsigned int mode)
            {
            if (mode == int(0)) { return m_count_total_rel; }
            else { return m_count_total_tot; }
            }

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(unsigned int timestep)
            {
            // Choose whether or not to update the external field
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterExternalFieldWall, m_seed, timestep);
            unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
            unsigned int move_ratio = m_move_ratio * 65536;
            // Attempt and evaluate a move
            if (move_type_select < move_ratio)
                {
                m_count_total_rel++;
                m_count_total_tot++;

                // update the current copies of the walls
                m_CurrSpheres = m_external->GetSphereWalls();
                m_CurrCylinders = m_external->GetCylinderWalls();
                m_CurrPlanes = m_external->GetPlaneWalls();

                // call back to python to update the external field
                m_py_updater(timestep);

                // the only thing that changed was the external field,
                // not particle positions or orientations. so all we need to do is
                // make sure our update didn't result in infinite energy (any overlaps).
                unsigned int boltz = m_external->calculateBoltzmannWeight(timestep);

                if( boltz != 0 )
                    {
                    m_count_accepted_rel++;
                    m_count_accepted_tot++;
                    }
                else
                    {
                    // restore the current copies of the walls
                    m_external->SetSphereWalls(m_CurrSpheres);
                    m_external->SetCylinderWalls(m_CurrCylinders);
                    m_external->SetPlaneWalls(m_CurrPlanes);
                    }
                }
            }

    private:
        std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc;      //!< Integrator
        std::shared_ptr< ExternalFieldWall<Shape> > m_external; //!< External field wall object
        pybind11::object m_py_updater;                       //!< Python call back for external field update
        Scalar m_move_ratio;                                      //!< Ratio of lattice vector length versus shearing move
        unsigned int m_count_accepted_rel;                        //!< Accepted moves count, relative to start of run
        unsigned int m_count_total_rel;                           //!< Accept/reject total count, relative to start of run
        unsigned int m_count_accepted_tot;                        //!< Accepted moves count, TOTAL
        unsigned int m_count_total_tot;                           //!< Accept/reject total count, TOTAL
        unsigned int m_seed;                                      //!< Seed for pseudo-random number generator
        std::vector<SphereWall> m_CurrSpheres;                    //!< Copy of current sphere walls
        std::vector<CylinderWall> m_CurrCylinders;                //!< Copy of current cylinder walls
        std::vector<PlaneWall> m_CurrPlanes;                      //!< Copy of current plane walls
    };

template< class Shape >
void export_UpdaterExternalFieldWall(pybind11::module& m, std::string name)
    {
   pybind11::class_< UpdaterExternalFieldWall<Shape>, std::shared_ptr< UpdaterExternalFieldWall<Shape> > >(m, name.c_str(), pybind11::base< Updater >())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr< IntegratorHPMCMono<Shape> >, std::shared_ptr< ExternalFieldWall<Shape> >, pybind11::object, Scalar, unsigned int >())
    .def("getAcceptedCount", &UpdaterExternalFieldWall<Shape>::getAcceptedCount)
    .def("getTotalCount", &UpdaterExternalFieldWall<Shape>::getTotalCount)
    .def("resetStats", &UpdaterExternalFieldWall<Shape>::resetStats)
    ;
    }
} // namespace

#endif // inclusion guard
