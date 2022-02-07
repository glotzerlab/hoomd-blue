// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_EXTERNAL_FIELD_H_
#define _UPDATER_EXTERNAL_FIELD_H_

/*! \file UpdaterExternalField.h
    \brief Updates ExternalField base class
*/
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/Updater.h"
#include "hoomd/VectorMath.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h" // do we need anything else?
#include "IntegratorHPMCMono.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
template<class Shape>
class __attribute__((visibility("hidden"))) UpdaterExternalFieldWall : public Updater
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param mc HPMC integrator object
        \param py_updater Python call back for wall update. Actually UPDATES the wall.
        \param move_probability Probability of attempting wall update move
        \param seed PRNG seed
    */
    UpdaterExternalFieldWall(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                             std::shared_ptr<ExternalFieldWall<Shape>> external,
                             pybind11::object py_updater,
                             Scalar move_probability,
                             unsigned int seed)
        : Updater(sysdef), m_mc(mc), m_external(external), m_py_updater(py_updater),
          m_move_probability(move_probability), m_seed(seed)
        {
// broadcast the seed from rank 0 to all other ranks.
#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
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
    virtual ~UpdaterExternalFieldWall() { }

    //! Sets parameters
    /*! \param move_probability Probability of attempting external field update move
     */
    void setMoveRatio(Scalar move_probability)
        {
        m_move_probability = move_probability;
        };

    //! Get move_probability parameter
    /*! \returns move_probability parameter
     */
    Scalar getMoveRatio()
        {
        return m_move_probability;
        }

    //! Reset statistics counters
    void resetStats()
        {
        m_count_total_rel = 0;
        m_count_accepted_rel = 0;
        }

    //! Get accepted count
    //! If mode!=0, return absolute quantities. If mode=0, return quantities relative to start of
    //! run.
    Scalar getAcceptedCount(unsigned int mode)
        {
        if (mode == int(0))
            {
            return m_count_accepted_rel;
            }
        else
            {
            return m_count_accepted_tot;
            }
        }

    //! Get total count
    //! If mode!=0, return absolute quantities. If mode=0, return quantities relative to start of
    //! run.
    Scalar getTotalCount(unsigned int mode)
        {
        if (mode == int(0))
            {
            return m_count_total_rel;
            }
        else
            {
            return m_count_total_tot;
            }
        }

    /// Set the RNG instance
    void setInstance(unsigned int instance)
        {
        m_instance = instance;
        }

    /// Get the RNG instance
    unsigned int getInstance()
        {
        return m_instance;
        }

    //! Take one timestep forward
    /*! \param timestep timestep at which update is being evaluated
     */
    virtual void update(uint64_t timestep)
        {
        // Choose whether or not to update the external field
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterExternalFieldWall,
                                               timestep,
                                               m_sysdef->getSeed()),
                                   hoomd::Counter(m_instance));
        unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
        unsigned int move_probability = (unsigned int)(m_move_probability * 65536);
        // Attempt and evaluate a move
        if (move_type_select < move_probability)
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
            Scalar boltz = m_external->calculateBoltzmannWeight(timestep);

            if (boltz != 0.0)
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
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;      //!< Integrator
    std::shared_ptr<ExternalFieldWall<Shape>> m_external; //!< External field wall object
    pybind11::object m_py_updater;         //!< Python call back for external field update
    Scalar m_move_probability;             //!< Ratio of lattice vector length versus shearing move
    unsigned int m_count_accepted_rel;     //!< Accepted moves count, relative to start of run
    unsigned int m_count_total_rel;        //!< Accept/reject total count, relative to start of run
    unsigned int m_count_accepted_tot;     //!< Accepted moves count, TOTAL
    unsigned int m_count_total_tot;        //!< Accept/reject total count, TOTAL
    unsigned int m_seed;                   //!< Seed for pseudo-random number generator
    std::vector<SphereWall> m_CurrSpheres; //!< Copy of current sphere walls
    std::vector<CylinderWall> m_CurrCylinders; //!< Copy of current cylinder walls
    std::vector<PlaneWall> m_CurrPlanes;       //!< Copy of current plane walls

    unsigned int m_instance = 0; //!< Unique ID for RNG seeding
    };

namespace detail
    {
template<class Shape> void export_UpdaterExternalFieldWall(pybind11::module& m, std::string name)
    {
    pybind11::class_<UpdaterExternalFieldWall<Shape>,
                     Updater,
                     std::shared_ptr<UpdaterExternalFieldWall<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<ExternalFieldWall<Shape>>,
                            pybind11::object,
                            Scalar,
                            unsigned int>())
        .def("getAcceptedCount", &UpdaterExternalFieldWall<Shape>::getAcceptedCount)
        .def("getTotalCount", &UpdaterExternalFieldWall<Shape>::getTotalCount)
        .def("resetStats", &UpdaterExternalFieldWall<Shape>::resetStats)
        .def_property("instance",
                      &UpdaterExternalFieldWall<Shape>::getInstance,
                      &UpdaterExternalFieldWall<Shape>::setInstance);
    }
    } // end namespace detail
    } // namespace hpmc
    } // namespace hoomd

#endif // inclusion guard
