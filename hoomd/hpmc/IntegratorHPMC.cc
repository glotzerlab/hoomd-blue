// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegratorHPMC.h"

#include "hoomd/VectorMath.h"
#include <sstream>

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::hpmc::PairPotential>>);

using namespace std;

/*! \file IntegratorHPMC.cc
    \brief Definition of common methods for HPMC integrators
*/

namespace hoomd
    {
namespace hpmc
    {
IntegratorHPMC::IntegratorHPMC(std::shared_ptr<SystemDefinition> sysdef)
    : Integrator(sysdef, 0.005), m_translation_move_probability(32768), m_nselect(4),
      m_nominal_width(1.0), m_extra_ghost_width(0), m_external_base(NULL), m_past_first_run(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMC" << endl;

    GlobalArray<hpmc_counters_t> counters(1, this->m_exec_conf);
    m_count_total.swap(counters);

    GPUVector<Scalar> d(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d.swap(d);

    GPUVector<Scalar> a(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_a.swap(a);

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::overwrite);
    // set default values
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        h_d.data[typ] = 0.1;
        h_a.data[typ] = 0.1;
        }

    resetStats();

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        assert(m_comm);

        m_comm->getGhostLayerWidthRequestSignal()
            .connect<IntegratorHPMC, &IntegratorHPMC::getGhostLayerWidth>(this);

        m_comm->getCommFlagsRequestSignal().connect<IntegratorHPMC, &IntegratorHPMC::getCommFlags>(
            this);
        }
#endif
    }

IntegratorHPMC::~IntegratorHPMC()
    {
    m_exec_conf->msg->notice(5) << "Destroying IntegratorHPMC" << endl;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getGhostLayerWidthRequestSignal()
            .disconnect<IntegratorHPMC, &IntegratorHPMC::getGhostLayerWidth>(this);
        m_comm->getCommFlagsRequestSignal()
            .disconnect<IntegratorHPMC, &IntegratorHPMC::getCommFlags>(this);
        }
#endif
    }

/*! \returns True if the particle orientations are normalized
 */
bool IntegratorHPMC::checkParticleOrientations()
    {
    // get the orientations data array
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    bool result = true;

    // loop through particles and return false if any is out of norm
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        quat<Scalar> o(h_orientation.data[i]);
        if (fabs(Scalar(1.0) - norm2(o)) > 1e-3)
            {
            m_exec_conf->msg->notice(2)
                << "Particle " << h_tag.data[i] << " has an unnormalized orientation" << endl;
            result = false;
            }
        }

#ifdef ENABLE_MPI
    unsigned int result_int = (unsigned int)result;
    unsigned int result_reduced;
    MPI_Reduce(&result_int,
               &result_reduced,
               1,
               MPI_UNSIGNED,
               MPI_LOR,
               0,
               m_exec_conf->getMPICommunicator());
    result = bool(result_reduced);
#endif

    return result;
    }

/*! Set new box with particle positions scaled from previous box
    and check for overlaps

    \param newBox new box dimensions

    \note The particle positions and the box dimensions are updated in any case, even if the
    new box dimensions result in overlaps. To restore old particle positions,
    they have to be backed up before calling this method.

    \returns false if resize results in overlaps
*/
bool IntegratorHPMC::attemptBoxResize(uint64_t timestep, const BoxDim& new_box)
    {
    unsigned int N = m_pdata->getN();

    // Get old and new boxes;
    BoxDim curBox = m_pdata->getGlobalBox();

        // Use lexical scope block to make sure ArrayHandles get cleaned up
        {
        // Get particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        // move the particles to be inside the new box
        for (unsigned int i = 0; i < N; i++)
            {
            Scalar3 old_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

            // obtain scaled coordinates in the old global box
            Scalar3 f = curBox.makeFraction(old_pos);

            // scale particles
            Scalar3 scaled_pos = new_box.makeCoordinates(f);
            h_pos.data[i].x = scaled_pos.x;
            h_pos.data[i].y = scaled_pos.y;
            h_pos.data[i].z = scaled_pos.z;
            }
        } // end lexical scope

    m_pdata->setGlobalBox(new_box);

    // scale the origin
    Scalar3 old_origin = m_pdata->getOrigin();
    Scalar3 fractional_old_origin = curBox.makeFraction(old_origin);
    Scalar3 new_origin = new_box.makeCoordinates(fractional_old_origin);
    m_pdata->translateOrigin(new_origin - old_origin);

    // we have moved particles, communicate those changes
    this->communicate(false);

    // check overlaps
    return !this->countOverlaps(true);
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the
   last executed step \return The current state of the acceptance counters

    IntegratorHPMC maintains a count of the number of accepted and rejected moves since
   instantiation. getCounters() provides the current value. The parameter *mode* controls whether
   the returned counts are absolute, relative to the start of the run, or relative to the start of
   the last executed step.
*/
hpmc_counters_t IntegratorHPMC::getCounters(unsigned int mode)
    {
    ArrayHandle<hpmc_counters_t> h_counters(m_count_total,
                                            access_location::host,
                                            access_mode::read);
    hpmc_counters_t result;

    if (mode == 0)
        result = h_counters.data[0];
    else if (mode == 1)
        result = h_counters.data[0] - m_count_run_start;
    else
        result = h_counters.data[0] - m_count_step_start;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // MPI Reduction to total result values on all nodes.
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.translate_accept_count,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.translate_reject_count,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.rotate_accept_count,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.rotate_reject_count,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.overlap_checks,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.overlap_err_count,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

namespace detail
    {
void export_IntegratorHPMC(pybind11::module& m)
    {
    pybind11::class_<hpmc::PatchEnergy, Autotuned, std::shared_ptr<hpmc::PatchEnergy>>(
        m,
        "PatchEnergy")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());

    pybind11::class_<IntegratorHPMC, Integrator, std::shared_ptr<IntegratorHPMC>>(m,
                                                                                  "IntegratorHPMC")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setD", &IntegratorHPMC::setD)
        .def("setA", &IntegratorHPMC::setA)
        .def("setTranslationMoveProbability", &IntegratorHPMC::setTranslationMoveProbability)
        .def("setNSelect", &IntegratorHPMC::setNSelect)
        .def("getD", &IntegratorHPMC::getD)
        .def("getA", &IntegratorHPMC::getA)
        .def("getTranslationMoveProbability", &IntegratorHPMC::getTranslationMoveProbability)
        .def("getNSelect", &IntegratorHPMC::getNSelect)
        .def("getMaxCoreDiameter", &IntegratorHPMC::getMaxCoreDiameter)
        .def("countOverlaps", &IntegratorHPMC::countOverlaps)
        .def("checkParticleOrientations", &IntegratorHPMC::checkParticleOrientations)
        .def("getMPS", &IntegratorHPMC::getMPS)
        .def("getCounters", &IntegratorHPMC::getCounters)
        .def("communicate", &IntegratorHPMC::communicate)
        .def_property("nselect", &IntegratorHPMC::getNSelect, &IntegratorHPMC::setNSelect)
        .def_property("translation_move_probability",
                      &IntegratorHPMC::getTranslationMoveProbability,
                      &IntegratorHPMC::setTranslationMoveProbability)
        .def_property_readonly("pair_potentials", &IntegratorHPMC::getPairPotentials)
        .def("computeTotalPairEnergy", &IntegratorHPMC::computeTotalPairEnergy)
        .def_property_readonly("external_potentials", &IntegratorHPMC::getExternalPotentials)
        .def("computeTotalExternalEnergy", &IntegratorHPMC::computeTotalExternalEnergy);

    pybind11::class_<hpmc_counters_t>(m, "hpmc_counters_t")
        .def_readonly("overlap_checks", &hpmc_counters_t::overlap_checks)
        .def_readonly("overlap_errors", &hpmc_counters_t::overlap_err_count)
        .def_property_readonly("translate", &hpmc_counters_t::getTranslateCounts)
        .def_property_readonly("rotate", &hpmc_counters_t::getRotateCounts);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
