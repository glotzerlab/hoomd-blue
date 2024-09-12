// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_SHAPE_H
#define _UPDATER_SHAPE_H

#include "IntegratorHPMCMono.h"
#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/Updater.h"
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd
    {

namespace hpmc
    {

template<typename Shape> class UpdaterShape : public Updater
    {
    public:
    UpdaterShape(std::shared_ptr<SystemDefinition> sysdef,
                 std::shared_ptr<Trigger> trigger,
                 std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                 std::shared_ptr<ShapeMoveBase<Shape>> move);

    ~UpdaterShape();

    virtual void update(uint64_t timestep);

    void initializeDeterminatsInertiaTensor();

    std::vector<Scalar> getParticleVolumes()
        {
        std::vector<Scalar> volumes;
        for (unsigned int type = 0; type < m_ntypes.getNumElements(); type++)
            {
            detail::MassProperties<Shape> mp(m_mc->getParams()[type]);
            volumes.emplace_back(mp.getVolume());
            }
        return volumes;
        }

    Scalar getShapeMoveEnergy(uint64_t timestep)
        {
        Scalar energy = 0.0;
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_det(m_determinant_inertia_tensor,
                                  access_location::host,
                                  access_mode::readwrite);
        for (unsigned int ndx = 0; ndx < m_ntypes.getNumElements(); ndx++)
            {
            energy += m_move_function->computeEnergy(timestep,
                                                     h_ntypes.data[ndx],
                                                     ndx,
                                                     m_mc->getParams()[ndx],
                                                     h_det.data[ndx]);
            }
        return energy;
        }

    std::pair<unsigned int, unsigned int> getShapeMovesCount()
        {
        unsigned int total_accepted_count = getAcceptedCount();
        unsigned int total_rejected_count = getTotalCount() - total_accepted_count;
        return std::make_pair(total_accepted_count, total_rejected_count);
        }

    unsigned int getAcceptedCount()
        {
        unsigned int total_accepted = 0;
        for (unsigned int ndx = 0; ndx < m_ntypes.getNumElements(); ndx++)
            {
            total_accepted += m_count_accepted[ndx];
            }
        return total_accepted;
        }

    unsigned int getTotalCount()
        {
        unsigned int total_count = 0;
        for (unsigned int ndx = 0; ndx < m_ntypes.getNumElements(); ndx++)
            {
            total_count += m_count_total[ndx];
            }
        return total_count;
        }

    unsigned int getAcceptedBox(unsigned int ndx)
        {
        return m_box_accepted[ndx];
        }

    unsigned int getTotalBox(unsigned int ndx)
        {
        return m_box_total[ndx];
        }

    void resetStats()
        {
        std::fill(m_count_accepted.begin(), m_count_accepted.end(), 0);
        std::fill(m_count_total.begin(), m_count_total.end(), 0);
        std::fill(m_box_accepted.begin(), m_box_accepted.end(), 0);
        std::fill(m_box_total.begin(), m_box_total.end(), 0);
        }

    void setShapeMove(std::shared_ptr<ShapeMoveBase<Shape>> move);

    std::shared_ptr<ShapeMoveBase<Shape>> getShapeMove();

    bool getPretend()
        {
        return m_pretend;
        }

    void setPretend(bool pretend)
        {
        m_pretend = pretend;
        }

    unsigned int getTypeSelect()
        {
        return m_type_select;
        }

    void setTypeSelect(unsigned int type_select)
        {
        if (type_select > m_pdata->getNTypes())
            {
            throw std::runtime_error(
                "type_select must be less than or equal to the number of types");
            }
        m_type_select = type_select;
        }

    unsigned int getNsweeps()
        {
        return m_nsweeps;
        }

    void setNsweeps(unsigned int nsweeps)
        {
        m_nsweeps = nsweeps;
        }

    bool getMultiPhase()
        {
        return m_multi_phase;
        }

    void setMultiPhase(bool multi_phase)
        {
        m_multi_phase = multi_phase;
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

    void countParticlesPerType();

    private:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; // hpmc particle integrator
    std::shared_ptr<ShapeMoveBase<Shape>>
        m_move_function;        // shape move function to apply in the updater
    unsigned int m_type_select; // number of particle types to update in each move
    unsigned int m_nsweeps;     // number of sweeps to run the updater each time it is called
    bool m_pretend;             // whether or not to pretend or actually perform shape move
    bool m_multi_phase;         // whether or not the simulation is multi-phase
    GPUArray<Scalar>
        m_determinant_inertia_tensor; // determinant of the shape's moment of inertia tensor
    GPUArray<unsigned int> m_ntypes;  // number of particle types in the simulation
    bool m_initialized;               // whether or not the updater has been initialized
    std::vector<unsigned int> m_count_accepted; // number of accepted updater moves
    std::vector<unsigned int> m_count_total;    // number of attempted updater moves
    std::vector<unsigned int>
        m_box_accepted; // number of accepted moves between boxes in multi-phase simulations
    std::vector<unsigned int>
        m_box_total; // number of attempted moves between boxes in multi-phase simulations
    detail::UpdateOrder m_update_order; // order of particle types to apply the updater to
    unsigned int m_instance = 0;        //!< Unique ID for RNG seeding
    };

template<class Shape>
UpdaterShape<Shape>::UpdaterShape(std::shared_ptr<SystemDefinition> sysdef,
                                  std::shared_ptr<Trigger> trigger,
                                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                  std::shared_ptr<ShapeMoveBase<Shape>> move)
    : Updater(sysdef, trigger), m_mc(mc), m_move_function(move), m_type_select(1), m_nsweeps(1),
      m_pretend(false), m_multi_phase(false),
      m_determinant_inertia_tensor(m_pdata->getNTypes(), m_exec_conf),
      m_ntypes(m_pdata->getNTypes(), m_exec_conf), m_initialized(false)
    {
    m_count_accepted.resize(m_pdata->getNTypes(), 0);
    m_count_total.resize(m_pdata->getNTypes(), 0);
    m_box_accepted.resize(m_pdata->getNTypes(), 0);
    m_box_total.resize(m_pdata->getNTypes(), 0);
    initializeDeterminatsInertiaTensor();
    countParticlesPerType();

#ifdef ENABLE_MPI
    if (m_multi_phase)
        {
        assert(m_exec_conf->getNRanks() < 2);
        }
#endif
    }

template<class Shape> UpdaterShape<Shape>::~UpdaterShape()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterShape " << std::endl;
    }

/*! Perform Metropolis Monte Carlo shape deformations
\param timestep Current time step of the simulation
*/
template<class Shape> void UpdaterShape<Shape>::update(uint64_t timestep)
    {
    m_exec_conf->msg->notice(4) << "UpdaterShape update: " << timestep
                                << ", initialized: " << std::boolalpha << m_initialized << " @ "
                                << std::hex << this << std::dec << std::endl;
    if (!m_initialized)
        initializeDeterminatsInertiaTensor();

    // Shuffle the order of particles types for this step
    m_update_order.resize(m_pdata->getNTypes());
    m_update_order.shuffle(timestep,
                           this->m_sysdef->getSeed(),
                           RNGIdentifier::HPMCShapeMoveUpdateOrder);

    Scalar log_boltz;
    for (unsigned int i_sweep = 0; i_sweep < m_nsweeps; i_sweep++)
        {
        m_exec_conf->msg->notice(6) << "UpdaterShape copying data" << std::endl;
        GPUArray<Scalar> determinant_inertia_tensor_old(m_determinant_inertia_tensor);
        m_move_function->prepare(timestep);
        auto& mc_params = m_mc->getParams();

        for (unsigned int cur_type = 0; cur_type < m_type_select; cur_type++)
            {
            // make a trial move for i
            unsigned int typ_i = m_update_order[cur_type];
            // Skip move if step size is zero
            if (m_move_function->getStepSize(m_pdata->getNameByType(typ_i)) == 0)
                {
                m_exec_conf->msg->notice(5) << " Skipping moves for particle typeid=" << typ_i
                                            << ", " << cur_type << std::endl;
                continue;
                }
            else
                {
                m_exec_conf->msg->notice(5)
                    << " UpdaterShape making trial move for typeid=" << typ_i << ", " << cur_type
                    << std::endl;
                }

            m_count_total[typ_i]++;

            // cache old shape parameters before it is modified
            auto shape_param_old = typename Shape::param_type(mc_params[typ_i]);
            auto shape_param_new = typename Shape::param_type(shape_param_old);

            ArrayHandle<Scalar> h_det(m_determinant_inertia_tensor,
                                      access_location::host,
                                      access_mode::readwrite);
            ArrayHandle<Scalar> h_det_old(determinant_inertia_tensor_old,
                                          access_location::host,
                                          access_mode::readwrite);
            ArrayHandle<unsigned int> h_ntypes(m_ntypes,
                                               access_location::host,
                                               access_mode::readwrite);

            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::UpdaterShapeConstruct,
                                                     timestep,
                                                     this->m_sysdef->getSeed()),
                                         hoomd::Counter(typ_i, 0, i_sweep));

            // perform an in-place shape update on shape_param_new
            m_move_function->update_shape(timestep,
                                          typ_i,
                                          shape_param_new,
                                          rng_i,
                                          m_exec_conf->isCUDAEnabled());

            // update det(I)
            detail::MassProperties<Shape> mp(shape_param_new);
            h_det.data[typ_i] = mp.getDetInertiaTensor();

            m_exec_conf->msg->notice(5) << " UpdaterShape I=" << h_det.data[typ_i] << ", "
                                        << h_det_old.data[typ_i] << std::endl;

            // consistency check
            assert(h_det.data[typ_i] != 0 && h_det_old.data[typ_i] != 0);

            // compute log_boltz factor
            log_boltz = m_move_function->computeLogBoltzmann(
                timestep,             // current timestep
                h_ntypes.data[typ_i], // number of particles of type typ_i,
                typ_i,                // the type id
                shape_param_new,      // new shape parameter
                h_det.data[typ_i],    // new determinant_inertia_tensor
                shape_param_old,      // old shape parameter
                h_det_old.data[typ_i] // old determinant_inertia_tensor
            );

            // actually update the shape parameter in the integrator
            m_mc->setParam(typ_i, shape_param_new);

            // check if at least one overlap was caused
            bool overlaps = static_cast<bool>(m_mc->countOverlaps(true));
            // automatically reject if there are overlaps
            if (overlaps)
                {
                m_exec_conf->msg->notice(5)
                    << "UpdaterShape move rejected -- overlaps found" << std::endl;
                // revert shape parameter changes
                m_move_function->retreat(timestep, typ_i);
                h_det.data[typ_i] = h_det_old.data[typ_i];
                m_mc->setParam(typ_i, shape_param_old);
                }
            else // potentially accept, check Metropolis first
                {
                Scalar p = hoomd::detail::generate_canonical<Scalar>(rng_i);
                Scalar Z = slow::exp(log_boltz);

#ifdef ENABLE_MPI
                if (m_multi_phase)
                    {
                    std::vector<Scalar> Zs;
                    all_gather_v(Z, Zs, m_exec_conf->getHOOMDWorldMPICommunicator());
                    Z = std::accumulate(Zs.begin(), Zs.end(), 1, std::multiplies<Scalar>());
                    }
#endif

                bool accept = p < Z;

                if (accept)
                    {
#ifdef ENABLE_MPI
                    if (m_multi_phase)
                        {
                        m_box_accepted[typ_i]++;
                        }
                    std::vector<int> all_a;
                    all_gather_v((int)accept, all_a, m_exec_conf->getHOOMDWorldMPICommunicator());
                    accept = std::accumulate(all_a.begin(), all_a.end(), 1, std::multiplies<int>());
#endif

                    // if pretend mode, revert shape changes but keep acceptance count
                    if (m_pretend)
                        {
                        m_exec_conf->msg->notice(5)
                            << "UpdaterShape move accepted -- pretend mode" << std::endl;
                        m_move_function->retreat(timestep, typ_i);
                        h_det.data[typ_i] = h_det_old.data[typ_i];
                        m_mc->setParam(typ_i, shape_param_old);
                        m_count_accepted[typ_i]++;
                        }
                    else // actually accept the move
                        {
                        m_exec_conf->msg->notice(5) << "UpdaterShape move accepted" << std::endl;
                        m_count_accepted[typ_i]++;
                        }
                    }
                else // actually reject move
                    {
                    m_exec_conf->msg->notice(5) << " UpdaterShape move rejected" << std::endl;
                    // revert shape parameter changes
                    h_det.data[typ_i] = h_det_old.data[typ_i];
                    m_mc->setParam(typ_i, shape_param_old);
                    m_move_function->retreat(timestep, typ_i);
                    }
                }

            } // end loop over particle types
        } // end loop over n_sweeps
    m_exec_conf->msg->notice(4) << "UpdaterShape update done" << std::endl;
    } // end UpdaterShape<Shape>::update(unsigned int timestep)

template<typename Shape> void UpdaterShape<Shape>::initializeDeterminatsInertiaTensor()
    {
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_det(m_determinant_inertia_tensor,
                              access_location::host,
                              access_mode::readwrite);
    auto& params = m_mc->getParams();
    for (unsigned int i = 0; i < m_pdata->getNTypes(); i++)
        {
        detail::MassProperties<Shape> mp(params[i]);
        h_det.data[i] = mp.getDetInertiaTensor();
        }
    m_initialized = true;
    }

template<typename Shape>
void UpdaterShape<Shape>::setShapeMove(std::shared_ptr<ShapeMoveBase<Shape>> move)
    {
    m_move_function = move;
    }

template<typename Shape> std::shared_ptr<ShapeMoveBase<Shape>> UpdaterShape<Shape>::getShapeMove()
    {
    return m_move_function;
    }

template<typename Shape> void UpdaterShape<Shape>::countParticlesPerType()
    {
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        int typ_j = __scalar_as_int(h_postype.data[j].w);
        h_ntypes.data[typ_j]++;
        }

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      h_ntypes.data,
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    }

template<typename Shape> void export_UpdaterShape(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<UpdaterShape<Shape>, Updater, std::shared_ptr<UpdaterShape<Shape>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<ShapeMoveBase<Shape>>>())
        .def("getShapeMovesCount", &UpdaterShape<Shape>::getShapeMovesCount)
        .def_property_readonly("particle_volumes", &UpdaterShape<Shape>::getParticleVolumes)
        .def("getShapeMoveEnergy", &UpdaterShape<Shape>::getShapeMoveEnergy)
        .def_property("shape_move",
                      &UpdaterShape<Shape>::getShapeMove,
                      &UpdaterShape<Shape>::setShapeMove)
        .def_property("pretend", &UpdaterShape<Shape>::getPretend, &UpdaterShape<Shape>::setPretend)
        .def_property("type_select",
                      &UpdaterShape<Shape>::getTypeSelect,
                      &UpdaterShape<Shape>::setTypeSelect)
        .def_property("nsweeps", &UpdaterShape<Shape>::getNsweeps, &UpdaterShape<Shape>::setNsweeps)
        .def_property("multi_phase",
                      &UpdaterShape<Shape>::getMultiPhase,
                      &UpdaterShape<Shape>::setMultiPhase);
    }

    } // namespace hpmc
    } // namespace hoomd

#endif
