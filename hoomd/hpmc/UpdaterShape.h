// Copyright (c) 2009-2022 The Regents of the University of Michigan.
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
                 std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                 std::shared_ptr<ShapeMoveBase<Shape>> move);

    ~UpdaterShape();

    virtual void update(uint64_t timestep);

    void initializeDeterminatsInertiaTensor();

    Scalar getTotalParticleVolume()
        {
        Scalar volume = 0.0;
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::read);
        for (unsigned int ndx = 0; ndx < m_ntypes.getNumElements(); ndx++)
            {
            detail::MassProperties<Shape> mp(m_mc->getParams()[ndx]);
            volume += mp.getVolume() * Scalar(h_ntypes.data[ndx]);
            }
        return volume;
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

    void resetStatistics()
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

    unsigned int getNselect()
        {
        return m_type_select;
        }

    void setNselect(unsigned int nselect)
        {
        if (nselect > m_pdata->getNTypes())
            {
            throw std::runtime_error("nselect must be less than or equal to the number of types");
            }
        m_type_select = nselect;
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

    unsigned int getNumPhase()
        {
        return m_num_phase;
        }

    void setNumPhase(unsigned int num_phase)
        {
        m_num_phase = num_phase;
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

    //! Get maximum displacement (by type name)
    inline Scalar getStepSize(std::string name)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);
        return m_step_size[id];
        }

    inline void setStepSize(std::string name, Scalar d)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);
        m_step_size[id] = d;
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
    unsigned int m_num_phase;   // number of phases in a multi-phase simulation
    int m_global_partition;     // number of MPI partitions
    GPUArray<Scalar>
        m_determinant_inertia_tensor; // determinant of the shape's moment of inertia tensor
    GPUArray<unsigned int> m_ntypes;  // number of particle types in the simulation
    bool m_initialized;               // whether or not the updater has been initialized
    std::vector<Scalar> m_step_size;  // shape move stepsize
    std::vector<unsigned int> m_count_accepted; // number of accepted updater moves
    std::vector<unsigned int> m_count_total;    // number of attempted updater moves
    std::vector<unsigned int>
        m_box_accepted; // number of accepted moves between boxes in multi-phase simulations
    std::vector<unsigned int>
        m_box_total; // number of attempted moves between boxes in multi-phase simulations
    detail::UpdateOrder m_update_order;      // order of particle types to apply the updater to
    unsigned int m_instance = 0;             //!< Unique ID for RNG seeding
    static constexpr Scalar m_tol = 0.00001; // minimum move size required to not be ignored.
    };

template<class Shape>
UpdaterShape<Shape>::UpdaterShape(std::shared_ptr<SystemDefinition> sysdef,
                                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                  std::shared_ptr<ShapeMoveBase<Shape>> move)
    : Updater(sysdef), m_mc(mc), m_move_function(move), m_global_partition(0),
      m_determinant_inertia_tensor(m_pdata->getNTypes(), m_exec_conf),
      m_ntypes(m_pdata->getNTypes(), m_exec_conf), m_initialized(false)
    {
    m_step_size.resize(m_pdata->getNTypes(), 0);
    m_count_accepted.resize(m_pdata->getNTypes(), 0);
    m_count_total.resize(m_pdata->getNTypes(), 0);
    m_box_accepted.resize(m_pdata->getNTypes(), 0);
    m_box_total.resize(m_pdata->getNTypes(), 0);
    initializeDeterminatsInertiaTensor();

    countParticlesPerType();

    // TODO: add a sanity check to makesure that MPI is setup correctly
    if (m_multi_phase)
        {
#ifdef ENABLE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &m_global_partition);
        assert(m_global_partition < 2);
#endif
        }
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
    m_update_order.shuffle(timestep, this->m_sysdef->getSeed(), m_exec_conf->getRank());
    for (unsigned int i_sweep = 0; i_sweep < m_nsweeps; i_sweep++)
        {
        Scalar log_boltz = 0.0;
        m_exec_conf->msg->notice(6) << "UpdaterShape copying data" << std::endl;
        GPUArray<Scalar> determinant_inertia_tensor_old(m_determinant_inertia_tensor);
        m_move_function->prepare(timestep);
        auto& mc_params = m_mc->getParams();
        for (unsigned int cur_type = 0; cur_type < m_type_select; cur_type++)
            {
            // make a trial move for i
            unsigned int typ_i = m_update_order[cur_type];
            // Skip move if step size is smaller than allowed tolerance
            if (m_step_size[typ_i] < m_tol)
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
                                         hoomd::Counter(typ_i, m_exec_conf->getRank(), i_sweep));

            // perform an in-place shape update on shape_param_new
            m_move_function->update_shape(timestep,
                                          m_step_size[typ_i],
                                          typ_i,
                                          shape_param_new,
                                          rng_i);

            // update det(I)
            detail::MassProperties<Shape> mp(shape_param_new);
            h_det.data[typ_i] = mp.getDetInertiaTensor();

            m_exec_conf->msg->notice(5) << " UpdaterShape I=" << h_det.data[typ_i] << ", "
                                        << h_det_old.data[typ_i] << std::endl;

            // consistency check
            assert(h_det.data[typ_i] != 0 && h_det_old.data[typ_i] != 0);

            // compute log_boltz factor
            log_boltz
                += (*m_move_function)(timestep,             // current timestep
                                      h_ntypes.data[typ_i], // number of particles of type typ_i,
                                      typ_i,                // the type id
                                      shape_param_new,      // new shape parameter
                                      h_det.data[typ_i],    // new determinant_inertia_tensor
                                      shape_param_old,      // old shape parameter
                                      h_det_old.data[typ_i] // old determinant_inertia_tensor
                );

            // actually update the shape parameter in the integrator
            m_mc->setParam(typ_i, shape_param_new);

            // looks redundant but it is not because of the pretend mode.
            bool accept = false, reject = true;
            Scalar p = hoomd::detail::generate_canonical<Scalar>(rng_i);
            Scalar Z = slow::exp(log_boltz);
            m_exec_conf->msg->notice(5) << " UpdaterShape p=" << p << ", z=" << Z << std::endl;

            if (m_multi_phase)
                {
#ifdef ENABLE_MPI
                std::vector<Scalar> Zs;
                all_gather_v(Z, Zs, MPI_COMM_WORLD);
                Z = std::accumulate(Zs.begin(), Zs.end(), 1, std::multiplies<Scalar>());
#endif
                }

            // potentially accept but need to check for overlaps first
            if (p < Z)
                {
                m_exec_conf->msg->notice(10)
                    << "HPMCMono count overlaps: " << timestep << std::endl;

                // This is a literal copy of the HPMC countOverlaps loop. The only difference here
                // is that we loop only over types instead of all the particles and always exit
                // early.

                // build an up to date AABB tree
                const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();
                // update the image list
                std::vector<vec3<Scalar>> image_list = m_mc->updateImageList();

                const Index2D& overlap_idx = m_mc->getOverlapIndexer();

                // access particle data and system box
                ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                               access_location::host,
                                               access_mode::read);
                ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                                   access_location::host,
                                                   access_mode::read);
                ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                                access_location::host,
                                                access_mode::read);

                // access parameters and interaction matrix
                ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(),
                                                     access_location::host,
                                                     access_mode::read);

                bool early_exit = true;
                unsigned int overlap_count = 0;
                unsigned int err_count = 0;
                // Loop over all the particles type whose shape we just updated
                // and check if they caused an overlap
                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    Scalar4 postype_i = h_postype.data[i];
                    // Only check if current particle i is of the type we just updated
                    if (typ_i == m_update_order[cur_type])
                        {
                        // read in the current position and orientation
                        Scalar4 orientation_i = h_orientation.data[i];
                        Shape shape_i(quat<Scalar>(orientation_i), m_mc->getParams()[typ_i]);
                        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
                        // Check particle against AABB tree for neighbors
                        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0, 0, 0));

                        const unsigned int n_images = (unsigned int)image_list.size();
                        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                            {
                            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
                            hoomd::detail::AABB aabb = aabb_i_local;
                            aabb.translate(pos_i_image);

                            // stackless search
                            for (unsigned int cur_node_idx = 0;
                                 cur_node_idx < aabb_tree.getNumNodes();
                                 cur_node_idx++)
                                {
                                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                                    {
                                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                                        {
                                        for (unsigned int cur_p = 0;
                                             cur_p < aabb_tree.getNodeNumParticles(cur_node_idx);
                                             cur_p++)
                                            {
                                            // read in its position and orientation
                                            unsigned int k
                                                = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                            // skip i==j in the 0 image
                                            if (cur_image == 0 && i == k)
                                                continue;

                                            Scalar4 postype_k = h_postype.data[k];
                                            Scalar4 orientation_k = h_orientation.data[k];

                                            // put particles in coordinate system of particle i
                                            vec3<Scalar> r_ik
                                                = vec3<Scalar>(postype_k) - pos_i_image;

                                            unsigned int typ_k = __scalar_as_int(postype_k.w);
                                            Shape shape_k(quat<Scalar>(orientation_k),
                                                          m_mc->getParams()[typ_k]);

                                            if (h_overlaps.data[overlap_idx(typ_i, typ_k)]
                                                && check_circumsphere_overlap(r_ik,
                                                                              shape_i,
                                                                              shape_k)
                                                && test_overlap(r_ik, shape_i, shape_k, err_count)
                                                && test_overlap(-r_ik, shape_k, shape_i, err_count))
                                                {
                                                overlap_count++;
                                                if (early_exit)
                                                    {
                                                    // exit early from loop over neighbor particles
                                                    break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                else
                                    {
                                    // skip ahead
                                    cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                                    }

                                if (overlap_count && early_exit)
                                    {
                                    break;
                                    }
                                } // end loop over AABB nodes

                            if (overlap_count && early_exit)
                                {
                                break;
                                }
                            } // end loop over images

                        if (overlap_count && early_exit)
                            {
                            break;
                            }
                        }
                    } // end loop over particles

#ifdef ENABLE_MPI
                if (this->m_pdata->getDomainDecomposition())
                    {
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &overlap_count,
                                  1,
                                  MPI_UNSIGNED,
                                  MPI_SUM,
                                  m_exec_conf->getMPICommunicator());
                    if (early_exit && overlap_count > 1)
                        overlap_count = 1;
                    }
#endif

                accept = !overlap_count;

                m_exec_conf->msg->notice(5)
                    << " UpdaterShape counted " << overlap_count << " overlaps" << std::endl;
                if (m_multi_phase)
                    {
#ifdef ENABLE_MPI
                    if (accept)
                        {
                        m_box_accepted[typ_i]++;
                        }
                    }
                std::vector<int> all_a;
                all_gather_v((int)accept, all_a, MPI_COMM_WORLD);
                accept = std::accumulate(all_a.begin(), all_a.end(), 1, std::multiplies<int>());
#endif
                }
            m_exec_conf->msg->notice(5) << " UpdaterShape p=" << p << ", z=" << Z << std::endl;
            }

        if (!accept) // categorically reject the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move retreating" << std::endl;
            m_move_function->retreat(timestep);
            }
        else if (m_pretend) // pretend to accept the move but actually reject it.
            {
            m_exec_conf->msg->notice(5)
                << " UpdaterShape move accepted -- pretend mode" << std::endl;
            m_move_function->retreat(timestep);
            m_count_accepted[typ_i]++;
            }
        else // actually accept the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move accepted" << std::endl;
            m_count_accepted[typ_i]++;
            reject = false;
            }

        if (reject)
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move rejected" << std::endl;
            // revert shape parameter changes
            h_det.data[typ_i] = h_det_old.data[typ_i];
            m_mc->setParam(typ_i, shape_param_old);
            }
        } // end loop over particle types
    }     // end loop over n_sweeps
m_exec_conf->msg->notice(4) << " UpdaterShape update done" << std::endl;
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
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<ShapeMoveBase<Shape>> >())
        .def("getShapeMovesCount", &UpdaterShape<Shape>::getShapeMovesCount)
        .def_property_readonly("total_particle_volume",
                               &UpdaterShape<Shape>::getTotalParticleVolume)
        .def("getShapeMoveEnergy", &UpdaterShape<Shape>::getShapeMoveEnergy)
        .def_property("shape_move",
                      &UpdaterShape<Shape>::getShapeMove,
                      &UpdaterShape<Shape>::setShapeMove)
        .def_property("pretend", &UpdaterShape<Shape>::getPretend, &UpdaterShape<Shape>::setPretend)
        .def_property("nselect", &UpdaterShape<Shape>::getNselect, &UpdaterShape<Shape>::setNselect)
        .def_property("nsweeps", &UpdaterShape<Shape>::getNsweeps, &UpdaterShape<Shape>::setNsweeps)
        .def_property("multi_phase",
                      &UpdaterShape<Shape>::getMultiPhase,
                      &UpdaterShape<Shape>::setMultiPhase)
        .def_property("num_phase",
                      &UpdaterShape<Shape>::getNumPhase,
                      &UpdaterShape<Shape>::setNumPhase)
        .def("getStepSize", &UpdaterShape<Shape>::getStepSize)
        .def("setStepSize", &UpdaterShape<Shape>::setStepSize);
    }

    } // namespace hpmc
} // namespace hoomd

#endif
