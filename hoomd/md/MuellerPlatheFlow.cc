// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MuellerPlatheFlow.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/HOOMDMath.h"

using namespace std;

//! \file MuellerPlatheFlow.cc Implementation of CPU version of MuellerPlatheFlow.

namespace hoomd
    {
namespace md
    {
const unsigned int INVALID_TAG = UINT_MAX;
const Scalar INVALID_VEL = FLT_MAX; // should be ok, even for double.

MuellerPlatheFlow::MuellerPlatheFlow(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<Trigger> trigger,
                                     std::shared_ptr<ParticleGroup> group,
                                     std::shared_ptr<Variant> flow_target,
                                     std::string slab_direction_str,
                                     std::string flow_direction_str,
                                     const unsigned int N_slabs,
                                     const unsigned int min_slab,
                                     const unsigned int max_slab,
                                     Scalar flow_epsilon)
    : Updater(sysdef, trigger), m_group(group), m_flow_target(flow_target),
      m_flow_epsilon(flow_epsilon), m_N_slabs(N_slabs), m_min_slab(min_slab), m_max_slab(max_slab),
      m_exchanged_momentum(0), m_has_min_slab(true), m_has_max_slab(true),
      m_needs_orthorhombic_check(true)
    {
    assert(m_flow_target);

    m_flow_direction = this->getDirectionFromString(flow_direction_str);
    m_slab_direction = this->getDirectionFromString(slab_direction_str);

    if (slab_direction_str == flow_direction_str)
        {
        m_exec_conf->msg->warning()
            << " MuellerPlatheFlow setup with "
               "slab and flow direction to be equal. This is no shear flow. "
               "The method has not been developed for this application."
            << endl;
        }

    m_pdata->getBoxChangeSignal()
        .connect<MuellerPlatheFlow, &MuellerPlatheFlow::forceOrthorhombicBoxCheck>(this);

    m_last_max_vel.x = m_last_max_vel.y = -INVALID_VEL;
    m_last_max_vel.z = __int_as_scalar(INVALID_TAG);
    m_last_min_vel.x = m_last_min_vel.y = INVALID_VEL;
    m_last_min_vel.z = __int_as_scalar(INVALID_TAG);

    m_exec_conf->msg->notice(5) << "Constructing MuellerPlatheFlow " << endl;
    this->updateDomainDecomposition();
    // m_exec_conf->msg->notice(0)<<m_exec_conf->getRank()<<": "<< m_max_swap.gbl_rank<<"
    // "<<m_min_swap.gbl_rank<<endl;

    // Check min max slab.
    this->setMinSlab(m_min_slab);
    this->setMaxSlab(m_max_slab);
    }

MuellerPlatheFlow::~MuellerPlatheFlow(void)
    {
    m_exec_conf->msg->notice(5) << "Destroying MuellerPlatheFlow " << endl;
    m_pdata->getBoxChangeSignal()
        .disconnect<MuellerPlatheFlow, &MuellerPlatheFlow::forceOrthorhombicBoxCheck>(this);
    }

void MuellerPlatheFlow::update(uint64_t timestep)
    {
    Updater::update(timestep);
    if (m_needs_orthorhombic_check)
        this->verifyOrthorhombicBox();

    const BoxDim& box = m_pdata->getGlobalBox();
    double area = 1.; // Init to shut up compiler warning
    switch (m_slab_direction)
        {
    case flow_enum::X:
        area = box.getL().y * box.getL().z;
    case flow_enum::Y:
        area = box.getL().x * box.getL().z;
    case flow_enum::Z:
        area = box.getL().y * box.getL().x;
        }

    // Determine switch direction for this update call
    // Switch outside while loop, to prevent oscillations around the target.
    bool bigger_swap_needed
        = (*m_flow_target)(timestep) > this->getSummedExchangedMomentum() / area;
    bigger_swap_needed &= this->getMinSlab() > this->getMaxSlab();
    bool smaller_swap_needed
        = (*m_flow_target)(timestep) < this->getSummedExchangedMomentum() / area;
    smaller_swap_needed &= this->getMinSlab() < this->getMaxSlab();

    if ((bigger_swap_needed || smaller_swap_needed)
        && (fabs((*m_flow_target)(timestep) - this->getSummedExchangedMomentum() / area)
            > this->getFlowEpsilon()))
        {
        this->swapMinMaxSlab();
        }
    // Sign for summed exchanged momentum depends on hierarchy of min and max slab.
    const int sign = this->getMaxSlab() > this->getMinSlab() ? 1 : -1;

    unsigned int counter = 0;
    const unsigned int max_iteration = 100;
    while (fabs((*m_flow_target)(timestep) - this->getSummedExchangedMomentum() / area)
               > this->getFlowEpsilon()
           && counter < max_iteration)
        {
        counter++;

        m_last_max_vel.x = m_last_max_vel.y = -INVALID_VEL;
        m_last_max_vel.z = __int_as_scalar(INVALID_TAG);
        m_last_min_vel.x = m_last_min_vel.y = INVALID_VEL;
        m_last_min_vel.z = __int_as_scalar(INVALID_TAG);
        searchMinMaxVelocity();

#ifdef ENABLE_MPI
        mpiExchangeVelocity();
#endif // ENABLE_MPI

        if (m_last_max_vel.x == -INVALID_VEL
            || static_cast<unsigned int>(__scalar_as_int(m_last_max_vel.z)) == INVALID_TAG
            || m_last_min_vel.x == -INVALID_VEL
            || static_cast<unsigned int>(__scalar_as_int(m_last_min_vel.z)) == INVALID_TAG)
            {
            m_exec_conf->msg->warning()
                << "WARNING: at time " << timestep
                << "  MuellerPlatheFlow could not find a min/max pair." << endl;
            // Increase the counter to stop the while loop.
            counter = max_iteration;
            }
        else
            {
            updateMinMaxVelocity();

            m_exchanged_momentum += sign * (m_last_max_vel.x - m_last_min_vel.x);
            }
        }
    if (counter >= max_iteration)
        {
        m_exec_conf->msg->warning()
            << " After " << counter
            << " MuellerPlatheFlow could not achieve the target: " << (*m_flow_target)(timestep)
            << " only " << this->getSummedExchangedMomentum() / area << " could be achieved."
            << endl;
        }
    // stringstream s;
    // s<<this->getSummedExchangedMomentum()/area<<"\t"<<m_flow_target->getValue(timestep)<<endl;
    // m_exec_conf->msg->collectiveNoticeStr(0,s.str());
    }

void MuellerPlatheFlow::swapMinMaxSlab(void)
    {
    std::swap(m_max_slab, m_min_slab);

    std::swap(m_last_max_vel, m_last_min_vel);

    std::swap(m_has_max_slab, m_has_min_slab);

#ifdef ENABLE_MPI
    std::swap(m_max_swap, m_min_swap);
#endif // ENABLE_MPI
    m_exec_conf->msg->notice(4) << "MuellerPlatheUpdater swapped min/max slab: "
                                << this->getMinSlab() << " " << this->getMaxSlab() << endl;
    }

void MuellerPlatheFlow::setMinSlab(const unsigned int min_slab)
    {
    if (min_slab >= m_N_slabs)
        {
        ostringstream s;
        s << "MuellerPlatheFlow is initialized with invalid min_slab: " << min_slab << "/"
          << m_N_slabs << ".";
        throw runtime_error(s.str());
        }
    if (min_slab != m_min_slab)
        this->updateDomainDecomposition();
    }

void MuellerPlatheFlow::setMaxSlab(const unsigned int max_slab)
    {
    if (max_slab >= m_N_slabs)
        {
        ostringstream s;
        s << "MuellerPlatheFlow is initialized with invalid max_slab: " << max_slab << "/"
          << m_N_slabs << ".";
        throw runtime_error(s.str());
        }
    if (max_slab != m_max_slab)
        this->updateDomainDecomposition();
    }

void MuellerPlatheFlow::updateDomainDecomposition(void)
    {
#ifdef ENABLE_MPI
    std::shared_ptr<DomainDecomposition> dec = m_pdata->getDomainDecomposition();
    if (dec)
        {
        const Scalar min_frac = m_min_slab / static_cast<Scalar>(m_N_slabs);
        const Scalar max_frac = m_max_slab / static_cast<Scalar>(m_N_slabs);

        const uint3 grid = dec->getGridSize();
        const uint3 pos = dec->getGridPos();
        const unsigned int my_grid = m_slab_direction == flow_enum::X
                                         ? grid.x
                                         : (m_slab_direction == flow_enum::Y ? grid.y : grid.z);
        const unsigned int my_pos = m_slab_direction == flow_enum::X
                                        ? pos.x
                                        : (m_slab_direction == flow_enum::Y ? pos.y : pos.z);
        if (m_N_slabs % my_grid != 0)
            {
            m_exec_conf->msg->warning() << "MuellerPlatheFlow::N_slabs does is divisible "
                                           " by the domain decomposition. Adjusting N_slabs.\n"
                                        << endl;
            m_N_slabs += m_N_slabs % my_grid;
            m_min_slab = static_cast<unsigned int>(min_frac * m_N_slabs);
            m_max_slab = static_cast<unsigned int>(max_frac * m_N_slabs);
            }

        m_has_min_slab = false;
        if (my_pos == this->getMinSlab() / (m_N_slabs / my_grid))
            m_has_min_slab = true;

        m_has_max_slab = false;
        if (my_pos == this->getMaxSlab() / (m_N_slabs / my_grid))
            m_has_max_slab = true;

        // Create the communicator.
        const int min_color = this->hasMinSlab() ? 0 : MPI_UNDEFINED;
        initMPISwap(&m_min_swap, min_color);

        const int max_color = this->hasMaxSlab() ? 0 : MPI_UNDEFINED;
        initMPISwap(&m_max_swap, max_color);
        }
#endif // ENABLE_MPI
    }

void MuellerPlatheFlow::searchMinMaxVelocity(void)
    {
    const unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    if (!this->hasMaxSlab() and !this->hasMinSlab())
        return;
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    const BoxDim& gl_box = m_pdata->getGlobalBox();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        const unsigned int j = m_group->getMemberIndex(group_idx);
        if (j < m_pdata->getN())
            {
            unsigned int index = 0; // Init to shut up compiler warning
            switch (m_slab_direction)
                {
            case flow_enum::X:
                index = (unsigned int)(((h_pos.data[j].x) / gl_box.getL().x + .5)
                                       * this->getNSlabs());
                break;
            case flow_enum::Y:
                index = (unsigned int)(((h_pos.data[j].y) / gl_box.getL().y + .5)
                                       * this->getNSlabs());
                break;
            case flow_enum::Z:
                index = (unsigned int)(((h_pos.data[j].z) / gl_box.getL().z + .5)
                                       * this->getNSlabs());
                break;
                }
            index %= this->getNSlabs(); // border cases. wrap periodic box
            assert(index >= 0);
            assert(index < this->getNSlabs());
            if (index == this->getMaxSlab() || index == this->getMinSlab())
                {
                Scalar vel = 0; // Init to shut up compiler warning
                switch (m_flow_direction)
                    {
                case flow_enum::X:
                    vel = h_vel.data[j].x;
                    break;
                case flow_enum::Y:
                    vel = h_vel.data[j].y;
                    break;
                case flow_enum::Z:
                    vel = h_vel.data[j].z;
                    break;
                    }
                const Scalar mass = h_vel.data[j].w;
                vel *= mass; // Use momentum instead of velocity
                if (index == this->getMaxSlab() && m_last_max_vel.x < vel && this->hasMaxSlab())
                    {
                    m_last_max_vel.x = vel;
                    m_last_max_vel.y = mass;
                    m_last_max_vel.z = __int_as_scalar(h_tag.data[j]);
                    }
                if (index == this->getMinSlab() && m_last_min_vel.x > vel && this->hasMinSlab())
                    {
                    m_last_min_vel.x = vel;
                    m_last_max_vel.y = mass;
                    m_last_min_vel.z = __int_as_scalar(h_tag.data[j]);
                    }
                }
            }
        }
    }

void MuellerPlatheFlow::updateMinMaxVelocity(void)
    {
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    const unsigned int min_tag = __scalar_as_int(m_last_min_vel.z);
    const unsigned int min_idx = h_rtag.data[min_tag];
    const unsigned int max_tag = __scalar_as_int(m_last_max_vel.z);
    const unsigned int max_idx = h_rtag.data[max_tag];
    const unsigned int Ntotal = m_pdata->getN() + m_pdata->getNGhosts();
    // Is my particle local on the processor?
    if (min_idx < Ntotal || max_idx < Ntotal)
        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        // Swap the particles the new velocities.
        if (min_idx < Ntotal)
            {
            const Scalar new_min_vel = m_last_max_vel.x / m_last_min_vel.y;
            switch (m_flow_direction)
                {
            case flow_enum::X:
                h_vel.data[min_idx].x = new_min_vel;
                break;
            case flow_enum::Y:
                h_vel.data[min_idx].y = new_min_vel;
                break;
            case flow_enum::Z:
                h_vel.data[min_idx].z = new_min_vel;
                break;
                }
            }
        if (max_idx < Ntotal)
            {
            const Scalar new_max_vel = m_last_min_vel.x / m_last_max_vel.y;
            switch (m_flow_direction)
                {
            case flow_enum::X:
                h_vel.data[max_idx].x = new_max_vel;
                break;
            case flow_enum::Y:
                h_vel.data[max_idx].y = new_max_vel;
                break;
            case flow_enum::Z:
                h_vel.data[max_idx].z = new_max_vel;
                break;
                }
            }
        }
    }

void MuellerPlatheFlow::verifyOrthorhombicBox(void)
    {
    bool valid = true;
    const BoxDim box = m_pdata->getGlobalBox();
    valid &= fabs(box.getTiltFactorXY()) < 1e-7;
    valid &= fabs(box.getTiltFactorXZ()) < 1e-7;
    valid &= fabs(box.getTiltFactorYZ()) < 1e-7;

    if (not valid)
        {
        throw runtime_error("MuellerPlatheFlow can only be used with orthorhombic boxes.");
        }
    // Disable check for the next update call.
    m_needs_orthorhombic_check = false;
    }
#ifdef ENABLE_MPI

// Not performance optimized: could be slow. It is meant for init.
void MuellerPlatheFlow::initMPISwap(struct MPI_SWAP* ms, const int color)
    {
    if (ms->initialized && ms->rank != MPI_UNDEFINED)
        MPI_Comm_free(&(ms->comm));

    MPI_Comm_split(m_exec_conf->getMPICommunicator(), color, m_exec_conf->getRank(), &(ms->comm));
    if (color != MPI_UNDEFINED)
        {
        MPI_Comm_rank(ms->comm, &(ms->rank));
        MPI_Comm_size(ms->comm, &(ms->size));
        }
    else
        {
        ms->rank = MPI_UNDEFINED;
        ms->size = MPI_UNDEFINED;
        }
    int send = 0;
    if (ms->rank == 0)
        {
        send = 1;
        }
    vector<int> recv;
    all_gather_v(send, recv, m_exec_conf->getMPICommunicator());

    // ARG_MAX;
    ms->gbl_rank = -1;
    for (unsigned int i = 0; i < recv.size(); i++)
        {
        if (recv[i] == 1)
            {
            assert(ms->gbl_rank == -1);
            ms->gbl_rank = i;
            }
        }
    assert(ms->gbl_rank > -1);
    ms->initialized = true;
    }

void MuellerPlatheFlow::bcastVelToAll(struct MPI_SWAP* ms, Scalar3* vel, const MPI_Op op)
    {
    if (ms->rank != MPI_UNDEFINED)
        {
        Scalar_Int tmp;
        tmp.s = vel->x;
        tmp.i = ms->rank;
        MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_HOOMD_SCALAR_INT, op, ms->comm);
        // Send to subset local rank
        if (tmp.i != 0)
            {
            if (ms->rank == 0)
                {
                recv(vel->x, tmp.i, ms->comm);
                recv(vel->y, tmp.i, ms->comm);
                recv(vel->z, tmp.i, ms->comm);
                }
            if (ms->rank == tmp.i)
                {
                send(vel->x, 0, ms->comm);
                send(vel->y, 0, ms->comm);
                send(vel->z, 0, ms->comm);
                }
            }
        // ms->rank == 0 has the definite answer
        }
    // Broadcast the result to every rank.
    // This way, each rank can check, whether
    // it needs to update and can determine the exchanged momentum.
    bcast(vel->x, ms->gbl_rank, m_exec_conf->getMPICommunicator());
    bcast(vel->y, ms->gbl_rank, m_exec_conf->getMPICommunicator());
    bcast(vel->z, ms->gbl_rank, m_exec_conf->getMPICommunicator());
    }

void MuellerPlatheFlow::mpiExchangeVelocity(void)
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        bcastVelToAll(&m_min_swap, &m_last_min_vel, MPI_MINLOC);
        bcastVelToAll(&m_max_swap, &m_last_max_vel, MPI_MAXLOC);
        }
#endif // ENABLE_MPI
    }

#endif // ENABLE_MPI

namespace detail
    {
void export_MuellerPlatheFlow(pybind11::module& m)
    {
    pybind11::class_<MuellerPlatheFlow, Updater, std::shared_ptr<MuellerPlatheFlow>> flow(
        m,
        "MuellerPlatheFlow");
    flow.def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::string,
                            std::string,
                            const unsigned int,
                            const unsigned int,
                            const unsigned int,
                            Scalar>())
        .def_property_readonly("n_slabs", &MuellerPlatheFlow::getNSlabs)
        .def_property_readonly("min_slab", &MuellerPlatheFlow::getMinSlab)
        .def_property_readonly("max_slab", &MuellerPlatheFlow::getMaxSlab)
        .def_property_readonly("has_min_slab", &MuellerPlatheFlow::hasMinSlab)
        .def_property_readonly("has_max_slab", &MuellerPlatheFlow::hasMaxSlab)
        .def_property_readonly("flow_target", &MuellerPlatheFlow::getFlowTarget)
        .def_property_readonly("slab_direction", &MuellerPlatheFlow::getSlabDirectionPython)
        .def_property_readonly("flow_direction", &MuellerPlatheFlow::getFlowDirectionPython)
        .def_property_readonly("flow_epsilon", &MuellerPlatheFlow::getFlowEpsilon)
        .def_property_readonly("summed_exchanged_momentum",
                               &MuellerPlatheFlow::getSummedExchangedMomentum);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
