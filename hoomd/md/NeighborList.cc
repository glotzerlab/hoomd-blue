// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#include "NeighborList.h"
#include "hoomd/BondedGroupData.h"

#include <iostream>
#include <stdexcept>

using namespace std;

/*! \file NeighborList.cc
    \brief Defines the NeighborList class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System the neighborlist is to compute neighbors for
    \param _r_cut Cutoff radius for all pairs under which particles are considered neighbors
    \param r_buff Buffer radius around \a r_cut in which neighbors will be included

    \post NeighborList is initialized and the list memory has been allocated,
        but the list will not be computed until compute is called.
    \post The storage mode defaults to half
*/
NeighborList::NeighborList(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff)
    : Compute(sysdef), m_typpair_idx(m_pdata->getNTypes()), m_rcut_max_max(0.0), m_rcut_min(0.0),
      m_r_buff(r_buff), m_filter_body(false), m_storage_mode(half), m_meshbond_data(NULL),
      m_rcut_changed(true), m_updates(0), m_forced_updates(0), m_dangerous_updates(0),
      m_force_update(true), m_dist_check(true), m_has_been_updated_once(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing Neighborlist" << endl;

    // r_buff must be non-negative or it is not physical
    if (m_r_buff < 0.0)
        {
        throw runtime_error("Requested buffer radius is less than zero.");
        }

    // initialize values
    m_last_updated_tstep = 0;
    m_last_checked_tstep = 0;
    m_last_check_result = false;
    m_rebuild_check_delay = 0;
    m_exclusions_set = false;

    m_n_particles_changed = false;

    // initialize box length at last update
    m_last_L = m_pdata->getGlobalBox().getNearestPlaneDistance();
    m_last_L_local = m_pdata->getBox().getNearestPlaneDistance();

    // allocate r_cut pairwise storage
    GlobalArray<Scalar> r_cut(m_typpair_idx.getNumElements(), m_exec_conf);
    m_r_cut.swap(r_cut);
    TAG_ALLOCATION(m_r_cut);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_r_cut.get(),
                      m_r_cut.getNumElements() * sizeof(Scalar),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // holds the maximum rcut on a per type basis
    GlobalArray<Scalar> rcut_max(m_pdata->getNTypes(), m_exec_conf);
    m_rcut_max.swap(rcut_max);
    TAG_ALLOCATION(m_rcut_max);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // store in host memory for faster access from CPU
        cudaMemAdvise(m_rcut_max.get(),
                      m_rcut_max.getNumElements() * sizeof(Scalar),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // holds the base rcut on a per type basis
    GlobalArray<Scalar> rcut_base(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcut_base.swap(rcut_base);
    TAG_ALLOCATION(m_rcut_base);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // store in host memory for faster access from CPU
        cudaMemAdvise(m_rcut_base.get(),
                      m_rcut_base.getNumElements() * sizeof(Scalar),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // allocate the r_listsq array which accelerates CPU calculations
    GlobalArray<Scalar> r_listsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_r_listsq.swap(r_listsq);
    TAG_ALLOCATION(m_r_listsq);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_r_listsq.get(),
                      m_r_listsq.getNumElements() * sizeof(Scalar),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // allocate the number of neighbors (per particle)
    GlobalArray<unsigned int> n_neigh(m_pdata->getMaxN(), m_exec_conf);
    m_n_neigh.swap(n_neigh);
    TAG_ALLOCATION(m_n_neigh);

    // default allocation of 4 neighbors per particle for the neighborlist
    GlobalArray<unsigned int> nlist(4 * m_pdata->getMaxN(), m_exec_conf);
    m_nlist.swap(nlist);
    TAG_ALLOCATION(m_nlist);

    // allocate head list indexer
    GlobalArray<size_t> head_list(m_pdata->getMaxN(), m_exec_conf);
    m_head_list.swap(head_list);
    TAG_ALLOCATION(m_head_list);

    // allocate the max number of neighbors per type allowed
    GlobalArray<unsigned int> Nmax(m_pdata->getNTypes(), m_exec_conf);
    m_Nmax.swap(Nmax);
    TAG_ALLOCATION(m_Nmax);

        // flood Nmax with 4s initially
        {
        ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            h_Nmax.data[i] = 4;
            }
        }

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_Nmax.get(),
                      m_Nmax.getNumElements() * sizeof(unsigned int),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // allocate overflow flags for the number of neighbors per type
    GlobalArray<unsigned int> conditions(m_pdata->getNTypes(), m_exec_conf);
    m_conditions.swap(conditions);
    TAG_ALLOCATION(m_conditions);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // store in host memory for faster access from CPU
        cudaMemAdvise(m_conditions.get(),
                      m_conditions.getNumElements() * sizeof(unsigned int),
                      cudaMemAdviseSetPreferredLocation,
                      cudaCpuDeviceId);
        CHECK_CUDA_ERROR();
        }
#endif

        {
        // initially reset conditions
        ArrayHandle<unsigned int> h_conditions(m_conditions,
                                               access_location::host,
                                               access_mode::overwrite);
        memset(h_conditions.data, 0, sizeof(unsigned int) * m_pdata->getNTypes());
        }

    // allocate m_last_pos
    GlobalArray<Scalar4> last_pos(m_pdata->getMaxN(), m_exec_conf);
    m_last_pos.swap(last_pos);
    TAG_ALLOCATION(m_last_pos);

    // allocate initial memory allowing 4 exclusions per particle (will grow to match specified
    // exclusions)

    // note: this breaks O(N/P) memory scaling
    GlobalVector<unsigned int> n_ex_tag(m_pdata->getRTags().size(), m_exec_conf);
    m_n_ex_tag.swap(n_ex_tag);
    TAG_ALLOCATION(m_n_ex_tag);

    GlobalArray<unsigned int> ex_list_tag(m_pdata->getRTags().size(), 1, m_exec_conf);
    m_ex_list_tag.swap(ex_list_tag);
    TAG_ALLOCATION(m_ex_list_tag);

    GlobalArray<unsigned int> n_ex_idx(m_pdata->getMaxN(), m_exec_conf);
    m_n_ex_idx.swap(n_ex_idx);
    TAG_ALLOCATION(m_n_ex_idx);

    GlobalArray<unsigned int> ex_list_idx(m_pdata->getMaxN(), 1, m_exec_conf);
    m_ex_list_idx.swap(ex_list_idx);
    TAG_ALLOCATION(m_ex_list_idx);

    // reset exclusions
    resizeAndClearExclusions();

    m_ex_list_indexer = Index2D((unsigned int)m_ex_list_idx.getPitch(), 1);
    m_ex_list_indexer_tag = Index2D((unsigned int)m_ex_list_tag.getPitch(), 1);

    // connect to particle sort to force rebuild
    m_pdata->getParticleSortSignal().connect<NeighborList, &NeighborList::forceUpdate>(this);

    // connect to max particle change to resize neighborlist arrays
    m_pdata->getMaxParticleNumberChangeSignal().connect<NeighborList, &NeighborList::reallocate>(
        this);

    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalParticleNumberChange>(this);

    m_sysdef->getBondData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getAngleData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getDihedralData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getImproperData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getConstraintData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getPairData()
        ->getGroupNumChangeSignal()
        .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    // connect locally to the rcut changing signal
    getRCutChangeSignal().connect<NeighborList, &NeighborList::slotRCutChange>(this);

    // allocate m_update_periods tracking info
    m_update_periods.resize(100);
    for (unsigned int i = 0; i < m_update_periods.size(); i++)
        m_update_periods[i] = 0;

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        m_last_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();

        m_comm->getMigrateSignal().connect<NeighborList, &NeighborList::peekUpdate>(this);
        m_comm->getCommFlagsRequestSignal()
            .connect<NeighborList, &NeighborList::getRequestedCommFlags>(this);
        m_comm->getGhostLayerWidthRequestSignal()
            .connect<NeighborList, &NeighborList::getGhostLayerWidth>(this);
        }
#endif
    }

void NeighborList::reallocate()
    {
    // resize the exclusions
    m_last_pos.resize(m_pdata->getMaxN());
    size_t old_n_ex = m_n_ex_idx.getNumElements();
    m_n_ex_idx.resize(m_pdata->getMaxN());

        {
        ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx,
                                             access_location::host,
                                             access_mode::readwrite);
        memset(h_n_ex_idx.data + old_n_ex,
               0,
               sizeof(unsigned int) * (m_n_ex_idx.getNumElements() - old_n_ex));
        }

    unsigned int ex_list_height = m_ex_list_indexer.getH();
    m_ex_list_idx.resize(m_pdata->getMaxN(), ex_list_height);
    m_ex_list_indexer = Index2D((unsigned int)m_ex_list_idx.getPitch(), ex_list_height);

    // resize the head list and number of neighbors per particle
    m_head_list.resize(m_pdata->getMaxN());
    m_n_neigh.resize(m_pdata->getMaxN());

    // force a rebuild
    forceUpdate();
    }

NeighborList::~NeighborList()
    {
    m_exec_conf->msg->notice(5) << "Destroying Neighborlist" << endl;

    m_pdata->getParticleSortSignal().disconnect<NeighborList, &NeighborList::forceUpdate>(this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<NeighborList, &NeighborList::reallocate>(
        this);
    m_pdata->getGlobalParticleNumberChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalParticleNumberChange>(this);

    m_sysdef->getBondData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getAngleData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getDihedralData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getImproperData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getConstraintData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

    m_sysdef->getPairData()
        ->getGroupNumChangeSignal()
        .disconnect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getMigrateSignal().disconnect<NeighborList, &NeighborList::peekUpdate>(this);
        m_comm->getCommFlagsRequestSignal()
            .disconnect<NeighborList, &NeighborList::getRequestedCommFlags>(this);
        m_comm->getGhostLayerWidthRequestSignal()
            .disconnect<NeighborList, &NeighborList::getGhostLayerWidth>(this);
        }
#endif

    getRCutChangeSignal().disconnect<NeighborList, &NeighborList::slotRCutChange>(this);
    }

/*! Updates the neighborlist if it has not yet been updated this times step
    \param timestep Current time step of the simulation
*/
void NeighborList::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    // check if the rcut array has changed and update it
    if (m_rcut_changed)
        {
        updateRList();
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            m_comm->communicate(timestep);
            }
#endif
        }

    // skip if we shouldn't compute this step
    if (!shouldCompute(timestep) && !m_force_update)
        return;

    // when the number of particles or bonds in the system changes, rebuild the exclusion list
    if (m_n_particles_changed || m_topology_changed)
        {
        resizeAndClearExclusions();
        m_n_particles_changed = false;
        m_topology_changed = false;

        for (const std::string& exclusion : m_exclusions)
            {
            setSingleExclusion(exclusion);
            }
        }

    // take care of some updates if things have changed since construction
    if (m_force_update)
        {
        // build the head list since some sort of change (like a particle sort) happened
        buildHeadList();

        if (m_exclusions_set)
            updateExListIdx();
        }

    // check if the list needs to be updated and update it
    if (needsUpdating(timestep))
        {
        // check simulation box size is OK
        checkBoxSize();

        // rebuild the list until there is no overflow
        bool overflowed = false;
        do
            {
            buildNlist(timestep);

            overflowed = checkConditions();
            // if we overflowed, need to reallocate memory and reset the conditions
            if (overflowed)
                {
                // always rebuild the head list after an overflow
                buildHeadList();

                // zero out the conditions for the next build
                resetConditions();
                }
            } while (overflowed);

        if (m_exclusions_set)
            filterNlist();

        setLastUpdatedPos();
        m_has_been_updated_once = true;
        }
    }

/*! \param r_buff New buffer radius to set
    \note Changing the buffer radius does NOT immediately update the neighborlist.
            The new buffer will take effect when compute is called for the next timestep.
*/
void NeighborList::setRBuff(Scalar r_buff)
    {
    m_r_buff = r_buff;
    if (m_r_buff < 0.0)
        {
        throw runtime_error("Requested buffer radius is less than zero.");
        }
    notifyRCutMatrixChange();
    forceUpdate();
    }

void NeighborList::updateRList()
    {
    // overwrite the new r_cut matrix
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_rcut_base(m_rcut_base, access_location::host, access_mode::overwrite);

    for (unsigned int i = 0; i < m_r_cut.getNumElements(); i++)
        {
        h_r_cut.data[i] = h_rcut_base.data[i];
        }

    // first: loop over the consumer r_cut matrices and take their max in h_r_cut
    for (unsigned int matrix = 0; matrix < m_consumer_r_cut.size(); matrix++)
        {
        ArrayHandle<Scalar> h_consumer_r_cut(*m_consumer_r_cut[matrix],
                                             access_location::host,
                                             access_mode::read);

        if (m_consumer_r_cut[matrix]->getNumElements() != m_r_cut.getNumElements())
            {
            throw std::invalid_argument("given r_cut_matrix is not the right size");
            }

        // take the maximum
        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            for (unsigned int j = 0; j < m_pdata->getNTypes(); ++j)
                {
                h_r_cut.data[m_typpair_idx(i, j)]
                    = std::max(h_r_cut.data[m_typpair_idx(i, j)],
                               h_consumer_r_cut.data[m_typpair_idx(i, j)]);
                }
            }
        }

    // now, update the r_list which includes r_buff we need to read and write on the r_list
    ArrayHandle<Scalar> h_r_listsq(m_r_listsq, access_location::host, access_mode::overwrite);

    // update the maximum cutoff of all those set so far
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::readwrite);

    Scalar r_cut_max = 0.0f;
    for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
        {
        // get the maximum cutoff for this type
        Scalar r_cut_max_i = 0.0f;
        for (unsigned int j = 0; j < m_pdata->getNTypes(); ++j)
            {
            const Scalar r_cut_ij = h_r_cut.data[m_typpair_idx(i, j)];
            if (r_cut_ij > r_cut_max_i)
                r_cut_max_i = r_cut_ij;

            // precompute rlistsq while we're at it
            Scalar r_list = (r_cut_ij > Scalar(0.0)) ? r_cut_ij + m_r_buff : Scalar(0.0);
            h_r_listsq.data[m_typpair_idx(i, j)] = r_list * r_list;
            }
        h_rcut_max.data[i] = r_cut_max_i;
        if (r_cut_max_i > r_cut_max)
            r_cut_max = r_cut_max_i;
        }
    m_rcut_max_max = r_cut_max;

    // loop back through and compute the minimum
    // this extra loop guards against some weird case where all of the cutoffs are turned off
    // and we accidentally get infinity
    Scalar r_cut_min = m_rcut_max_max;
    for (unsigned int cur_pair = 0; cur_pair < m_typpair_idx.getNumElements(); ++cur_pair)
        {
        const Scalar r_cut_ij = h_r_cut.data[cur_pair];
        // if cutoff is defined and less than total minimum
        if (r_cut_ij > Scalar(0.0) && r_cut_ij < r_cut_min)
            r_cut_min = r_cut_ij;
        }
    m_rcut_min = r_cut_min;

    // rcut has been updated to the latest values now
    m_rcut_changed = false;
    }

/*!
 * Check that the largest neighbor search radius is not bigger than twice the shortest box size.
 * Raises an error if this condition is not met. Otherwise, nothing happens.
 */
void NeighborList::checkBoxSize()
    {
    const BoxDim& box = m_pdata->getBox();
    const uchar3 periodic = box.getPeriodic();

    // check that rcut fits in the box
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
    Scalar rmax = m_rcut_max_max + m_r_buff;

    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0)
        || (periodic.y && nearest_plane_distance.y <= rmax * 2.0)
        || (m_sysdef->getNDimensions() == 3 && periodic.z
            && nearest_plane_distance.z <= rmax * 2.0))
        {
        std::ostringstream oss;
        oss << "nlist: Simulation box is too small, the neighbor list is searching beyond the "
            << " minimum image: rmax=" << rmax << std::endl;

        if (box.getPeriodic().x)
            oss << "nearest_plane_distance.x=" << nearest_plane_distance.x << std::endl;
        if (box.getPeriodic().y)
            oss << "nearest_plane_distance.y=" << nearest_plane_distance.y << std::endl;
        if (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z)
            oss << "nearest_plane_distance.z=" << nearest_plane_distance.z << std::endl;
        throw std::runtime_error(oss.str());
        }
    }

/*! \param tag1 TAG (not index) of the first particle in the pair
    \param tag2 TAG (not index) of the second particle in the pair
    \post The pair \a tag1, \a tag2 will not appear in the neighborlist
    \note This only takes effect on the next call to compute() that updates the list
    \note Duplicates are checked for and not added.
*/
void NeighborList::addExclusion(unsigned int tag1, unsigned int tag2)
    {
    assert(tag1 <= m_pdata->getMaximumTag());
    assert(tag2 <= m_pdata->getMaximumTag());

    assert(!m_n_particles_changed);

    m_exclusions_set = true;

    // don't add an exclusion twice
    if (isExcluded(tag1, tag2))
        return;

    // this is clunky, but needed due to the fact that we cannot have an array handle in scope when
    // calling grow exclusion list
    bool grow = false;
        {
        // access arrays
        ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag,
                                             access_location::host,
                                             access_mode::readwrite);

        // grow the list if necessary
        if (h_n_ex_tag.data[tag1] == m_ex_list_indexer.getH())
            grow = true;

        if (h_n_ex_tag.data[tag2] == m_ex_list_indexer.getH())
            grow = true;
        }

    if (grow)
        {
        growExclusionList();
        }

        {
        // access arrays
        ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag,
                                                access_location::host,
                                                access_mode::readwrite);
        ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag,
                                             access_location::host,
                                             access_mode::readwrite);

        // add tag2 to tag1's exclusion list
        unsigned int pos1 = h_n_ex_tag.data[tag1];
        assert(pos1 < m_ex_list_indexer.getH());
        h_ex_list_tag.data[m_ex_list_indexer_tag(tag1, pos1)] = tag2;
        h_n_ex_tag.data[tag1]++;

        // add tag1 to tag2's exclusion list
        unsigned int pos2 = h_n_ex_tag.data[tag2];
        assert(pos2 < m_ex_list_indexer.getH());
        h_ex_list_tag.data[m_ex_list_indexer_tag(tag2, pos2)] = tag1;
        h_n_ex_tag.data[tag2]++;
        }

    forceUpdate();
    }

/*! \post No particles are excluded from the neighbor list
 */
void NeighborList::resizeAndClearExclusions()
    {
    // reallocate list of exclusions per tag if necessary
    if (m_n_particles_changed)
        {
        m_n_ex_tag.resize(m_pdata->getRTags().size());

        // slave the width of the exclusion list to the capacity of the number of exclusions array
        // in order to amortize reallocation costs
        if (m_ex_list_tag.getPitch() != m_n_ex_tag.getNumElements())
            {
            m_ex_list_tag.resize(m_n_ex_tag.getNumElements(), m_ex_list_tag.getHeight());
            m_ex_list_indexer_tag = Index2D((unsigned int)m_ex_list_tag.getPitch(),
                                            (unsigned int)m_ex_list_tag.getHeight());
            }
        }

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::overwrite);

    memset(h_n_ex_tag.data, 0, sizeof(unsigned int) * m_n_ex_tag.getNumElements());
    memset(h_n_ex_idx.data, 0, sizeof(unsigned int) * m_n_ex_idx.getNumElements());
    m_exclusions_set = false;

    forceUpdate();
    }

//! Get number of exclusions involving n particles
unsigned int NeighborList::getNumExclusions(unsigned int size)
    {
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    unsigned int count = 0;
    unsigned int ntags = (unsigned int)m_pdata->getRTags().size();
    for (unsigned int tag = 0; tag <= ntags; tag++)
        {
        if (!m_pdata->isTagActive(tag))
            {
            continue;
            }
        unsigned int num_excluded = h_n_ex_tag.data[tag];

        if (num_excluded == size)
            count++;
        }

    return count;
    }

void NeighborList::setExclusions(pybind11::list exclusions)
    {
    resizeAndClearExclusions();
    setFilterBody(false);
    m_exclusions = set<std::string>();
    for (auto exclusion : exclusions)
        {
        setSingleExclusion(exclusion.cast<std::string>());
        }
    }

void NeighborList::setSingleExclusion(std::string exclusion)
    {
    if (exclusion == "bond")
        {
        addExclusionsFromBonds();
        m_exclusions.insert("bond");
        }
    else if (exclusion == "meshbond" && m_meshbond_data)
        {
        addExclusionsFromMeshBonds();
        m_exclusions.insert("meshbond");
        }
    else if (exclusion == "special_pair")
        {
        addExclusionsFromPairs();
        m_exclusions.insert("special_pair");
        }
    else if (exclusion == "constraint")
        {
        addExclusionsFromConstraints();
        m_exclusions.insert("constraint");
        }
    else if (exclusion == "angle")
        {
        addExclusionsFromAngles();
        m_exclusions.insert("angle");
        }
    else if (exclusion == "dihedral")
        {
        addExclusionsFromDihedrals();
        m_exclusions.insert("dihedral");
        }
    else if (exclusion == "body")
        {
        setFilterBody(true);
        m_exclusions.insert("body");
        }
    else if (exclusion == "1-3")
        {
        addOneThreeExclusionsFromTopology();
        m_exclusions.insert("1-3");
        }
    else if (exclusion == "1-4")
        {
        addOneFourExclusionsFromTopology();
        m_exclusions.insert("1-4");
        }
    }

pybind11::list NeighborList::getExclusions()
    {
    auto exclusions = pybind11::list();
    for (auto exclusion : m_exclusions)
        {
        exclusions.append(exclusion);
        }
    return exclusions;
    }

/*! \post Gather some statistics about exclusions usage.
 */
void NeighborList::countExclusions()
    {
    unsigned int MAX_COUNT_EXCLUDED = 16;
    unsigned int excluded_count[MAX_COUNT_EXCLUDED + 2];
    unsigned int num_excluded, max_num_excluded;

    assert(!m_n_particles_changed);

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);

    max_num_excluded = 0;
    for (unsigned int c = 0; c <= MAX_COUNT_EXCLUDED + 1; ++c)
        excluded_count[c] = 0;

    unsigned int max_tag = (unsigned int)m_pdata->getRTags().size();
    for (unsigned int i = 0; i < max_tag; i++)
        {
        num_excluded = h_n_ex_tag.data[i];

        if (num_excluded > max_num_excluded)
            max_num_excluded = num_excluded;

        if (num_excluded > MAX_COUNT_EXCLUDED)
            num_excluded = MAX_COUNT_EXCLUDED + 1;

        excluded_count[num_excluded] += 1;
        }

    m_exec_conf->msg->notice(2) << "-- Neighborlist exclusion statistics -- :" << endl;
    for (unsigned int i = 0; i <= MAX_COUNT_EXCLUDED; ++i)
        {
        if (excluded_count[i] > 0)
            m_exec_conf->msg->notice(2)
                << "Particles with " << i << " exclusions             : " << excluded_count[i]
                << endl;
        }

    if (excluded_count[MAX_COUNT_EXCLUDED + 1])
        {
        m_exec_conf->msg->notice(2)
            << "Particles with more than " << MAX_COUNT_EXCLUDED
            << " exclusions: " << excluded_count[MAX_COUNT_EXCLUDED + 1] << endl;
        }

    if (m_filter_body)
        m_exec_conf->msg->notice(2) << "Neighbors excluded when in the same body: yes" << endl;
    else
        m_exec_conf->msg->notice(2) << "Neighbors excluded when in the same body: no" << endl;

    bool has_bodies = m_pdata->hasBodies();
    if (!m_filter_body && has_bodies)
        {
        m_exec_conf->msg->warning()
            << "Disabling the body exclusion will cause rigid bodies to behave erratically" << endl
            << "            unless inter-body pair forces are very small." << endl;
        }
    }

/*! After calling addExclusionsFromBonds() all bonds specified in the attached ParticleData will be
    added as exclusions. Any additional bonds added after this will not be automatically added as
   exclusions.
*/
void NeighborList::addExclusionsFromBonds()
    {
    std::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

    // access bond data by snapshot
    BondData::Snapshot snapshot;
    bond_data->takeSnapshot(snapshot);

    // broadcast global bond list
    std::vector<BondData::members_t> bonds;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            bonds = snapshot.groups;

        bcast(bonds, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        bonds = snapshot.groups;
        }

    // for each bond
    for (unsigned int i = 0; i < bonds.size(); i++)
        // add an exclusion
        addExclusion(bonds[i].tag[0], bonds[i].tag[1]);
    }

/*! After calling addExclusionsFromMeshBonds() all meshbonds specified in the attached Mesh will be
    added as exclusions. Any additional meshbonds added after this will not be automatically added
   as exclusions.
*/
void NeighborList::addExclusionsFromMeshBonds()
    {
    // access bond data by snapshot
    BondData::Snapshot snapshot;
    m_meshbond_data->takeSnapshot(snapshot);

    // broadcast global bond list
    std::vector<BondData::members_t> bonds;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            bonds = snapshot.groups;

        bcast(bonds, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        bonds = snapshot.groups;
        }

    // for each bond
    for (unsigned int i = 0; i < bonds.size(); i++)
        // add an exclusion
        addExclusion(bonds[i].tag[0], bonds[i].tag[1]);
    }

/*! After calling addExclusionsFromAngles(), all angles specified in the attached ParticleData will
   be added to the exclusion list. Only the two end particles in the angle are excluded from
   interacting.
*/
void NeighborList::addExclusionsFromAngles()
    {
    std::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();

    // access angle data by snapshot
    AngleData::Snapshot snapshot;
    angle_data->takeSnapshot(snapshot);

    // broadcast global angle list
    std::vector<AngleData::members_t> angles;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            angles = snapshot.groups;

        bcast(angles, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        angles = snapshot.groups;
        }

    // for each angle
    for (unsigned int i = 0; i < angles.size(); i++)
        addExclusion(angles[i].tag[0], angles[i].tag[2]);
    }

/*! After calling addExclusionsFromDihedrals(), all dihedrals specified in the attached ParticleData
   will be added to the exclusion list. Only the two end particles in the dihedral are excluded from
   interacting.
*/
void NeighborList::addExclusionsFromDihedrals()
    {
    std::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();

    // access dihedral data by snapshot
    DihedralData::Snapshot snapshot;
    dihedral_data->takeSnapshot(snapshot);

    // broadcast global dihedral list
    std::vector<DihedralData::members_t> dihedrals;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            dihedrals = snapshot.groups;

        bcast(dihedrals, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        dihedrals = snapshot.groups;
        }

    // for each dihedral
    for (unsigned int i = 0; i < dihedrals.size(); i++)
        addExclusion(dihedrals[i].tag[0], dihedrals[i].tag[3]);
    }

/*! After calling addExclusionFromConstraints() all constraints specified in the attached
   ConstraintData will be added as exclusions. Any additional constraints added after this will not
   be automatically added as exclusions.
*/
void NeighborList::addExclusionsFromConstraints()
    {
    std::shared_ptr<ConstraintData> constraint_data = m_sysdef->getConstraintData();

    // access constraint data by snapshot
    ConstraintData::Snapshot snapshot;
    constraint_data->takeSnapshot(snapshot);

    // broadcast global constraint list
    std::vector<ConstraintData::members_t> constraints;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            constraints = snapshot.groups;

        bcast(constraints, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        constraints = snapshot.groups;
        }

    // for each constraint
    for (unsigned int i = 0; i < constraints.size(); i++)
        // add an exclusion
        addExclusion(constraints[i].tag[0], constraints[i].tag[1]);
    }

/*! After calling addExclusionFromPairs() all pairs specified in the attached ParticleData will be
    added as exclusions. Any additional pairs added after this will not be automatically added as
   exclusions.
*/
void NeighborList::addExclusionsFromPairs()
    {
    std::shared_ptr<PairData> pair_data = m_sysdef->getPairData();

    // access pair data by snapshot
    PairData::Snapshot snapshot;
    pair_data->takeSnapshot(snapshot);

    // broadcast global bond list
    std::vector<PairData::members_t> pairs;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            pairs = snapshot.groups;

        bcast(pairs, 0, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        pairs = snapshot.groups;
        }

    // for each pair
    for (unsigned int i = 0; i < pairs.size(); i++)
        // add an exclusion
        addExclusion(pairs[i].tag[0], pairs[i].tag[1]);
    }

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 have been excluded from the neighbor list
*/
bool NeighborList::isExcluded(unsigned int tag1, unsigned int tag2)
    {
    assert(!m_n_particles_changed);

    assert(tag1 <= m_pdata->getMaximumTag());
    assert(tag2 <= m_pdata->getMaximumTag());

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag,
                                            access_location::host,
                                            access_mode::read);

    unsigned int n_ex = h_n_ex_tag.data[tag1];
    for (unsigned int i = 0; i < n_ex; i++)
        {
        if (h_ex_list_tag.data[m_ex_list_indexer_tag(tag1, i)] == tag2)
            return true;
        }

    return false;
    }

/*! Add topologically derived exclusions for angles
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are bonded to the same atom.
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 */
void NeighborList::addOneThreeExclusionsFromTopology()
    {
    std::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = (unsigned int)m_pdata->getRTags().size();
    const unsigned int MAXNBONDS
        = 7
          + 1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNGlobal();

    if (nBonds == 0)
        {
        m_exec_conf->msg->warning()
            << "nlist: No bonds defined while trying to add topology derived 1-3 exclusions"
            << endl;
        return;
        }

    // build a per atom list with all bonding partners from the list of bonds.
    unsigned int* localBondList = new unsigned int[MAXNBONDS * myNAtoms];
    memset((void*)localBondList, 0, sizeof(unsigned int) * MAXNBONDS * myNAtoms);

    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exclusion map

        // FIXME: this will not work when the group tags are not contiguous
        Bond bondi = bond_data->getGroupByTag(i);

        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;

        // next, increment the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA * MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB * MAXNBONDS];

        if (nBondsA >= MAXNBONDS)
            {
            std::ostringstream s;
            s << "Too many bonds to process exclusions for particle with tag: " << tagA << ".";
            s << "Compile time maximum set to: " << MAXNBONDS - 1 << endl;
            throw runtime_error(s.str());
            }

        if (nBondsB >= MAXNBONDS)
            {
            std::ostringstream s;
            s << "Too many bonds to process exclusions for particle with tag: " << tagB << ".";
            throw runtime_error(s.str());
            }

        localBondList[tagA * MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB * MAXNBONDS + nBondsB] = tagA;
        }

    // now loop over the atoms and build exclusions if we have more than
    // one bonding partner, i.e. we are in the center of an angle.
    for (unsigned int i = 0; i < myNAtoms; i++)
        {
        // now, loop over all atoms, and find those in the middle of an angle
        const unsigned int iAtom = i * MAXNBONDS;
        const unsigned int nBonds = localBondList[iAtom];

        if (nBonds > 1) // need at least two bonds
            {
            for (unsigned int j = 1; j < nBonds; ++j)
                {
                for (unsigned int k = j + 1; k <= nBonds; ++k)
                    addExclusion(localBondList[iAtom + j], localBondList[iAtom + k]);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }

/*! Add topologically derived exclusions for dihedrals
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are connected to a common bond.
 *
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 * and then loop over bonded partners.
 */
void NeighborList::addOneFourExclusionsFromTopology()
    {
    std::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = (unsigned int)m_pdata->getRTags().size();
    const unsigned int MAXNBONDS
        = 7
          + 1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNGlobal();

    if (nBonds == 0)
        {
        m_exec_conf->msg->warning()
            << "nlist: No bonds defined while trying to add topology derived 1-4 exclusions"
            << endl;
        return;
        }

    // allocate and clear data.
    unsigned int* localBondList = new unsigned int[MAXNBONDS * myNAtoms];
    memset((void*)localBondList, 0, sizeof(unsigned int) * MAXNBONDS * myNAtoms);

    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exclusion map
        Bond bondi = bond_data->getGroupByTag(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;

        // next, increment the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA * MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB * MAXNBONDS];

        if (nBondsA >= MAXNBONDS)
            {
            std::ostringstream s;
            s << "Too many bonds to process exclusions for particle with tag: " << tagA << ".";
            throw runtime_error(s.str());
            }

        if (nBondsB >= MAXNBONDS)
            {
            std::ostringstream s;
            s << "Too many bonds to process exclusions for particle with tag: " << tagB << ".";
            throw runtime_error(s.str());
            }

        localBondList[tagA * MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB * MAXNBONDS + nBondsB] = tagA;
        }

    //  loop over all bonds
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // FIXME: this will not work when the group tags are not contiguous
        Bond bondi = bond_data->getGroupByTag(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;

        const unsigned int nBondsA = localBondList[tagA * MAXNBONDS];
        const unsigned int nBondsB = localBondList[tagB * MAXNBONDS];

        for (unsigned int j = 1; j <= nBondsA; j++)
            {
            const unsigned int tagJ = localBondList[tagA * MAXNBONDS + j];
            if (tagJ == tagB) // skip the bond in the middle of the dihedral
                continue;

            for (unsigned int k = 1; k <= nBondsB; k++)
                {
                const unsigned int tagK = localBondList[tagB * MAXNBONDS + k];
                if (tagK == tagA) // skip the bond in the middle of the dihedral
                    continue;

                addExclusion(tagJ, tagK);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }

/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since
   the last call to this method that returned true. \returns false If none of the particles has been
   moved more than 1/2 of the buffer distance since the last call to this method that returned true.

    Note: this method relies on data set by setLastUpdatedPos(), which must be called to set the
   previous data used in the next call to distanceCheck();
*/
bool NeighborList::distanceCheck(uint64_t timestep)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);

    // temporary storage for the result
    bool result = false;

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // get current nearest plane distances
    Scalar3 L_g = m_pdata->getGlobalBox().getNearestPlaneDistance();

    // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
    Scalar3 lambda = L_g / m_last_L;
    Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
    lambda_min = (lambda_min < lambda.z) ? lambda_min : (Scalar)lambda.z;

    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        const unsigned int type_i = __scalar_as_int(h_pos.data[i].w);

        // minimum distance within which all particles should be included
        Scalar old_rmin = h_rcut_max.data[type_i];

        // maximum value we have checked for neighbors, defined by the buffer layer
        Scalar rmax = old_rmin + m_r_buff;

        // max displacement for each particle (after subtraction of homogeneous dilations)
        const Scalar delta_max = (rmax * lambda_min - old_rmin) / Scalar(2.0);
        Scalar maxsq = (delta_max > 0) ? delta_max * delta_max : 0;

        Scalar3 dx = make_scalar3(h_pos.data[i].x - lambda.x * h_last_pos.data[i].x,
                                  h_pos.data[i].y - lambda.y * h_last_pos.data[i].y,
                                  h_pos.data[i].z - lambda.z * h_last_pos.data[i].z);

        dx = box.minImage(dx);

        if (dot(dx, dx) >= maxsq)
            {
            result = true;
            break;
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // check if migrate criterion is fulfilled on any rank
        int local_result = result ? 1 : 0;
        int global_result = 0;
        MPI_Allreduce(&local_result,
                      &global_result,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      m_exec_conf->getMPICommunicator());
        result = (global_result > 0);
        }
#endif

    // don't worry about computing flops here, this is fast
    return result;
    }

/*! Copies the current positions of all particles over to m_last_x etc...
 */
void NeighborList::setLastUpdatedPos()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);

    // update the last position arrays
    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_last_pos.data[i]
            = make_scalar4(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z, Scalar(0.0));
        }

    // update last box nearest plane distance
    m_last_L = m_pdata->getGlobalBox().getNearestPlaneDistance();
    m_last_L_local = m_pdata->getBox().getNearestPlaneDistance();
    }

bool NeighborList::shouldCheckDistance(uint64_t timestep)
    {
    return !m_force_update && !(timestep < (m_last_updated_tstep + m_rebuild_check_delay));
    }

/*! \returns true If the neighbor list needs to be updated
    \returns false If the neighbor list does not need to be updated
    \note This is designed to be called if (needsUpdating()) then update every step.
        It internally handles many state variables that rely on this assumption.

    \param timestep Current time step in the simulation
*/
bool NeighborList::needsUpdating(uint64_t timestep)
    {
    if (m_last_checked_tstep == timestep)
        {
        if (m_force_update)
            {
            // force update is counted only once per time step
            m_force_update = false;
            return true;
            }
        return m_last_check_result;
        }

    m_last_checked_tstep = timestep;

    if (!m_force_update && !shouldCheckDistance(timestep))
        {
        m_last_check_result = false;
        return false;
        }

    // temporary storage for return result
    bool result = false;

    // check if this is a dangerous time
    // we are dangerous if m_rebuild_check_delay is greater than 1 and this is the first check after
    // the last build
    bool dangerous = false;
    if (m_dist_check
        && (m_rebuild_check_delay > 1
            && timestep == (m_last_updated_tstep + m_rebuild_check_delay)))
        dangerous = true;

    // if the update has been forced, the result defaults to true
    if (m_force_update)
        {
        result = true;
        m_force_update = false;
        m_forced_updates += 1;
        m_last_updated_tstep = timestep;

        // when an update is forced, there is no way to tell if the build
        // is dangerous or not: filter out the false positive errors
        dangerous = false;
        }
    else
        {
        // not a forced update, perform the distance check to determine
        // if the list needs to be updated - no dist check needed if r_buff is tiny
        // it also needs to be updated if m_rebuild_check_delay is 0, or the check period is hit
        // when distance checks are disabled
        if (m_r_buff < 1e-6
            || (!m_dist_check
                && (m_rebuild_check_delay == 0
                    || (m_rebuild_check_delay > 1
                        && timestep == (m_last_updated_tstep + m_rebuild_check_delay)))))
            {
            result = true;
            }
        else
            {
            result = distanceCheck(timestep);
            }

        if (result)
            {
            // record update histogram - but only if the period is positive
            if (timestep > m_last_updated_tstep)
                {
                uint64_t period = timestep - m_last_updated_tstep;
                if (period >= m_update_periods.size())
                    period = m_update_periods.size() - 1;
                m_update_periods[period]++;
                }

            m_last_updated_tstep = timestep;
            m_updates += 1;
            }
        }

    // warn the user if this is a dangerous build
    if (result && dangerous)
        {
        m_exec_conf->msg->notice(2)
            << "nlist: Dangerous neighborlist build occurred. Continuing this simulation may "
               "produce incorrect results and/or program crashes. Decrease the neighborlist "
               "check_period and rerun."
            << endl;
        m_dangerous_updates += 1;
        }

    m_last_check_result = result;
    return result;
    }

void NeighborList::resetStats()
    {
    m_updates = m_forced_updates = m_dangerous_updates = 0;

    for (unsigned int i = 0; i < m_update_periods.size(); i++)
        m_update_periods[i] = 0;
    }

unsigned int NeighborList::getSmallestRebuild()
    {
    for (unsigned int i = 0; i < m_update_periods.size(); i++)
        {
        if (m_update_periods[i] != 0)
            return i;
        }
    return (unsigned int)m_update_periods.size();
    }

/*! This method is now deprecated, and deriving classes must supply it.
 */
void NeighborList::buildNlist(uint64_t timestep)
    {
    throw runtime_error("Not implemented.");
    }

/*! Translates the exclusions set in \c m_n_ex_tag and \c m_ex_list_tag to indices in \c m_n_ex_idx
 * and \c m_ex_list_idx
 */
void NeighborList::updateExListIdx()
    {
    assert(!m_n_particles_changed);

    // access data
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_ex_list_idx(m_ex_list_idx,
                                            access_location::host,
                                            access_mode::overwrite);

    // translate the number and exclusions from one array to the other
    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        // get the tag for this index
        unsigned int tag = h_tag.data[idx];

        // copy the number of exclusions over
        unsigned int n = h_n_ex_tag.data[tag];
        h_n_ex_idx.data[idx] = n;

        // construct the exclusion list
        for (unsigned int offset = 0; offset < n; offset++)
            {
            unsigned int ex_tag = h_ex_list_tag.data[m_ex_list_indexer_tag(tag, offset)];
            unsigned int ex_idx = h_rtag.data[ex_tag];

            // store excluded particle idx
            h_ex_list_idx.data[m_ex_list_indexer(idx, offset)] = ex_idx;
            }
        }
    }

/*! Loops through the neighbor list and filters out any excluded pairs
 */
void NeighborList::filterNlist()
    {
    // access data
    ArrayHandle<size_t> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_idx(m_ex_list_idx,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::readwrite);

    // for each particle's neighbor list
    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        size_t myHead = h_head_list.data[idx];
        unsigned int n_neigh = h_n_neigh.data[idx];
        unsigned int n_ex = h_n_ex_idx.data[idx];
        unsigned int new_n_neigh = 0;

        // loop over the list, regenerating it as we go
        for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
            {
            unsigned int cur_neigh = h_nlist.data[myHead + cur_neigh_idx];

            // test if excluded
            bool excluded = false;
            for (unsigned int cur_ex_idx = 0; cur_ex_idx < n_ex; cur_ex_idx++)
                {
                unsigned int cur_ex = h_ex_list_idx.data[m_ex_list_indexer(idx, cur_ex_idx)];
                if (cur_ex == cur_neigh)
                    {
                    excluded = true;
                    break;
                    }
                }

            // add it back to the list if it is not excluded
            if (!excluded)
                {
                h_nlist.data[myHead + new_n_neigh] = cur_neigh;
                new_n_neigh++;
                }
            }

        // update the number of neighbors
        h_n_neigh.data[idx] = new_n_neigh;
        }
    }

/*!
 * Iterates through each particle, and calculates a running sum of the starting index for that
 * particle in the flat array of neighbors.
 *
 * \note The neighbor list is also resized when it requires more memory than is currently allocated.
 */
void NeighborList::buildHeadList()
    {
    size_t headAddress = 0;
        {
        ArrayHandle<size_t> h_head_list(m_head_list, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < m_pdata->getN(); ++i)
            {
            h_head_list.data[i] = headAddress;

            // move the head address along
            unsigned int myType = __scalar_as_int(h_pos.data[i].w);
            headAddress += h_Nmax.data[myType];
            }
        }

    resizeNlist(headAddress);
    }

/*!
 * \param size the requested number of elements in the neighbor list
 *
 * Increases the size of the neighbor list memory using amortized resizing (growth factor: 9/8)
 * only when needed.
 */
void NeighborList::resizeNlist(size_t size)
    {
    if (size > m_nlist.getNumElements())
        {
        m_exec_conf->msg->notice(6)
            << "nlist: (Re-)allocating neighbor list, new size " << size << " uints " << endl;

        size_t alloc_size = m_nlist.getNumElements() ? m_nlist.getNumElements() : 1;

        while (size > alloc_size)
            {
            alloc_size = ((size_t)(((float)alloc_size) * 1.125f)) + 1;
            }

        // round up to nearest multiple of 4
        alloc_size = (alloc_size > 4) ? (alloc_size + 3) & ~3 : 4;

        m_nlist.resize(alloc_size);
        }
    }

/*!
 * \returns true if an overflow is detected for any particle type
 * \returns false if all particle types have enough memory for their neighbors
 *
 * The maximum number of neighbors per particle (rounded up to the nearest 4, min of 4) is
 * recomputed when an overflow happens.
 */
bool NeighborList::checkConditions()
    {
    bool result = false;

    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
        {
        if (h_conditions.data[i] > h_Nmax.data[i])
            {
            h_Nmax.data[i] = (h_conditions.data[i] > 4) ? (h_conditions.data[i] + 3) & ~3 : 4;
            result = true;
            }
        }

    return result;
    }

void NeighborList::resetConditions()
    {
    ArrayHandle<unsigned int> h_conditions(m_conditions,
                                           access_location::host,
                                           access_mode::overwrite);
    memset(h_conditions.data, 0, sizeof(unsigned int) * m_pdata->getNTypes());
    }

void NeighborList::growExclusionList()
    {
    unsigned int new_height = m_ex_list_indexer.getH() + 1;

    m_ex_list_tag.resize(m_pdata->getRTags().size(), new_height);
    m_ex_list_idx.resize(m_pdata->getMaxN(), new_height);

    // update the indexers
    m_ex_list_indexer = Index2D((unsigned int)m_ex_list_idx.getPitch(), new_height);
    m_ex_list_indexer_tag = Index2D((unsigned int)m_ex_list_tag.getPitch(), new_height);

    // we didn't copy data for the new idx list, force an update so it will be correct
    forceUpdate();
    }

#ifdef ENABLE_MPI
//! Returns true if the particle migration criterion is fulfilled
/*! \note The criterion for when to request particle migration is the same as the one for neighbor
   list rebuilds, which is implemented in needsUpdating().
 */
bool NeighborList::peekUpdate(uint64_t timestep)
    {
    bool result = needsUpdating(timestep);

    return result;
    }
#endif

#ifdef ENABLE_HIP
//! Update GPU memory locality
void NeighborList::updateMemoryMapping()
    {
#ifdef _HIP_PLATFORM_NVCC__
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        const GPUPartition& gpu_partition = m_pdata->getGPUPartition();

        // stash this partition for the future, so we can unset hints again
        m_last_gpu_partition = gpu_partition;

            // split preferred location of neighbor list across GPUs
            {
            ArrayHandle<size_t> h_head_list(m_head_list, access_location::host, access_mode::read);

            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                auto range = gpu_partition.getRange(idev);

                size_t start = h_head_list.data[range.first];
                unsigned int end = (range.second == m_pdata->getN())
                                       ? m_nlist.getNumElements()
                                       : h_head_list.data[range.second];

                if (end - start > 0)
                    // set preferred location
                    cudaMemAdvise(m_nlist.get() + h_head_list.data[range.first],
                                  sizeof(unsigned int) * (end - start),
                                  cudaMemAdviseSetPreferredLocation,
                                  gpu_map[idev]);
                }
            }
        CHECK_CUDA_ERROR();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // set preferred location
            auto range = gpu_partition.getRange(idev);
            unsigned int nelem = range.second - range.first;

            if (nelem == 0)
                continue;

            cudaMemAdvise(m_head_list.get() + range.first,
                          sizeof(size_t) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_n_neigh.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_last_pos.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);

            // pin to that device by prefetching
            cudaMemPrefetchAsync(m_head_list.get() + range.first,
                                 sizeof(size_t) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_n_neigh.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_last_pos.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif // __HIP_PLATFORM_NVCC__
    }
#endif

pybind11::array_t<uint32_t> NeighborList::getLocalPairListPython(uint64_t timestep)
    {
    compute(timestep);

    bool third_law = getStorageMode() == NeighborList::half;

    ArrayHandle<unsigned int> h_n_neigh(getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<size_t> h_head_list(getHeadList(), access_location::host, access_mode::read);

    auto* pair_list = new std::vector<vec2<uint32_t>>();

    // We can get an upper bound for the number of pairs by accumulating n_neigh,
    // but without explicitly checking tags we cannot be sure of the number of duplicates
    size_t max_n_elem = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        max_n_elem += h_n_neigh.data[i];
        }

    pair_list->reserve(max_n_elem);

    pybind11::capsule delete_when_done(pair_list,
                                       [](void* f)
                                       {
                                           auto data
                                               = reinterpret_cast<std::vector<vec2<uint32_t>>*>(f);
                                           delete data;
                                       });

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        const size_t my_head = h_head_list.data[i];
        const unsigned int size = h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            const unsigned int j = h_nlist.data[my_head + k];
            // if j is not a ghost, only accept it if i < j
            if (!third_law && j < m_pdata->getN() && i > j)
                continue;

            pair_list->push_back(vec2(i, j));
            }
        }

    pair_list->shrink_to_fit();

    unsigned long shape[] = {pair_list->size(), 2};
    unsigned long stride[] = {2 * sizeof(uint32_t), sizeof(uint32_t)};

    return pybind11::array_t(shape,                        // shape
                             stride,                       // stride
                             (uint32_t*)pair_list->data(), // data pointer
                             delete_when_done);
    }

pybind11::object NeighborList::getPairListPython(uint64_t timestep)
    {
    compute(timestep);

    bool third_law = getStorageMode() == NeighborList::half;

#ifdef ENABLE_MPI
    bool root = m_exec_conf->isRoot();
#endif

    ArrayHandle<unsigned int> h_n_neigh(getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<size_t> h_head_list(getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_tags(m_pdata->getTags(), access_location::host, access_mode::read);

    // We can get an upper bound for the number of pairs by accumulating n_neigh,
    // but without explicitly checking tags we cannot be sure of the number of duplicates
    size_t max_n_elem = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        max_n_elem += h_n_neigh.data[i];
        }

    auto* pair_list = new std::vector<vec2<uint32_t>>();
    pair_list->reserve(max_n_elem);

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int tag_i = h_tags.data[i];
        const size_t my_head = h_head_list.data[i];
        const unsigned int size = h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            const unsigned int j = h_nlist.data[my_head + k];
            const unsigned int tag_j = h_tags.data[j];
            if ((!third_law || j >= m_pdata->getN()) && tag_i > tag_j)
                continue;

            pair_list->push_back(vec2(tag_i, tag_j));
            }
        }

    pair_list->shrink_to_fit();

#ifdef ENABLE_MPI

    uint32_t* global_pair_list = nullptr;
    pybind11::capsule delete_when_done;
    unsigned long shape[] = {0, 2}; // first index needs to be set later
    unsigned long stride[] = {2 * sizeof(uint32_t), sizeof(uint32_t)};

    if (m_sysdef->isDomainDecomposed())
        {
        max_n_elem = pair_list->size();
        // Throw error if n_elem does not fit into an int
        if (max_n_elem > std::numeric_limits<int>::max() / 2)
            throw std::runtime_error("The pair list is too large to be communicated with MPI. \
                                        Please use the rank local pair list.");
        auto n_elem = static_cast<int>(2 * max_n_elem);
        std::vector<int> global_n_elem;
        void* recvbuf = nullptr;
        if (root)
            {
            global_n_elem = std::vector<int>(m_exec_conf->getNRanks(), 0);
            recvbuf = global_n_elem.data();
            }

        // Use a gather command to get the number of pairs on each processor
        MPI_Gather(&n_elem, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, m_exec_conf->getMPICommunicator());

        // Allocate array to hold the offsets
        std::vector<int> offsets;
        int* recvcount = nullptr;
        int* displs = nullptr;
        long total_elements = 0;
        if (root)
            {
            offsets = std::vector<int>(m_exec_conf->getNRanks(), 0);
            total_elements = std::accumulate(global_n_elem.begin(), global_n_elem.end(), 0);
            std::partial_sum(global_n_elem.begin(), global_n_elem.end() - 1, offsets.begin() + 1);
            recvcount = global_n_elem.data();
            displs = offsets.data();
            }

        if (total_elements > std::numeric_limits<int>::max())
            throw std::runtime_error("The pair list is too large to be communicated with MPI. \
                                      Please use the rank local pair list.");

        // Use a gatherv command to produce the global_pair_list
        if (root)
            {
            global_pair_list = new unsigned int[total_elements];
            delete_when_done = pybind11::capsule(global_pair_list,
                                                 [](void* f)
                                                 {
                                                     auto data = reinterpret_cast<uint32_t*>(f);
                                                     delete[] data;
                                                 });
            }

        shape[0] = total_elements / 2;

        MPI_Gatherv(pair_list->data(),
                    n_elem,
                    MPI_UINT32_T,
                    global_pair_list,
                    recvcount,
                    displs,
                    MPI_UINT32_T,
                    0,
                    m_exec_conf->getMPICommunicator());

        // cleanup allocations
        delete pair_list;
        }
    else
        {
        // Simulation is not domain decomposed.
        global_pair_list = reinterpret_cast<uint32_t*>(pair_list->data());
        shape[0] = pair_list->size();
        delete_when_done
            = pybind11::capsule(pair_list,
                                [](void* f)
                                {
                                    auto data = reinterpret_cast<std::vector<vec2<uint32_t>>*>(f);
                                    delete data;
                                });
        }

    if (root)
        {
        return pybind11::array_t(shape, stride, global_pair_list, delete_when_done);
        }
    return pybind11::none();

#else
    auto* global_pair_list = pair_list;

    pybind11::capsule delete_when_done(pair_list,
                                       [](void* f)
                                       {
                                           auto data
                                               = reinterpret_cast<std::vector<vec2<uint32_t>>*>(f);
                                           delete data;
                                       });

    unsigned long shape[] = {global_pair_list->size(), 2};
    unsigned long stride[] = {2 * sizeof(uint32_t), sizeof(uint32_t)};

    return pybind11::array_t(shape,
                             stride,
                             reinterpret_cast<uint32_t*>(global_pair_list->data()),
                             delete_when_done);
#endif
    }

void NeighborList::validateTypes(unsigned int typ1, unsigned int typ2, std::string action)
    {
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        throw std::runtime_error("Error in" + action + " for NeighborList. Invalid type");
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
void NeighborList::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "setting rcut_base");
        {
        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_rcut_base(m_rcut_base, access_location::host, access_mode::readwrite);
        h_rcut_base.data[m_typpair_idx(typ1, typ2)] = rcut;
        h_rcut_base.data[m_typpair_idx(typ2, typ1)] = rcut;
        }

    notifyRCutMatrixChange();
    }

void NeighborList::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRcut(typ1, typ2, r_cut);
    }

Scalar NeighborList::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting rcut_base.");
    ArrayHandle<Scalar> h_rcut_base(m_rcut_base, access_location::host, access_mode::read);
    return h_rcut_base.data[m_typpair_idx(typ1, typ2)];
    }

namespace detail
    {
void export_NeighborList(pybind11::module& m)
    {
    pybind11::class_<NeighborList, Compute, std::shared_ptr<NeighborList>> nlist(m, "NeighborList");
    nlist.def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("buffer", &NeighborList::getRBuff, &NeighborList::setRBuff)
        .def_property("rebuild_check_delay",
                      &NeighborList::getRebuildCheckDelay,
                      &NeighborList::setRebuildCheckDelay)
        .def_property("check_dist", &NeighborList::getDistCheck, &NeighborList::setDistCheck)
        .def("setStorageMode", &NeighborList::setStorageMode)
        .def_property("exclusions", &NeighborList::getExclusions, &NeighborList::setExclusions)
        .def("addMesh", &NeighborList::AddMesh)
        .def("getMaxRCut", &NeighborList::getMaxRCut)
        .def("getMinRCut", &NeighborList::getMinRCut)
        .def("getMaxRList", &NeighborList::getMaxRList)
        .def("getMinRList", &NeighborList::getMinRList)
        .def("forceUpdate", &NeighborList::forceUpdate)
        .def("getSmallestRebuild", &NeighborList::getSmallestRebuild)
        .def("getNumUpdates", &NeighborList::getNumUpdates)
        .def("getNumExclusions", &NeighborList::getNumExclusions)
        .def_property_readonly("num_builds", &NeighborList::getNumUpdates)
        .def("getLocalPairList", &NeighborList::getLocalPairListPython)
        .def("getPairList", &NeighborList::getPairListPython)
        .def("setRCut", &NeighborList::setRCutPython)
        .def("getRCut", &NeighborList::getRCut)
        .def("compute", &NeighborList::compute);

    pybind11::enum_<NeighborList::storageMode>(nlist, "storageMode")
        .value("half", NeighborList::storageMode::half)
        .value("full", NeighborList::storageMode::full)
        .export_values();
    };

void export_LocalNeighborListDataHost(pybind11::module& m)
    {
    export_LocalNeighborListData<HOOMDHostBuffer>(m, "LocalNeighborListDataHost");
    };

#if ENABLE_HIP
void export_LocalNeighborListDataGPU(pybind11::module& m)
    {
    export_LocalNeighborListData<HOOMDDeviceBuffer>(m, "LocalNeighborListDataDevice");
    };
#endif

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
