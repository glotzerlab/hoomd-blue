/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <boost/bind.hpp>

#include "NeighborList.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <sstream>
#include <fstream>

#include <iostream>
#include <stdexcept>

using namespace boost;
using namespace std;

/*! \file NeighborList.cc
    \brief Defines the NeighborList class
*/

/*! \param sysdef System the neighborlist is to compute neighbors for
    \param r_cut Cuttoff radius under which particles are considered neighbors
    \param r_buff Buffere radius around \a r_cut in which neighbors will be included

    \post NeighborList is initialized and the list memory has been allocated,
        but the list will not be computed until compute is called.
    \post The storage mode defaults to half
*/
NeighborList::NeighborList(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff)
    : Compute(sysdef), m_r_cut(r_cut), m_r_buff(r_buff), m_d_max(1.0), m_filter_body(false), m_filter_diameter(false),
      m_storage_mode(half), m_updates(0), m_forced_updates(0), m_dangerous_updates(0),
      m_force_update(true), m_dist_check(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing Neighborlist" << endl;

    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        m_exec_conf->msg->error() << "nlist: Requested cuttoff radius is less than zero" << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    if (m_r_buff < 0.0)
        {
        m_exec_conf->msg->error() << "nlist: Requested buffer radius is less than zero" << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    // initialize values
    m_last_updated_tstep = 0;
    m_last_checked_tstep = 0;
    m_last_check_result = true;
    m_every = 0;
    m_Nmax = 0;
    m_exclusions_set = false;

 
    // initialize box length at last update
    m_last_L = m_pdata->getGlobalBox().getL();

    // allocate conditions flags
    GPUFlags<unsigned int> conditions(exec_conf);
    m_conditions.swap(conditions);

    // allocate m_n_neigh
    GPUArray<unsigned int> n_neigh(m_pdata->getMaxN(), exec_conf);
    m_n_neigh.swap(n_neigh);

    // allocate neighbor list
    allocateNlist();

    // allocate m_last_pos
    GPUArray<Scalar4> last_pos(m_pdata->getMaxN(), exec_conf);
    m_last_pos.swap(last_pos);
   
    // allocate initial memory allowing 4 exclusions per particle (will grow to match specified exclusions)
    GPUArray<unsigned int> n_ex_tag(m_pdata->getNGlobal(), exec_conf);
    m_n_ex_tag.swap(n_ex_tag);
    GPUArray<unsigned int> n_ex_idx(m_pdata->getMaxN(), exec_conf);
    m_n_ex_idx.swap(n_ex_idx);
    GPUArray<unsigned int> ex_list_tag(m_pdata->getNGlobal(), 1, exec_conf);
    m_ex_list_tag.swap(ex_list_tag);
    GPUArray<unsigned int> ex_list_idx(m_pdata->getMaxN(), 1, exec_conf);
    m_ex_list_idx.swap(ex_list_idx);
    m_ex_list_indexer = Index2D(m_ex_list_idx.getPitch(), 1);
    m_ex_list_indexer_tag = Index2D(m_ex_list_tag.getPitch(), 1);
    
    m_sort_connection = m_pdata->connectParticleSort(bind(&NeighborList::forceUpdate, this));

    m_max_particle_num_change_connection = m_pdata->connectMaxParticleNumberChange(bind(&NeighborList::reallocate, this));

    // allocate m_update_periods tracking info
    m_update_periods.resize(100);
    for (unsigned int i = 0; i < m_update_periods.size(); i++)
        m_update_periods[i] = 0;
    }

/*! Reallocate internal data structures upon change of local maximum particle number
 */
void NeighborList::reallocate()
    {

    m_last_pos.resize(m_pdata->getMaxN());
    m_n_ex_idx.resize(m_pdata->getMaxN());
    unsigned int ex_list_height = m_ex_list_indexer.getH();
    m_ex_list_idx.resize(m_pdata->getMaxN(), ex_list_height );
    m_ex_list_indexer = Index2D(m_ex_list_idx.getPitch(), ex_list_height);

    m_nlist_indexer = Index2D(m_nlist.getPitch(), m_Nmax);

    m_nlist.resize(m_pdata->getMaxN(), m_Nmax+1);
    m_n_neigh.resize(m_pdata->getMaxN());
    }

NeighborList::~NeighborList()
    {
    m_exec_conf->msg->notice(5) << "Destroying Neighborlist" << endl;

    m_sort_connection.disconnect();
    m_max_particle_num_change_connection.disconnect();
#ifdef ENABLE_MPI
    if (m_migrate_request_connection.connected())
        m_migrate_request_connection.disconnect();
#endif
    }

/*! Updates the neighborlist if it has not yet been updated this times step
    \param timestep Current time step of the simulation
*/
void NeighborList::compute(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!shouldCompute(timestep) && !m_force_update)
        return;
        
    if (m_prof) m_prof->push("Neighbor");

    // update the exclusion data if this is a forced update
    if (m_force_update)
        {
        if (m_exclusions_set)
            updateExListIdx();
        }

    // check if the list needs to be updated and update it
    if (needsUpdating(timestep))
        {
        // rebuild the list until there is no overflow
        bool overflowed = false;
        do
            {
            buildNlist(timestep);

            overflowed = checkConditions();
            // if we overflowed, need to reallocate memory and reset the conditions
            if (overflowed)
                {
                allocateNlist();
                resetConditions();
                }
            } while (overflowed);
        
        if (m_exclusions_set)
            filterNlist();
        
        setLastUpdatedPos();
        }

    if (m_prof) m_prof->pop();
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls buildNlist repeatedly to benchmark the neighbor list.
*/
double NeighborList::benchmark(unsigned int num_iters)
    {
    ClockSource t;
    // warm up run
    forceUpdate();
    compute(0);
    buildNlist(0);
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        buildNlist(0);
        
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

/*! \param r_cut New cuttoff radius to set
    \param r_buff New buffer radius to set
    \note Changing the cuttoff radius does NOT immeadiately update the neighborlist.
            The new cuttoff will take effect when compute is called for the next timestep.
*/
void NeighborList::setRCut(Scalar r_cut, Scalar r_buff)
    {
    m_r_cut = r_cut;
    m_r_buff = r_buff;
    
    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        m_exec_conf->msg->error() << "nlist: Requested cuttoff radius is less than zero" << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }
        
    if (m_r_buff < 0.0)
        {
        m_exec_conf->msg->error() << "nlist: Requested buffer radius is less than zero" << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        Scalar rmax = m_r_cut + m_r_buff;
        // add d_max - 1.0 all the time - this is needed so that all interacting slj particles are communicated
        rmax += m_d_max - Scalar(1.0);
        m_comm->setGhostLayerWidth(rmax);
        m_comm->setRBuff(m_r_buff);
        }
#endif
    forceUpdate();
    }

/*! \returns an estimate of the number of neighbors per particle
    This mean-field estimate may be very bad dending on how clustered particles are.
    Derived classes can override this method to provide better estimates.

    \note Under NO circumstances should calling this method produce any
    appreciable amount of overhead. This is mainly a warning to
    derived classes.
*/
Scalar NeighborList::estimateNNeigh()
    {
    // calculate a number density of particles
    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();
    Scalar vol = L.x * L.y * L.z;
    Scalar n_dens = Scalar(m_pdata->getN()) / vol;
    
    // calculate the average number of neighbors by multiplying by the volume
    // within the cutoff
    Scalar r_max = m_r_cut + m_r_buff;
    Scalar vol_cut = Scalar(4.0/3.0 * M_PI) * r_max * r_max * r_max;
    return n_dens * vol_cut;
    }

/*! \param tag1 TAG (not index) of the first particle in the pair
    \param tag2 TAG (not index) of the second particle in the pair
    \post The pair \a tag1, \a tag2 will not appear in the neighborlist
    \note This only takes effect on the next call to compute() that updates the list
    \note Duplicates are checked for and not added.
*/
void NeighborList::addExclusion(unsigned int tag1, unsigned int tag2)
    {
    assert(tag1 < m_pdata->getNGlobal());
    assert(tag2 < m_pdata->getNGlobal());
    
    m_exclusions_set = true;

    // don't add an exclusion twice
    if (isExcluded(tag1, tag2))
        return;
    
    // this is clunky, but needed due to the fact that we cannot have an array handle in scope when
    // calling grow exclusion list
    bool grow = false;
        {
        // access arrays
        ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::readwrite);
    
        // grow the list if necessary
        if (h_n_ex_tag.data[tag1] == m_ex_list_indexer.getH())
            grow = true;
        
        if (h_n_ex_tag.data[tag2] == m_ex_list_indexer.getH())
            grow = true;
        }
        
    if (grow)
        growExclusionList();

        {
        // access arrays
        ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::readwrite);
    
        // add tag2 to tag1's exculsion list
        unsigned int pos1 = h_n_ex_tag.data[tag1];
        assert(pos1 < m_ex_list_indexer.getH());
        h_ex_list_tag.data[m_ex_list_indexer_tag(tag1,pos1)] = tag2;
        h_n_ex_tag.data[tag1]++;
        
        // add tag1 to tag2's exclusion list
        unsigned int pos2 = h_n_ex_tag.data[tag2];
        assert(pos2 < m_ex_list_indexer.getH());
        h_ex_list_tag.data[m_ex_list_indexer_tag(tag2,pos2)] = tag1;
        h_n_ex_tag.data[tag2]++;
        }
    
    forceUpdate();
    }

/*! \post No particles are excluded from the neighbor list
*/
void NeighborList::clearExclusions()
    {
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::overwrite);

    memset(h_n_ex_tag.data, 0, sizeof(unsigned int)*m_pdata->getN());
    memset(h_n_ex_idx.data, 0, sizeof(unsigned int)*m_pdata->getN());
    m_exclusions_set = false;

    forceUpdate();
    }

//! Get number of exclusions involving n particles
unsigned int NeighborList::getNumExclusions(unsigned int size)
    {
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    unsigned int count = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int num_excluded = h_n_ex_tag.data[i];

        if (num_excluded == size) count++;
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &count,
                      1,
                      MPI_INT,
                      MPI_SUM, 
                      m_exec_conf->getMPICommunicator());
        }
    #endif

    return count;
    }

/*! \post Gather some statistics about exclusions usage.
*/
void NeighborList::countExclusions()
    {
    unsigned int MAX_COUNT_EXCLUDED = 16;
    unsigned int excluded_count[MAX_COUNT_EXCLUDED+2];
    unsigned int num_excluded, max_num_excluded;
    
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    
    max_num_excluded = 0;
    for (unsigned int c=0; c <= MAX_COUNT_EXCLUDED+1; ++c)
        excluded_count[c] = 0;
        
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        num_excluded = h_n_ex_tag.data[i];
        
        if (num_excluded > max_num_excluded)
            max_num_excluded = num_excluded;
        
        if (num_excluded > MAX_COUNT_EXCLUDED)
            num_excluded = MAX_COUNT_EXCLUDED + 1;
        
        excluded_count[num_excluded] += 1;
        }

    m_exec_conf->msg->notice(2) << "-- Neighborlist exclusion statistics -- :" << endl;
    for (unsigned int i=0; i <= MAX_COUNT_EXCLUDED; ++i)
        {
        if (excluded_count[i] > 0)
            m_exec_conf->msg->notice(2) << "Particles with " << i << " exclusions             : " << excluded_count[i] << endl;
        }

    if (excluded_count[MAX_COUNT_EXCLUDED+1])
        {
        m_exec_conf->msg->notice(2) << "Particles with more than " << MAX_COUNT_EXCLUDED << " exclusions: "
             << excluded_count[MAX_COUNT_EXCLUDED+1] << endl;
        }

    if (m_filter_diameter)
        m_exec_conf->msg->notice(2) << "Neighbors excluded by diameter (slj)    : yes" << endl;
    else
        m_exec_conf->msg->notice(2) << "Neighbors excluded by diameter (slj)    : no" << endl;

    if (m_filter_body)
        m_exec_conf->msg->notice(2) << "Neighbors excluded when in the same body: yes" << endl;
    else
        m_exec_conf->msg->notice(2) << "Neighbors excluded when in the same body: no" << endl;
        
    if (!m_filter_body && m_sysdef->getRigidData()->getNumBodies() > 0)
        {
        m_exec_conf->msg->warning() << "Disabling the body exclusion will cause rigid bodies to behave erratically" << endl
             << "            unless inter-body pair forces are very small." << endl;
        }
    }

/*! After calling addExclusionFromBonds() all bonds specified in the attached ParticleData will be
    added as exlusions. Any additional bonds added after this will not be automatically added as exclusions.
*/
void NeighborList::addExclusionsFromBonds()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

    // access bond data by snapshot
    SnapshotBondData snapshot(bond_data->getNumBondsGlobal());
    bond_data->takeSnapshot(snapshot);

    // broadcast global bond list
    std::vector<uint2> bonds;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_exec_conf->getRank() == 0)
            bonds = snapshot.bonds;

        bcast(bonds, 0, m_exec_conf->getMPICommunicator());
        }
    else
        bonds = snapshot.bonds;
#else
    bonds = snapshot.bonds;
#endif

    // for each bond
    for (unsigned int i = 0; i < bonds.size(); i++)
        // add an exclusion
        addExclusion(bonds[i].x, bonds[i].y);
    }

/*! After calling addExclusionsFromAngles(), all angles specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the angle are excluded from interacting.
*/
void NeighborList::addExclusionsFromAngles()
    {
    boost::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();
    
    // for each bond
    for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
        {
        Angle angle = angle_data->getAngle(i);
        addExclusion(angle.a, angle.c);
        }
    }

/*! After calling addExclusionsFromAngles(), all dihedrals specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the dihedral are excluded from interacting.
*/
void NeighborList::addExclusionsFromDihedrals()
    {
    boost::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();
    
    // for each bond
    for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
        {
        Dihedral dihedral = dihedral_data->getDihedral(i);
        addExclusion(dihedral.a, dihedral.d);
        }
    }

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 have been excluded from the neighbor list
*/
bool NeighborList::isExcluded(unsigned int tag1, unsigned int tag2)
    {
    assert(tag1 < m_pdata->getNGlobal());
    assert(tag2 < m_pdata->getNGlobal());
        
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag, access_location::host, access_mode::read);
    
    unsigned int n_ex = h_n_ex_tag.data[tag1];
    for (unsigned int i = 0; i < n_ex; i++)
        {
        if (h_ex_list_tag.data[m_ex_list_indexer_tag(tag1,i)] == tag2)
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
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        m_exec_conf->msg->warning() << "nlist: No bonds defined while trying to add topology derived 1-3 exclusions" << endl;
        return;
        }
        
    // build a per atom list with all bonding partners from the list of bonds.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            m_exec_conf->msg->error() << "nlist: Too many bonds to process exclusions for particle with tag: " << tagA << endl;
            m_exec_conf->msg->error() << "Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            m_exec_conf->msg->error() << "nlist: Too many bonds to process exclusions for particle with tag: " << tagB << endl;
            m_exec_conf->msg->error() << "Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    // now loop over the atoms and build exclusions if we have more than
    // one bonding partner, i.e. we are in the center of an angle.
    for (unsigned int i = 0; i < myNAtoms; i++)
        {
        // now, loop over all atoms, and find those in the middle of an angle
        const unsigned int iAtom = i*MAXNBONDS;
        const unsigned int nBonds = localBondList[iAtom];
        
        if (nBonds > 1) // need at least two bonds
            {
            for (unsigned int j = 1; j < nBonds; ++j)
                {
                for (unsigned int k = j+1; k <= nBonds; ++k)
                    addExclusion(localBondList[iAtom+j],localBondList[iAtom+k]);
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
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        m_exec_conf->msg->warning() << "nlist: No bonds defined while trying to add topology derived 1-4 exclusions" << endl;
        return;
        }
        
    // allocate and clear data.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            m_exec_conf->msg->error() << "nlist: Too many bonds to process exclusions for particle with tag: " << tagA << endl;
            m_exec_conf->msg->error() << "Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            m_exec_conf->msg->error() << "nlist: Too many bonds to process exclusions for particle with tag: " << tagB << endl;
            m_exec_conf->msg->error() << "Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    //  loop over all bonds
    for (unsigned int i = 0; i < nBonds; i++)
        {
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        const unsigned int nBondsA = localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = localBondList[tagB*MAXNBONDS];
        
        for (unsigned int j = 1; j <= nBondsA; j++)
            {
            const unsigned int tagJ = localBondList[tagA*MAXNBONDS+j];
            if (tagJ == tagB) // skip the bond in the middle of the dihedral
                continue;
                
            for (unsigned int k = 1; k <= nBondsB; k++)
                {
                const unsigned int tagK = localBondList[tagB*MAXNBONDS+k];
                if (tagK == tagA) // skip the bond in the middle of the dihedral
                    continue;
                    
                addExclusion(tagJ,tagK);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }


/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since the last call
        to this method that returned true.
    \returns false If none of the particles has been moved more than 1/2 of the buffer distance since the last call to this
        method that returned true.

    Note: this method relies on data set by setLastUpdatedPos(), which must be called to set the previous data used
    in the next call to distanceCheck();
*/
bool NeighborList::distanceCheck()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // temporary storage for the result
    bool result = false;
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::read);
    
    // get current global box lengths
    Scalar3 L_g = m_pdata->getGlobalBox().getL();

    // Cutoff distance for inclusion in neighbor list
    Scalar rmax = m_r_cut + m_r_buff;
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);

    // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
    Scalar3 lambda = L_g / m_last_L;
    Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
    lambda_min = (lambda_min < lambda.z) ? lambda_min : lambda.z;

    // maximum displacement for each particle (after subtraction of homogeneous dilations)
    Scalar delta_max = (rmax*lambda_min - m_r_cut)/Scalar(2.0);
    Scalar maxsq = delta_max > 0  ? delta_max*delta_max : 0;

    bool boundary_check = false;
    bool out_of_bounds = false;
    unsigned int out_of_bounds_idx = 0;
#ifdef ENABLE_MPI
    boundary_check = m_pdata->getDomainDecomposition();
#endif

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        Scalar3 dx = make_scalar3(h_pos.data[i].x - lambda.x*h_last_pos.data[i].x,
                                  h_pos.data[i].y - lambda.y*h_last_pos.data[i].y,
                                  h_pos.data[i].z - lambda.z*h_last_pos.data[i].z);
        
        dx = box.minImage(dx);
        
        if (dot(dx, dx) >= maxsq)
            {
            result = true;
            break;
            }
        if (boundary_check)
            if (box.checkOutOfBounds(dx))
                {
                out_of_bounds = true;
                out_of_bounds_idx = i;
                break;
                }
        }

    if (boundary_check && out_of_bounds)
        {
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        unsigned int tag = h_tag.data[out_of_bounds_idx];
        m_exec_conf->msg->error() << "nlist: Particle " << tag << " has traveled more than one box length"
                                  << std::endl << "between neighbor list builds."
                                  << std::endl << std::endl;
        throw std::runtime_error("Error checking particle displacements");
        }

    // don't worry about computing flops here, this is fast
    if (m_prof) m_prof->pop();

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // use MPI all_reduce to check if the neighbor list build criterium is fulfilled on any processor
        int local_result = result ? 1 : 0;
        int global_result = 0;
        MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, m_exec_conf->getMPICommunicator());
        result = (global_result > 0);
        }
#endif

    return result;
    }

/*! Copies the current positions of all particles over to m_last_x etc...
*/
void NeighborList::setLastUpdatedPos()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // update the last position arrays
    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_last_pos.data[i] = make_scalar4(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z, Scalar(0.0));
        }
   
    // update last box length
    m_last_L = m_pdata->getGlobalBox().getL();

    if (m_prof) m_prof->pop();
    }

/*! \returns true If the neighbor list needs to be updated
    \returns false If the neighbor list does not need to be updated
    \note This is designed to be called if (needsUpdating()) then update every step.
        It internally handles many state variables that rely on this assumption.

    \param timestep Current time step in the simulation
*/
bool NeighborList::needsUpdating(unsigned int timestep)
    {
    if (m_last_checked_tstep == timestep)
        if (!m_force_update || m_last_check_result)
            {
            // reset just in case the update has been forced after the last check has returned true
            m_force_update = false;
            return m_last_check_result;
            }

    m_last_checked_tstep = timestep;

    if (timestep < (m_last_updated_tstep + m_every) && !m_force_update)
        {
        m_last_check_result = false;
        return false;
        }

    // temporary storage for return result
    bool result = false;

    // check if this is a dangerous time
    // we are dangerous if m_every is greater than 1 and this is the first check after the
    // last build
    bool dangerous = false;
    if (m_dist_check && (m_every > 1 && timestep == (m_last_updated_tstep + m_every)))
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
        // it also needs to be updated if m_every is 0, or the check period is hit when distance checks are disabled
        if (m_r_buff < 1e-6 ||
            (!m_dist_check && (m_every == 0 || (m_every > 1 && timestep == (m_last_updated_tstep + m_every)))))
            {
            result = true;
            }
        else
            {
            result = distanceCheck();
            }
        
        if (result)
            {
            // record update histogram - but only if the period is positive
            if (timestep > m_last_updated_tstep)
                {
                unsigned int period = timestep - m_last_updated_tstep;
                if (period >= m_update_periods.size())
                    period = m_update_periods.size()-1;
                m_update_periods[period]++;
                }

            m_last_updated_tstep = timestep;
            m_updates += 1;
            }
        }
        
    // warn the user if this is a dangerous build
    if (result && dangerous)
        {
        m_exec_conf->msg->notice(2) << "nlist: Dangerous neighborlist build occured. Continuing this simulation may produce incorrect results and/or program crashes. Decrease the neighborlist check_period and rerun." << endl;
        m_dangerous_updates += 1;
        }
       
    m_last_check_result = result;
    return result;
    }

/*! Generic statistics that apply to any neighbor list, like the number of updates,
    average number of neighbors, etc... are printed to stdout. Derived classes should
    print any pertinient information they see fit to.
 */
void NeighborList::printStats()
    {
    // return earsly if the notice level is less than 1
    if (m_exec_conf->msg->getNoticeLevel() < 1)
        return;

    m_exec_conf->msg->notice(1) << "-- Neighborlist stats:" << endl;
    m_exec_conf->msg->notice(1) << m_updates << " normal updates / " << m_forced_updates << " forced updates / " << m_dangerous_updates << " dangerous updates" << endl;

    // access the number of neighbors to generate stats
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::read);

    // build some simple statistics of the number of neighbors
    unsigned int n_neigh_min = m_pdata->getN();
    unsigned int n_neigh_max = 0;
    Scalar n_neigh_avg = 0.0;

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int n_neigh = (unsigned int)h_n_neigh.data[i];
        if (n_neigh < n_neigh_min)
            n_neigh_min = n_neigh;
        if (n_neigh > n_neigh_max)
            n_neigh_max = n_neigh;

        n_neigh_avg += Scalar(n_neigh);
        }

    // divide to get the average
    n_neigh_avg /= Scalar(m_pdata->getN());
    m_exec_conf->msg->notice(1) << "n_neigh_min: " << n_neigh_min << " / n_neigh_max: " << n_neigh_max << " / n_neigh_avg: " << n_neigh_avg << endl;

    m_exec_conf->msg->notice(1) << "shortest rebuild period: " << getSmallestRebuild() << endl;
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
    return m_update_periods.size();
    }

/*! Loops through the particles and finds all of the particles \c j who's distance is less than
    \param timestep Current time step of the simulation
    \c r_cut \c + \c r_buff from particle \c i, includes either i < j or all neighbors depending
    on the mode set by setStorageMode()
*/
void NeighborList::buildNlist(unsigned int timestep)
    {
    // sanity check
    assert(m_pdata);
    
    // start up the profile
    if (m_prof) m_prof->push("Build list");
    
    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);
    assert(h_diameter.data);
    assert(h_body.data);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getNearestPlaneDistance();
    
    // start by creating a temporary copy of r_cut sqaured
    Scalar rmax = m_r_cut + m_r_buff;
    // add d_max - 1.0, if diameter filtering is not already taking care of it
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);
    Scalar rmaxsq = rmax*rmax;
    
    if (L.x <= rmax * 2.0 || L.y <= rmax * 2.0 || L.z <= rmax * 2.0)
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << endl;
        throw runtime_error("Error updating neighborlist bins");
        }
        
    // access the nlist data
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);

    unsigned int conditions = 0;

    // start by clearing the entire list
    memset(h_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getN());
    
    // now we can loop over all particles in n^2 fashion and build the list
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar di = h_diameter.data[i];
        unsigned int bodyi = h_body.data[i];
        
        // for each other particle with i < j, including ghost particles
        for (unsigned int j = i + 1; j < m_pdata->getN() + m_pdata->getNGhosts(); j++)
            {
            // calculate dr
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pj - pi;

            dx = box.minImage(dx);
            
            bool excluded = false;
            if (m_filter_body && bodyi != NO_BODY)
                excluded = (bodyi == h_body.data[j]);

            Scalar sqshift = Scalar(0.0);
            if (m_filter_diameter)
                {
                // compute the shift in radius to accept neighbors based on their diameters
                Scalar delta = (di + h_diameter.data[j]) * Scalar(0.5) - Scalar(1.0);
                // r^2 < (r_max + delta)^2
                // r^2 < r_maxsq + delta^2 + 2*r_max*delta
                sqshift = (delta + Scalar(2.0) * rmax) * delta;
                }

            // now compare rsq to rmaxsq and add to the list if it meets the criteria
            Scalar rsq = dot(dx, dx);
            if (rsq <= (rmaxsq + sqshift) && !excluded)
                {
                if (m_storage_mode == full)
                    {
                    #pragma omp critical
                        {
                        unsigned int posi = h_n_neigh.data[i];
                        if (posi < m_Nmax)
                            h_nlist.data[m_nlist_indexer(i, posi)] = j;
                        else
                            conditions = max(conditions, h_n_neigh.data[i]+1);
                        
                        h_n_neigh.data[i]++;
                       
                        unsigned int posj = h_n_neigh.data[j];
                        if (posj < m_Nmax)
                            h_nlist.data[m_nlist_indexer(j, posj)] = i;
                        else
                            conditions = max(conditions, h_n_neigh.data[j]+1);

                        h_n_neigh.data[j]++;
                        }
                    }
                else
                    {
                    unsigned int pos = h_n_neigh.data[i];
                    
                    if (pos < m_Nmax)
                        h_nlist.data[m_nlist_indexer(i, pos)] = j;
                    else
                        conditions = max(conditions, h_n_neigh.data[i]+1);
                    
                    h_n_neigh.data[i]++;
                    }
                }
            }
        }
   
    // write out conditions
    m_conditions.resetFlags(conditions);

    if (m_prof) m_prof->pop();
    }

/*! Translates the exclusions set in \c m_n_ex_tag and \c m_ex_list_tag to indices in \c m_n_ex_idx and \c m_ex_list_idx
*/
void NeighborList::updateExListIdx()
    {
    if (m_prof)
        m_prof->push("update-ex");
    // access data
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_ex_list_idx(m_ex_list_idx, access_location::host, access_mode::overwrite);

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
            unsigned int ex_tag = h_ex_list_tag.data[m_ex_list_indexer_tag(tag,offset)];
            unsigned int ex_idx = h_rtag.data[ex_tag];

            // store excluded particle idx
            h_ex_list_idx.data[m_ex_list_indexer(idx, offset)] = ex_idx;
            }
        }
    
    if (m_prof)
        m_prof->pop();
    }

/*! Loops through the neighbor list and filters out any excluded pairs
*/
void NeighborList::filterNlist()
    {
    if (m_prof)
        m_prof->push("filter");
    
    // access data
    
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_idx(m_ex_list_idx, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::readwrite);
    
    // for each particle's neighbor list
    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        unsigned int n_neigh = h_n_neigh.data[idx];
        unsigned int n_ex = h_n_ex_idx.data[idx];
        unsigned int new_n_neigh = 0;
        
        // loop over the list, regenerating it as we go
        for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
            {
            unsigned int cur_neigh = h_nlist.data[m_nlist_indexer(idx, cur_neigh_idx)];
            
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
                h_nlist.data[m_nlist_indexer(idx, new_n_neigh)] = cur_neigh;
                new_n_neigh++;
                }
            }
        
        // update the number of neighbors
        h_n_neigh.data[idx] = new_n_neigh;
        }

    if (m_prof)
        m_prof->pop();
    }

void NeighborList::allocateNlist()
    {
    // round up to the nearest multiple of 8
    m_Nmax = m_Nmax + 8 - (m_Nmax & 7);

    m_exec_conf->msg->notice(6) << "nlist: (Re-)Allocating " << m_pdata->getMaxN() << " x " << m_Nmax+1 << endl;

    // re-allocate, overwriting old neighbor list
    GPUArray<unsigned int> nlist(m_pdata->getMaxN(), m_Nmax+1, exec_conf);
    m_nlist.swap(nlist);

    // update the indexer
    m_nlist_indexer = Index2D(m_nlist.getPitch(), m_Nmax);
    }

unsigned int NeighborList::readConditions()
    {
    return m_conditions.readFlags();
    }

bool NeighborList::checkConditions()
    {
    bool result = false;

    unsigned int conditions;
    conditions = readConditions();

    // up m_Nmax to the overflow value, reallocate memory and set the overflow condition
    if (conditions > m_Nmax)
        {
        m_Nmax = conditions;
        result = true;
        }

    return result;
    }

void NeighborList::resetConditions()
    {
    m_conditions.resetFlags(0);
    }

void NeighborList::growExclusionList()
    {
    unsigned int new_height = m_ex_list_indexer.getH() + 1;

    m_ex_list_tag.resize(m_pdata->getNGlobal(), new_height);
    m_ex_list_idx.resize(m_pdata->getMaxN(), new_height);

    // update the indexers
    m_ex_list_indexer = Index2D(m_ex_list_idx.getPitch(), new_height);
    m_ex_list_indexer_tag = Index2D(m_ex_list_tag.getPitch(), new_height);
    
    // we didn't copy data for the new idx list, force an update so it will be correct
    forceUpdate();
    }

#ifdef ENABLE_MPI
//! Set the communicator to use
void NeighborList::setCommunicator(boost::shared_ptr<Communicator> comm)
    {
    if (!m_comm)
        {
        // only add the migrate request on the first call
        comm->addMigrateRequest(bind(&NeighborList::peekUpdate, this, _1));
        }

    if (comm)
        {
        m_comm = comm;

        Scalar rmax = m_r_cut + m_r_buff;
        // add d_max - 1.0 all the time - this is needed so that all interacting slj particles are communicated
        rmax += m_d_max - Scalar(1.0);
        m_comm->setGhostLayerWidth(rmax);
        m_comm->setRBuff(m_r_buff);
        }
    }

//! Returns true if the particle migration criterium is fulfilled
/*! \note The criterium for when to request particle migration is the same as the one for neighbor list
    rebuilds, which is implemented in needsUpdating().
 */
bool NeighborList::peekUpdate(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Neighbor");

    bool result = needsUpdating(timestep);

    if (m_prof) m_prof->pop();

    return result; 
    }
#endif

void export_NeighborList()
    {
    class_< std::vector<unsigned int> >("std_vector_uint")
    .def(vector_indexing_suite<std::vector<unsigned int> >())
    .def("push_back", &std::vector<unsigned int>::push_back)
    ;
    
    scope in_nlist = class_<NeighborList, boost::shared_ptr<NeighborList>, bases<Compute>, boost::noncopyable >
                     ("NeighborList", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("setRCut", &NeighborList::setRCut)
                     .def("setEvery", &NeighborList::setEvery)
                     .def("setStorageMode", &NeighborList::setStorageMode)
                     .def("addExclusion", &NeighborList::addExclusion)
                     .def("clearExclusions", &NeighborList::clearExclusions)
                     .def("countExclusions", &NeighborList::countExclusions)
                     .def("addExclusionsFromBonds", &NeighborList::addExclusionsFromBonds)
                     .def("addExclusionsFromAngles", &NeighborList::addExclusionsFromAngles)
                     .def("addExclusionsFromDihedrals", &NeighborList::addExclusionsFromDihedrals)
                     .def("addOneThreeExclusionsFromTopology", &NeighborList::addOneThreeExclusionsFromTopology)
                     .def("addOneFourExclusionsFromTopology", &NeighborList::addOneFourExclusionsFromTopology)
                     .def("setFilterBody", &NeighborList::setFilterBody)
                     .def("getFilterBody", &NeighborList::getFilterBody)
                     .def("setFilterDiameter", &NeighborList::setFilterDiameter)
                     .def("getFilterDiameter", &NeighborList::getFilterDiameter)
                     .def("setMaximumDiameter", &NeighborList::setMaximumDiameter)
                     .def("forceUpdate", &NeighborList::forceUpdate)
                     .def("estimateNNeigh", &NeighborList::estimateNNeigh)
                     .def("getSmallestRebuild", &NeighborList::getSmallestRebuild)
                     .def("getNumUpdates", &NeighborList::getNumUpdates)
                     .def("getNumExclusions", &NeighborList::getNumExclusions)
#ifdef ENABLE_MPI
                     .def("setCommunicator", &NeighborList::setCommunicator)
#endif
                     ;
                     
    enum_<NeighborList::storageMode>("storageMode")
    .value("half", NeighborList::half)
    .value("full", NeighborList::full)
    ;
    }

