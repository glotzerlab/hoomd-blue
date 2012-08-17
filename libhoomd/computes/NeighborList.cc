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
using namespace boost::python;

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
      m_storage_mode(half), m_updates(0), m_forced_updates(0), m_dangerous_updates(0), m_force_update(true),
      m_dist_check(true)
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
    m_every = 0;
    m_Nmax = 0;
    m_exclusions_set = false;
    
    // allocate m_n_neigh and m_last_pos
    GPUArray<unsigned int> n_neigh(m_pdata->getN(), exec_conf);
    m_n_neigh.swap(n_neigh);
    GPUArray<Scalar4> last_pos(m_pdata->getN(), exec_conf);
    m_last_pos.swap(last_pos);
   
    // initialize box length at last update
    m_last_L = m_pdata->getGlobalBox().getL();

    // allocate conditions array
    GPUArray<unsigned int> conditions(1, exec_conf);
    m_conditions.swap(conditions);
    
    // allocate initial memory allowing 4 exclusions per particle (will grow to match specified exclusions)
    GPUArray<unsigned int> n_ex_tag(m_pdata->getN(), exec_conf);
    m_n_ex_tag.swap(n_ex_tag);
    GPUArray<unsigned int> n_ex_idx(m_pdata->getN(), exec_conf);
    m_n_ex_idx.swap(n_ex_idx);
    GPUArray<unsigned int> ex_list_tag(m_pdata->getN(), 1, exec_conf);
    m_ex_list_tag.swap(ex_list_tag);
    GPUArray<unsigned int> ex_list_idx(m_pdata->getN(), 1, exec_conf);
    m_ex_list_idx.swap(ex_list_idx);
    m_ex_list_indexer = Index2D(m_ex_list_tag.getPitch(), 1);
    
    // allocate nlist array
    allocateNlist();
    
    m_sort_connection = m_pdata->connectParticleSort(bind(&NeighborList::forceUpdate, this));

    // allocate m_update_periods tracking info
    m_update_periods.resize(100);
    for (unsigned int i = 0; i < m_update_periods.size(); i++)
        m_update_periods[i] = 0;
    }

NeighborList::~NeighborList()
    {
    m_exec_conf->msg->notice(5) << "Destroying Neighborlist" << endl;

    m_sort_connection.disconnect();
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
    assert(tag1 < m_pdata->getN());
    assert(tag2 < m_pdata->getN());
    
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
        h_ex_list_tag.data[m_ex_list_indexer(tag1, pos1)] = tag2;
        h_n_ex_tag.data[tag1]++;
        
        // add tag1 to tag2's exclusion list
        unsigned int pos2 = h_n_ex_tag.data[tag2];
        assert(pos2 < m_ex_list_indexer.getH());
        h_ex_list_tag.data[m_ex_list_indexer(tag2, pos2)] = tag1;
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
    
    // for each bond
    for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
        {
        // add an exclusion
        Bond bond = bond_data->getBond(i);
        addExclusion(bond.a, bond.b);
        }
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
    assert(tag1 < m_pdata->getN());
    assert(tag2 < m_pdata->getN());
        
    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag, access_location::host, access_mode::read);
    
    unsigned int n_ex = h_n_ex_tag.data[tag1];
    for (unsigned int i = 0; i < n_ex; i++)
        {
        if (h_ex_list_tag.data[m_ex_list_indexer(tag1, i)] == tag2)
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
    Scalar3 L = box.getL();

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
    Scalar maxsq = delta_max*delta_max;

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
        }
        
    // don't worry about computing flops here, this is fast
    if (m_prof) m_prof->pop();
    
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
    if (timestep < (m_last_updated_tstep + m_every) && !m_force_update)
        return false;
        
    // check if this is a dangerous time
    // we are dangerous if m_every is greater than 1 and this is the first check after the
    // last build
    bool dangerous = false;
    if (m_dist_check && (m_every > 1 && timestep == (m_last_updated_tstep + m_every)))
        dangerous = true;
        
    // temporary storage for return result
    bool result = false;
    
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
    Scalar3 L = box.getL();
    
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
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
    
    // start by clearing the entire list
    memset(h_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getN());
    
    // now we can loop over all particles in n^2 fashion and build the list
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar di = h_diameter.data[i];
        unsigned int bodyi = h_body.data[i];
        
        // for each other particle with i < j
        for (unsigned int j = i + 1; j < m_pdata->getN(); j++)
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
                            h_conditions.data[0] = max(h_conditions.data[0], h_n_neigh.data[i]+1);
                        
                        h_n_neigh.data[i]++;
                        
                        unsigned int posj = h_n_neigh.data[j];
                        if (posj < m_Nmax)
                            h_nlist.data[m_nlist_indexer(j, posj)] = i;
                        else
                            h_conditions.data[0] = max(h_conditions.data[0], h_n_neigh.data[j]+1);

                        h_n_neigh.data[j]++;
                        }
                    }
                else
                    {
                    unsigned int pos = h_n_neigh.data[i];
                    
                    if (pos < m_Nmax)
                        h_nlist.data[m_nlist_indexer(i, pos)] = j;
                    else
                        h_conditions.data[0] = max(h_conditions.data[0], h_n_neigh.data[i]+1);
                    
                    h_n_neigh.data[i]++;
                    }
                }
            }
        }
        
    if (m_prof) m_prof->pop();
    }

/*! Translates the exclusions set in \c m_n_ex_tag and \c m_ex_list_tag to indices in \c m_n_ex_idx and \c m_ex_list_idx
*/
void NeighborList::updateExListIdx()
    {
    if (m_prof)
        m_prof->push("update-ex");
    // access data
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_n_ex_tag(m_n_ex_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_ex_list_tag(m_ex_list_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_ex_idx(m_n_ex_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_ex_list_idx(m_ex_list_idx, access_location::host, access_mode::overwrite);
    
    // translate the number and exclusions from one array to the other
    for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
        {
        // get the index for this tag
        unsigned int idx = h_rtag.data[tag];
        
        // copy the number of exclusions over
        unsigned int n = h_n_ex_tag.data[tag];
        h_n_ex_idx.data[idx] = n;
        
        // copy the exclusion list
        for (unsigned int offset = 0; offset < n; offset++)
            {
            unsigned int ex_tag = h_ex_list_tag.data[m_ex_list_indexer(tag, offset)];
            unsigned int ex_idx = h_rtag.data[ex_tag];
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
    // the neighbor list might be large, filling the device memory - maybe we should deallocate the old nlist first
    // freing a gpu array isn't easy ... oh well, wait until a user complains
    
    // round up to the nearest multiple of 8
    m_Nmax = m_Nmax + 8 - (m_Nmax & 7);

    m_exec_conf->msg->notice(6) << "nlist: Allocating " << m_pdata->getN() << " x " << m_Nmax+1 << endl;

    // allocate the memory
    GPUArray<unsigned int> nlist(m_pdata->getN(), m_Nmax+1, exec_conf);
    m_nlist.swap(nlist);
    
    // update the indexer
    m_nlist_indexer = Index2D(m_nlist.getPitch(), m_Nmax);
    }

bool NeighborList::checkConditions()
    {
    bool result = false;

    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::read);

    // up m_Nmax to the overflow value, reallocate memory and set the overflow condition
    if (h_conditions.data[0] > m_Nmax)
        {
        m_Nmax = h_conditions.data[0];
        result = true;
        }

    return result;
    }

void NeighborList::resetConditions()
    {
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::overwrite);
    h_conditions.data[0] = 0;
    }

void NeighborList::growExclusionList()
    {
    unsigned int new_height = m_ex_list_indexer.getH() + 1;
    
    // allocate the two new arrays
    GPUArray<unsigned int> ex_list_tag(m_pdata->getN(), new_height, exec_conf);
    GPUArray<unsigned int> ex_list_idx(m_pdata->getN(), new_height, exec_conf);

    // copy the data across to the new arrays
        {
        ArrayHandle<unsigned int> h_ex_list_tag_old(m_ex_list_tag, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_ex_list_tag_new(ex_list_tag, access_location::host, access_mode::overwrite);
        
        memcpy(h_ex_list_tag_new.data, h_ex_list_tag_old.data, sizeof(unsigned int)*m_ex_list_indexer.getNumElements());
        }

    // swap the new arrays for the old
    m_ex_list_tag.swap(ex_list_tag);
    m_ex_list_idx.swap(ex_list_idx);

    // update the indexer
    m_ex_list_indexer = Index2D(m_ex_list_tag.getPitch(), new_height);
    
    // we didn't copy data for the new idx list, force an update so it will be correct
    forceUpdate();
    }

//! helper function for accessing an elemeng of the neighb rlist: python __getitem__
/*! \param list List to extract an item from
    \param i item to extract
*/
unsigned int getNlistItem(std::vector<unsigned int>* list, unsigned int i)
    {
    return (*list)[i];
    }

void export_NeighborList()
    {
    class_< std::vector<unsigned int> >("std_vector_uint")
    .def("__len__", &std::vector<unsigned int>::size)
    .def("__getitem__", &getNlistItem)
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
                     .def("setFilterDiameter", &NeighborList::setFilterDiameter)
                     .def("setMaximumDiameter", &NeighborList::setMaximumDiameter)
                     .def("forceUpdate", &NeighborList::forceUpdate)
                     .def("estimateNNeigh", &NeighborList::estimateNNeigh)
                     .def("getSmallestRebuild", &NeighborList::getSmallestRebuild)
                     ;
                     
    enum_<NeighborList::storageMode>("storageMode")
    .value("half", NeighborList::half)
    .value("full", NeighborList::full)
    ;
    }

