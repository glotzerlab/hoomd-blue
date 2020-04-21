// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

#ifndef __POTENTIAL_REVCROSS_H__
#define __POTENTIAL_REVCROSS_H__

#include <iostream>
#include <stdexcept>
#include <memory>
#include <fstream>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/GPUArray.h"
#include "hoomd/ForceCompute.h"
#include "NeighborList.h"


/*! \file PotentialRevCross.h
    \brief Defines the template class for standard three-body potentials
    \details The heart of the code that computes three-body potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

//! Template class for computing three-body potentials
/*! <b>Overview:</b>
    PotentialRevCross computes standard three-body potentials and forces between all particles in the
    simulation.  It employs the use of a neighbor list to limit the number of computations done to
    only those particles within the cutoff radius of each other.  the computation of the actual
    potential is not performed directly by this class, but by an evaluator class (e.g.
    EvaluatorRevCross) which is passed in as a template parameter so the computations are performed
    as efficiently as possible.

    PotentialRevCross handles most of the internal details common to all standard three-body potentials.
     - A cutoff radius to be specified per particle type-pair
     - Per type-pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - All the details about looping through the particles, computing dr, computing the virial, etc. are handled

    <b>Implementation details</b>

    Unlike the pair potentials, the three-body potentials offer two force directions: ij and ik.
    In addition, some three-body potentials (such as the RevCross potential) compute unique forces on
    each of the three particles involved.  Three-body evaluators must thus return six force magnitudes:
    two for each particle.  These values are returned in the Scalar3 values \a force_divr_ij and
    \a force_divr_ik.  The x components refer to particle i, y to particle j, and z to particle k.
    If your particular three-body potential does not compute one of these forces, then the evaluator
    can simply return 0 for that force.  In addition, the potential energy is stored in the w component
    of force_divr_ij.

    rcutsq, ronsq, and the params are stored per particle type-pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redundant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GPUArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.

    For profiling and logging, PotentialRevCross needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independently.

    \sa export_PotentialRevCross()
*/
template < class evaluator >
class PotentialRevCross : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Construct the potential
        PotentialRevCross(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialRevCross();

        //! Set the pair parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
        //! Set the rcut for a single type pair
        virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
        //! Set ron for a single type pair
        virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
        GPUArray<Scalar> m_rcutsq;                  //!< Cutoff radius squared per type pair
        GPUArray<Scalar> m_ronsq;                   //!< ron squared per type pair
        GPUArray<param_type> m_params;   //!< Pair parameters per type pair
        std::string m_prof_name;                    //!< Cached profiler name
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            // skip the reallocation if the number of types does not change
            // this keeps old potential coefficients when restoring a snapshot
            // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
            if (m_pdata->getNTypes() == m_typpair_idx.getW())
                return;

            m_typpair_idx = Index2D(m_pdata->getNTypes());

            // reallocate parameter arrays
            GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
            m_rcutsq.swap(rcutsq);
            GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
            m_ronsq.swap(ronsq);
            GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
            m_params.swap(params);
            }
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialRevCross< evaluator >::PotentialRevCross(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes())
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialRevCross" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("Triplet ") + evaluator::getName();
    m_log_name = std::string("pair_") + evaluator::getName() + std::string("_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().template connect<PotentialRevCross<evaluator>, &PotentialRevCross<evaluator>::slotNumTypesChange>(this);
    }

template < class evaluator >
PotentialRevCross< evaluator >::~PotentialRevCross()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialRevCross" << std::endl;
    m_pdata->getNumTypesChangeSignal().template disconnect<PotentialRevCross<evaluator>, &PotentialRevCross<evaluator>::slotNumTypesChange>(this);
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialRevCross< evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": Trying to set pair params for a non existent type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialRevCross");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialRevCross< evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << std::endl << "Trying to set rcut for a non existent type! "
                                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialRevCross");
        }

    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param ron XPLOR r_on radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialRevCross< evaluator >::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << std::endl << "Trying to set ron for a non existent type! "
                                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialRevCross");
        }

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

/*! PotentialRevCross provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > PotentialRevCross< evaluator >::getProvidedLogQuantities()
    {
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialRevCross< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": " << quantity << " is not a valid log quantity"
                                        << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialRevCross< evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // The three-body potentials can't handle a half neighbor list, so check now.
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error() << std::endl << "PotentialRevCross cannot handle a half neighborlist"
                                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialRevCross");
        }

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);


    //force and virial arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    // need to start from a zero force, energy
    memset(h_force.data, 0, sizeof(Scalar4)*(m_pdata->getN()+m_pdata->getNGhosts()));
    memset(h_virial.data, 0, sizeof(Scalar)*6*m_virial_pitch);

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current force and potential energy of particle i to 0
        Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
        Scalar pei = 0.0;

        Scalar ivirialxx(0.0);
        Scalar ivirialxy(0.0);
        Scalar ivirialxz(0.0);
        Scalar ivirialyy(0.0);
        Scalar ivirialyz(0.0);
        Scalar ivirialzz(0.0);

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // access the index of neighbor j (MEM TRANSFER: 1 scalar)
            unsigned int jj = h_nlist.data[head_i + j];
            assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

            // access the position and type of particle j
            Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
            unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
            assert(typej < m_pdata->getNTypes());

            // initialize the current force and potential energy of particle j to 0
            Scalar3 fj = make_scalar3(0.0, 0.0, 0.0);
            Scalar pej = 0.0;

            // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 dxij = posi - posj;

            // apply periodic boundary conditions
            dxij = box.minImage(dxij);

            // compute rij_sq (FLOPS: 5)
            Scalar rij_sq = dot(dxij, dxij);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // evaluate the base repulsive and attractive terms
            Scalar invratio = 0.0;
            Scalar invratio2 = 0.0;
            evaluator eval(rij_sq, rcutsq, param);
            bool evaluated = eval.evalRepulsiveAndAttractive(invratio, invratio2);

	    // Even though the i-j interaction is symmetric so in principle I could consider i>j only,
	    // I have to loop over both i-j-k and j-i-k because I search only in neighbors of of the first element
	    // (since nl are type-wise I can not even merge them because i, j and k could be different types)
            if (evaluated)
                {
		//printf("\nEvaluating the pair (i,j)=(%d, %d)  from inside HOOMD CPU",i,jj);
                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                eval.evalForceij(invratio, invratio2, force_divr, potential_eng);

                // add this force to particle i
                fi += force_divr * dxij;
                pei += potential_eng ;

                // add this force to particle j
                fj += Scalar(-1.0) * force_divr * dxij;
                pej += potential_eng ;
                
                //vir contribute for i j direct interaction on particle i and j
		if (compute_virial)
		    {
            	    ivirialxx += force_divr*dxij.x*dxij.x;
            	    ivirialxy += force_divr*dxij.x*dxij.y;
            	    ivirialxz += force_divr*dxij.x*dxij.z;
            	    ivirialyy += force_divr*dxij.y*dxij.y;
            	    ivirialyz += force_divr*dxij.y*dxij.z;
		    ivirialzz += force_divr*dxij.z*dxij.z;
		    }
                
                // evaluate the force from the ik interactions
                for (unsigned int k = j+1; k < size; k++)                    //I want to account only a single time for each triplets 
                    {
                    // access the index of neighbor k
                    unsigned int kk = h_nlist.data[head_i + k];
                    assert(kk < m_pdata->getN());

                    // access the position and type of neighbor k
                    Scalar3 posk = make_scalar3(h_pos.data[kk].x, h_pos.data[kk].y, h_pos.data[kk].z);
                    unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                    assert(typek < m_pdata->getNTypes());

                    // access the type pair parameters for i and k
                    typpair_idx = m_typpair_idx(typei, typek);
                    param_type temp_param = h_params.data[typpair_idx];             // use this to control the species wich have to interact
					
                    // compute dr_ik
                    Scalar3 dxik = posi - posk;
                    // apply periodic boundary conditions
                    dxik = box.minImage(dxik);
                    // compute rik_sq
                    Scalar rik_sq = dot(dxik, dxik);
                    
                    // check if k interacts using a temporary evaluator to analyze i-k parameters                        
                    evaluator temp_eval(rij_sq, rcutsq, temp_param);
                    temp_eval.setRik(rik_sq);
                    bool temp_evaluated = temp_eval.areInteractive();

		    // 3 Body interaction ******
                    if (temp_evaluated)
                        {
			eval.setRik(rik_sq);
                        // compute the total force and energy
                        Scalar3 fk = make_scalar3(0.0, 0.0, 0.0);
                        Scalar force_divr_ij = 0.0;
                        Scalar force_divr_ik = 0.0;
                        bool evaluatedk = eval.evalForceik(invratio,invratio2, force_divr_ij, force_divr_ik);
			// k interacts with the i-j as an additional third body
			if(evaluatedk)
				{
				//printf("\nEvaluating the triplet (i,j,k)=(%d, %d, %d)  from inside HOOMD CPU",i,jj,kk);
				// add the force to particle i
				fi += force_divr_ij * dxij + force_divr_ik * dxik;

				// add the force to particle j (FLOPS: 17)
				fj += force_divr_ij * dxij * Scalar(-1.0);

				// add the force to particle k
				fk += force_divr_ik * dxik * Scalar(-1.0);
				
				if (compute_virial)
			   	{
					//***look at 3 body pressure notes
					//i just need a single term to account for all of the 3 body virial that i decide to store in the i particle's data 
					//and i just defined the diagonal component of pressure tensor, I don't know how the off diagonal terms can be included
					ivirialxx += (force_divr_ij*dxij.x*dxij.x + force_divr_ik*dxik.x*dxik.x);	
					ivirialyy += (force_divr_ij*dxij.y*dxij.y + force_divr_ik*dxik.y*dxik.y);	
					ivirialzz += (force_divr_ij*dxij.z*dxij.z + force_divr_ik*dxik.z*dxik.z);	
					ivirialxy += (force_divr_ij*dxij.x*dxij.y + force_divr_ik*dxik.x*dxik.y);
                   			ivirialxz += (force_divr_ij*dxij.x*dxij.z + force_divr_ik*dxik.x*dxik.z);
                 			ivirialyz += (force_divr_ij*dxij.y*dxij.z + force_divr_ik*dxik.y*dxik.z);
	                	}

                              
				// increment the force for particle k
				unsigned int mem_idx = kk;
				h_force.data[mem_idx].x += fk.x;
				h_force.data[mem_idx].y += fk.y;
				h_force.data[mem_idx].z += fk.z;
			    }
                        }
                    }
                }
                    
            // increment the force and potential energy for particle j
            unsigned int mem_idx = jj;
            h_force.data[mem_idx].x += fj.x;
            h_force.data[mem_idx].y += fj.y;
            h_force.data[mem_idx].z += fj.z;
            h_force.data[mem_idx].w += pej;

	    }
            
        // finally, increment the force and potential energy for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        
        //imcrement vir for i
        if (compute_virial)
            {
            h_virial.data[0*m_virial_pitch+mem_idx] += ivirialxx;
            h_virial.data[1*m_virial_pitch+mem_idx] += ivirialxy;
            h_virial.data[2*m_virial_pitch+mem_idx] += ivirialxz;
            h_virial.data[3*m_virial_pitch+mem_idx] += ivirialyy;
	    h_virial.data[4*m_virial_pitch+mem_idx] += ivirialyz;
            h_virial.data[5*m_virial_pitch+mem_idx] += ivirialzz;
            }

        }

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template < class evaluator >
CommFlags PotentialRevCross< evaluator >::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    // enable reverse communication of forces
    flags[comm_flag::reverse_net_force] = 1;

    // reverse net force requires tags
    flags[comm_flag::tag] = 1;

    return flags;
    }
#endif


//! Export this triplet potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialRevCross class template.
*/
template < class T > void export_PotentialRevCross(pybind11::module& m, const std::string& name)
    {
        pybind11::class_<T, ForceCompute, std::shared_ptr<T> >(m, name.c_str())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
            .def("setParams", &T::setParams)
            .def("setRcut", &T::setRcut)
            .def("setRon", &T::setRon)
        ;
    }


#endif
