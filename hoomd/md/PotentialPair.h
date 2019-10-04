// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __POTENTIAL_PAIR_H__
#define __POTENTIAL_PAIR_H__

#include <iostream>
#include <stdexcept>
#include <memory>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "hoomd/extern/pybind/include/pybind11/numpy.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/ForceCompute.h"
#include "NeighborList.h"
#include "hoomd/GSDShapeSpecWriter.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif


/*! \file PotentialPair.h
    \brief Defines the template class for standard pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing pair potentials
/*! <b>Overview:</b>
    PotentialPair computes standard pair potentials (and forces) between all particle pairs in the simulation. It
    employs the use of a neighbor list to limit the number of computations done to only those particles with the
    cutoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an evaluator class (e.g. EvaluatorPairLJ) which is passed in as a template parameter so the computations
    are performed as efficiently as possible.

    PotentialPair handles most of the gory internal details common to all standard pair potentials.
     - A cutoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cutoff
     - XPLOR switching can be enabled
     - Per type pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - And all the details about looping through the particles, computing dr, computing the virial, etc. are handled

    A note on the design of XPLOR switching:
    We need to be able to handle smooth XPLOR switching in systems of mixed LJ/WCA particles. There are three modes to
    enable all of the various use-cases:
     - Mode 1: No shifting. All pair potentials are computed as is and not shifted to 0 at the cutoff.
     - Mode 2: Shift everything. All pair potentials (no matter what type pair) are shifted so they are 0 at the cutoff
     - Mode 3: XPLOR switching enabled. A r_on value is specified per type pair. When r_on is less than r_cut, normal
       XPLOR switching will be applied to the unshifted potential. When r_on is greater than r_cut, the energy will
       be shifted. In this manner, a valid r_on value can be given for the LJ interactions and r_on > r_cut can be set
       for WCA (which will then be shifted).

    XPLOR switching gets significantly more complicated for all pair potentials when shifted potentials are used. Thus,
    the combination of XPLOR switching + shifted potentials will not be supported to avoid slowing down the calculation
    for everyone.

    <b>Implementation details</b>

    rcutsq, ronsq, and the params are stored per particle type pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redundant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GlobalArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.

    For profiling and logging, PotentialPair needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independently.

    \sa export_PotentialPair()
*/
template < class evaluator >
class PotentialPair : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Construct the pair potential
        PotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist,
                      const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPair();

        //! Set the pair parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
        //! Set the rcut for a single type pair
        virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
        //! Set ron for a single type pair
        virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);

        //! Method that is called whenever the GSD file is written if connected to a GSD file.
        int slotWriteGSDShapeSpec(gsd_handle&) const;

        //! Method that is called to connect to the gsd write state signal
        void connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        enum energyShiftMode
            {
            no_shift = 0,
            shift,
            xplor
            };

        //! Set the mode to use for shifting the energy
        void setShiftMode(energyShiftMode mode)
            {
            m_shift_mode = mode;
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

        //! Calculates the energy between two lists of particles.
        template< class InputIterator >
        void computeEnergyBetweenSets(  InputIterator first1, InputIterator last1,
                                            InputIterator first2, InputIterator last2,
                                            Scalar& energy );
        //! Calculates the energy between two lists of particles.
        Scalar computeEnergyBetweenSetsPythonList(  pybind11::array_t<int, pybind11::array::c_style> tags1,
                                                    pybind11::array_t<int, pybind11::array::c_style> tags2);

        std::vector<std::string> getTypeShapeMapping(const GlobalArray<param_type> &params) const
            {
            ArrayHandle<param_type> h_params(params, access_location::host, access_mode::read);
            std::vector<std::string> type_shape_mapping(m_pdata->getNTypes());
            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
                {
                evaluator eval(Scalar(0.0),Scalar(0.0), h_params.data[m_typpair_idx(i,i)]);
                type_shape_mapping[i] = eval.getShapeSpec();
                }
            return type_shape_mapping;
            }

    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
        Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
        GlobalArray<Scalar> m_rcutsq;                  //!< Cutoff radius squared per type pair
        GlobalArray<Scalar> m_ronsq;                   //!< ron squared per type pair
        GlobalArray<param_type> m_params;              //!< Pair parameters per type pair
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

            // if the number of types is different, built a new indexer and reallocate memory
            m_typpair_idx = Index2D(m_pdata->getNTypes());

            // reallocate parameter arrays
            GlobalArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
            m_rcutsq.swap(rcutsq);
            GlobalArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
            m_ronsq.swap(ronsq);
            GlobalArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
            m_params.swap(params);

            #ifdef ENABLE_CUDA
            if (m_pdata->getExecConf()->isCUDAEnabled() && m_pdata->getExecConf()->allConcurrentManagedAccess())
                {
                cudaMemAdvise(m_rcutsq.get(), m_rcutsq.getNumElements()*sizeof(Scalar), cudaMemAdviseSetReadMostly, 0);
                cudaMemAdvise(m_ronsq.get(), m_ronsq.getNumElements()*sizeof(Scalar), cudaMemAdviseSetReadMostly, 0);
                cudaMemAdvise(m_params.get(), m_params.getNumElements()*sizeof(param_type), cudaMemAdviseSetReadMostly, 0);

                // prefetch
                auto& gpu_map = m_exec_conf->getGPUIds();

                for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                    {
                    // prefetch data on all GPUs
                    cudaMemPrefetchAsync(m_rcutsq.get(), sizeof(Scalar)*m_rcutsq.getNumElements(), gpu_map[idev]);
                    cudaMemPrefetchAsync(m_ronsq.get(), sizeof(Scalar)*m_ronsq.getNumElements(),gpu_map[idev]);
                    cudaMemPrefetchAsync(m_params.get(), sizeof(param_type)*m_params.getNumElements(), gpu_map[idev]);
                    }
                CHECK_CUDA_ERROR();
                }
            #endif
            }
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialPair< evaluator >::PotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift), m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialPair<" << evaluator::getName() << ">" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GlobalArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GlobalArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_ronsq.swap(ronsq);
    GlobalArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
    m_params.swap(params);

    #ifdef ENABLE_CUDA
    if (m_pdata->getExecConf()->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_rcutsq.get(), m_rcutsq.getNumElements()*sizeof(Scalar), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_ronsq.get(), m_ronsq.getNumElements()*sizeof(Scalar), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_params.get(), m_params.getNumElements()*sizeof(param_type), cudaMemAdviseSetReadMostly, 0);

        // prefetch
        auto& gpu_map = m_exec_conf->getGPUIds();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // prefetch data on all GPUs
            cudaMemPrefetchAsync(m_rcutsq.get(), sizeof(Scalar)*m_rcutsq.getNumElements(), gpu_map[idev]);
            cudaMemPrefetchAsync(m_ronsq.get(), sizeof(Scalar)*m_ronsq.getNumElements(),gpu_map[idev]);
            cudaMemPrefetchAsync(m_params.get(), sizeof(param_type)*m_params.getNumElements(), gpu_map[idev]);
            }
        }
    #endif

    // initialize name
    m_prof_name = std::string("Pair ") + evaluator::getName();
    m_log_name = std::string("pair_") + evaluator::getName() + std::string("_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().template connect<PotentialPair<evaluator>, &PotentialPair<evaluator>::slotNumTypesChange>(this);
    }

template< class evaluator >
PotentialPair< evaluator >::~PotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialPair<" << evaluator::getName() << ">" << std::endl;

    m_pdata->getNumTypesChangeSignal().template disconnect<PotentialPair<evaluator>, &PotentialPair<evaluator>::slotNumTypesChange>(this);
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialPair< evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": Trying to set pair params for a non existent type! "
                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
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
void PotentialPair< evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": Trying to set rcut for a non existent type! "
                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
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
void PotentialPair< evaluator >::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": Trying to set ron for a non existent type! "
                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
        }

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

template <class evaluator>
void PotentialPair<evaluator>::connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&PotentialPair<evaluator>::slotWriteGSDShapeSpec, this, std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <class evaluator>
int PotentialPair<evaluator>::slotWriteGSDShapeSpec(gsd_handle& handle) const
    {
    GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "PotentialPair writing to GSD File to name: " << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping(m_params));
    return retval;
    }

/*! PotentialPair provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > PotentialPair< evaluator >::getProvidedLogQuantities()
    {
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialPair< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
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

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialPair< evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
//     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);


    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(m_virial,access_location::host, access_mode::overwrite);


    const BoxDim& box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if (evaluator::needsDiameter())
            di = h_diameter.data[i];
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if (evaluator::needsDiameter())
                dj = h_diameter.data[j];
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (m_shift_mode == xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;
            else if (m_shift_mode == xplor)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (evaluated)
                {
                // modify the potential for xplor shifting
                if (m_shift_mode == xplor)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv =
                            Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq *
                                   (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq) * xplor_denom_inv;
                        Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a +
                        // But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }

                Scalar force_div2r = force_divr * Scalar(0.5);
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx*force_divr;
                pei += pair_eng * Scalar(0.5);
                if (compute_virial)
                    {
                    virialxxi += force_div2r*dx.x*dx.x;
                    virialxyi += force_div2r*dx.x*dx.y;
                    virialxzi += force_div2r*dx.x*dx.z;
                    virialyyi += force_div2r*dx.y*dx.y;
                    virialyzi += force_div2r*dx.y*dx.z;
                    virialzzi += force_div2r*dx.z*dx.z;
                    }

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                // only add force to local particles
                if (third_law && j < m_pdata->getN())
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x*force_divr;
                    h_force.data[mem_idx].y -= dx.y*force_divr;
                    h_force.data[mem_idx].z -= dx.z*force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
                        h_virial.data[0*m_virial_pitch+mem_idx] += force_div2r*dx.x*dx.x;
                        h_virial.data[1*m_virial_pitch+mem_idx] += force_div2r*dx.x*dx.y;
                        h_virial.data[2*m_virial_pitch+mem_idx] += force_div2r*dx.x*dx.z;
                        h_virial.data[3*m_virial_pitch+mem_idx] += force_div2r*dx.y*dx.y;
                        h_virial.data[4*m_virial_pitch+mem_idx] += force_div2r*dx.y*dx.z;
                        h_virial.data[5*m_virial_pitch+mem_idx] += force_div2r*dx.z*dx.z;
                        }
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        if (compute_virial)
            {
            h_virial.data[0*m_virial_pitch+mem_idx] += virialxxi;
            h_virial.data[1*m_virial_pitch+mem_idx] += virialxyi;
            h_virial.data[2*m_virial_pitch+mem_idx] += virialxzi;
            h_virial.data[3*m_virial_pitch+mem_idx] += virialyyi;
            h_virial.data[4*m_virial_pitch+mem_idx] += virialyzi;
            h_virial.data[5*m_virial_pitch+mem_idx] += virialzzi;
            }
        }

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template < class evaluator >
CommFlags PotentialPair< evaluator >::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    if (evaluator::needsDiameter())
        flags[comm_flag::diameter] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif


//! function to compute the energy between two lists of particles.
//! strictly speaking tags1 and tags2 should be disjoint for the result to make any sense.
//! \param energy is the sum of the energies between all particles in tags1 and tags2, U = \sum_{i \in tags1, j \in tags2} u_{ij}.
template< class evaluator >
template< class InputIterator >
inline void PotentialPair< evaluator >::computeEnergyBetweenSets(   InputIterator first1, InputIterator last1,
                                                                    InputIterator first2, InputIterator last2,
                                                                    Scalar& energy )
    {
    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    if( first1 == last1 || first2 == last2 )
        return;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // temporarily add tag comm flag
        CommFlags old_flags = m_comm->getFlags();
        CommFlags new_flags = old_flags;
        new_flags[comm_flag::tag] = 1;
        m_comm->setFlags(new_flags);

        // force communication
        m_comm->migrateParticles();
        m_comm->exchangeGhosts();

        // reset the old flags
        m_comm->setFlags(old_flags);
        }
    #endif

    energy = Scalar(0.0);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_rtags(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    // for each particle in tags1
    while (first1 != last1)
        {
        unsigned int i = h_rtags.data[*first1]; first1++;
        if (i >= m_pdata->getN()) // not owned by this processor.
            continue;
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if (evaluator::needsDiameter())
            di = h_diameter.data[i];
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // loop over all particles in tags2
        for (InputIterator iter = first2; iter != last2; ++iter)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_rtags.data[*iter];
            if (j >= m_pdata->getN() + m_pdata->getNGhosts()) // not on this processor at all
                continue;
            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if (evaluator::needsDiameter())
                dj = h_diameter.data[j];
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (m_shift_mode == xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;
            else if (m_shift_mode == xplor)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (evaluated)
                {
                // modify the potential for xplor shifting
                if (m_shift_mode == xplor)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv =
                            Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq *
                                   (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq) * xplor_denom_inv;
                        Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a +
                        // But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }
                energy += pair_eng;
                }
            }
        }
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_prof) m_prof->pop();
    }

//! Calculates the energy between two lists of particles.
template < class evaluator >
Scalar PotentialPair< evaluator >::computeEnergyBetweenSetsPythonList(  pybind11::array_t<int, pybind11::array::c_style> tags1,
                                                                        pybind11::array_t<int, pybind11::array::c_style> tags2 )
    {
    Scalar eng = 0.0;
    if (tags1.ndim() != 1)
        throw std::domain_error("error: ndim != 2");
    unsigned int* itags1 = (unsigned int*)tags1.mutable_data();

    if (tags2.ndim() != 1)
        throw std::domain_error("error: ndim != 2");
    unsigned int* itags2 = (unsigned int*)tags2.mutable_data();
    computeEnergyBetweenSets(   itags1, itags1 + tags1.size(),
                                itags2, itags2 + tags2.size(),
                                eng);
    return eng;
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template < class T > void export_PotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> > potentialpair(m, name.c_str(), pybind11::base<ForceCompute>());
    potentialpair.def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setParams", &T::setParams)
        .def("setRcut", &T::setRcut)
        .def("setRon", &T::setRon)
        .def("setShiftMode", &T::setShiftMode)
        .def("computeEnergyBetweenSets", &T::computeEnergyBetweenSetsPythonList)
        .def("slotWriteGSDShapeSpec", &T::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec", &T::connectGSDShapeSpec)
    ;

    pybind11::enum_<typename T::energyShiftMode>(potentialpair,"energyShiftMode")
        .value("no_shift", T::energyShiftMode::no_shift)
        .value("shift", T::energyShiftMode::shift)
        .value("xplor", T::energyShiftMode::xplor)
        .export_values()
    ;
    }


#endif // __POTENTIAL_PAIR_H__
