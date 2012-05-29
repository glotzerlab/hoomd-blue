/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __POTENTIAL_TRIPLET_H__
#define __POTENTIAL_TRIPLET_H__

#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <fstream>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "GPUArray.h"
#include "ForceCompute.h"
#include "NeighborList.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

/*! \file PotentialTriplet.h
    \brief Defines the template class for standard three-body potentials
    \details The heart of the code that computes three-body potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing three-body potentials
/*! <b>Overview:</b>
    PotentialTriplet computes standard three-body potentials and forces between all particles in the
    simulation.  It employs the use of a neighbor list to limit the number of comutations done to
    only those particles within the cutoff radius of each other.  the computation of the actual
    potential is not performed directly by this class, but by an evaluator class (e.g.
    EvaluatorTripletTersoff) which is passed in as a template parameter so the computations are performed
    as efficiently as possible.

    PotentialTriplet handles most of the internal details common to all standard three-body potentials.
     - A cutoff radius to be specified per particle type-pair
     - Per type-pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - All the details about looping through the particles, computing dr, computing the virial, etc. are handled

    <b>Implementation details</b>

    Unlike the pair potentials, the three-body potentials offer two force directions: ij and ik.
    In addition, some three-body potentials (such as the Tersoff potential) compute unique forces on
    each of the three particles involved.  Three-body evaluators must thus return six force magnitudes:
    two for each particle.  These values are returned in the Scalar4 values \a force_divr_ij and
    \a force_divr_ik.  The x components refer to particle i, y to particle j, and z to particle k.
    If your particular three-body potential does not compute one of these forces, then the evaluator
    can simply return 0 for that force.  In addition, the potential energy is stored in the w component
    of force_divr_ij.  Scalar4 values are used instead of Scalar3's in order to
    maintain compatibility between the CPU and GPU codes.

    rcutsq, ronsq, and the params are stored per particle type-pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redudant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GPUArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.

    For profiling and logging, PotentialTriplet needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independently.

    \sa export_PotentialTriplet()
*/
template < class evaluator >
class PotentialTriplet : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Construct the potential
        PotentialTriplet(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialTriplet() { };

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
    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
        Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
        GPUArray<Scalar> m_rcutsq;                  //!< Cuttoff radius squared per type pair
        GPUArray<Scalar> m_ronsq;                   //!< ron squared per type pair
        GPUArray<param_type> m_params;   //!< Pair parameters per type pair
        std::string m_prof_name;                    //!< Cached profiler name
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialTriplet< evaluator >::PotentialTriplet(boost::shared_ptr<SystemDefinition> sysdef,
                                                boost::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift), m_typpair_idx(m_pdata->getNTypes())
    {
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), exec_conf);
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("Triplet ") + evaluator::getName();
    m_log_name = std::string("pair_") + evaluator::getName() + std::string("_energy") + log_suffix;

    // initialize memory for per thread reduction
    allocateThreadPartial();
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialTriplet< evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set pair params for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTriplet");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cuttoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialTriplet< evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set rcut for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTriplet");
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
void PotentialTriplet< evaluator >::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set ron for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTriplet");
        }

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

/*! PotentialTriplet provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > PotentialTriplet< evaluator >::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialTriplet< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for PotentialTriplet"
                  << std::endl << endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialTriplet< evaluator >::computeForces(unsigned int timestep)
{
//    std::ofstream outfile;
//    outfile.open("errors.log", ios::app);

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // The three-body potentials can't handle a half neighbor list, so check now.
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
    {
        std::cerr << std::endl << "***Error! PotentialTriplet cannot handle a half neighborlist"
                  << std::endl << std::endl;
        throw std::runtime_error("Error computing forces in PotentialTriplet");
    }

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();

    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();

    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);


    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    // precalculate box lengths for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;

#pragma omp parallel
{
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy and virial
    memset(&m_fdata_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar4)*arrays.nparticles);
    memset(&m_virial_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar)*arrays.nparticles);

    // for each particle
#pragma omp for schedule(guided)
    for (int i = 0; i < (int)arrays.nparticles; i++)
    {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar posxi = arrays.x[i];
        Scalar posyi = arrays.y[i];
        Scalar poszi = arrays.z[i];
        unsigned int typei = arrays.type[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = 0.0;
        Scalar qi = 0.0;
        if (evaluator::needsDiameter())
            di = arrays.diameter[i];
        if (evaluator::needsCharge())
            qi = arrays.charge[i];

        // initialize current force, potential energy, and virial of particle i to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar viriali = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
        {
            // access the index of neighbor j (MEM TRANSFER: 1 scalar)
            unsigned int jj = h_nlist.data[nli(i, j)];
            assert(jj < m_pdata->getN());

            // access the type of particle j (MEM TRANSFER: 1 scalar)
            unsigned int typej = arrays.type[jj];
            assert(typej < m_pdata->getNTypes());

            // initialize the current force, potential energy, and virial of particle j to 0
            Scalar fxj = 0.0;
            Scalar fyj = 0.0;
            Scalar fzj = 0.0;
            Scalar pej = 0.0;
            Scalar virialj = 0.0;

            // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar dx_ij = posxi - arrays.x[jj];
            Scalar dy_ij = posyi - arrays.y[jj];
            Scalar dz_ij = poszi - arrays.z[jj];

            // access the diameter and charge of j (if needed)
            Scalar dj = 0.0;
            Scalar qj = 0.0;
            if (evaluator::needsDiameter())
                dj = arrays.diameter[jj];
            if (evaluator::needsCharge())
                qj = arrays.charge[jj];

            // apply periodic boundary conditions
            if (dx_ij >= box.xhi)
                dx_ij -= Lx;
            else if (dx_ij < box.xlo)
                dx_ij += Lx;

            if (dy_ij >= box.yhi)
                dy_ij -= Ly;
            else if (dy_ij < box.ylo)
                dy_ij += Ly;

            if (dz_ij >= box.zhi)
                dz_ij -= Lz;
            else if (dz_ij < box.zlo)
                dz_ij += Lz;

            // compute rij_sq (FLOPS: 5)
            Scalar rij_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // evaluate the base repulsive and attractive terms
            Scalar fR = 0.0;
            Scalar fA = 0.0;
            evaluator eval(rij_sq, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj, Scalar(0.0));
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj, Scalar(0.0));
            bool evaluated = eval.evalRepulsiveAndAttractive(fR, fA);

            if (evaluated)
            {
                // evaluate chi
                Scalar chi = 0.0;
                for (unsigned int k = 0; k < size; k++)
                {
                    // access the index of neighbor k
                    unsigned int kk = h_nlist.data[nli(i,k)];
                    assert(kk < m_pdata->getN());
                    unsigned int typek = arrays.type[kk];
					assert(typek < m_pdata->getNTypes());

					// access the type pair parameters for i and k
					typpair_idx = m_typpair_idx(typei, typek);
					param_type temp_param = h_params.data[typpair_idx];

					evaluator temp_eval(rij_sq, rcutsq, temp_param);
					bool temp_evaluated = temp_eval.areInteractive();

                    if (kk != jj && temp_evaluated)
                    {
                        // compute dr_ik
                        Scalar dx_ik = posxi - arrays.x[kk];
                        Scalar dy_ik = posyi - arrays.y[kk];
                        Scalar dz_ik = poszi - arrays.z[kk];

                        // access the diameter and charge (if needed)
                        Scalar dk = 0.0;
                        Scalar qk = 0.0;
                        if (evaluator::needsDiameter())
                            dk = arrays.diameter[kk];
                        if (evaluator::needsCharge())
                            qk = arrays.charge[kk];

                        // apply periodic boundary conditions
                        if (dx_ik >= box.xhi)
                            dx_ik -= Lx;
                        else if (dx_ik < box.xlo)
                            dx_ik += Lx;

                        if (dy_ik >= box.yhi)
                            dy_ik -= Ly;
                        else if (dy_ik < box.ylo)
                            dy_ik += Ly;

                        if (dz_ik >= box.zhi)
                            dz_ik -= Lz;
                        else if (dz_ik < box.zlo)
                            dz_ik += Lz;

                        // compute rik_sq
                        Scalar rik_sq = dx_ik * dx_ik + dy_ik * dy_ik + dz_ik * dz_ik;

                        // compute the bond angle (if needed)
                        Scalar cos_th = Scalar(0.0);
                        if (evaluator::needsAngle())
                            cos_th = (dx_ij * dx_ik + dy_ij * dy_ik + dz_ij * dz_ik) / sqrt(rij_sq * rik_sq);

                        // evaluate the partial chi term
                        eval.setRik(rik_sq);
                        if (evaluator::needsAngle())
                            eval.setAngle(cos_th);
                        if (evaluator::needsDiameter())
                            eval.setDiameter(di, dj, dk);
                        if (evaluator::needsCharge())
                            eval.setCharge(qi, qj, qk);

                        eval.evalChi(chi);
                    }
                }

                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                Scalar bij = Scalar(0.0);
                eval.evalForceij(fR, fA, chi, bij, force_divr, potential_eng);

                // compute the virial
                Scalar pair_virial = Scalar(1.0 / 6.0) * rij_sq * force_divr;

                // add this force to particle i
                fxi += force_divr * dx_ij;
                fyi += force_divr * dy_ij;
                fzi += force_divr * dz_ij;
                pei += potential_eng * Scalar(0.5);
                viriali += pair_virial;

                // add this force to particle j
                fxj -= force_divr * dx_ij;
                fyj -= force_divr * dy_ij;
                fzj -= force_divr * dz_ij;
                pej += potential_eng * Scalar(0.5);
                virialj += pair_virial;

                // evaluate the force from the ik interactions
                for (unsigned int k = 0; k < size; k++)
                {
                    // access the index of neighbor k
                    unsigned int kk = h_nlist.data[nli(i, k)];
                    assert(kk < m_pdata->getN());
                    unsigned int typek = arrays.type[kk];
					assert(typek < m_pdata->getNTypes());

					// access the type pair parameters for i and k
					typpair_idx = m_typpair_idx(typei, typek);
					param_type temp_param = h_params.data[typpair_idx];

					evaluator temp_eval(rij_sq, rcutsq, temp_param);
					bool temp_evaluated = temp_eval.areInteractive();

                    if (kk != jj && temp_evaluated)
                    {
                        // create variables for the force and virial on k
                        Scalar fxk = Scalar(0.0);
                        Scalar fyk = Scalar(0.0);
                        Scalar fzk = Scalar(0.0);
                        Scalar virialk = Scalar(0.0);

                        // compute dr_ik
                        Scalar dx_ik = posxi - arrays.x[kk];
                        Scalar dy_ik = posyi - arrays.y[kk];
                        Scalar dz_ik = poszi - arrays.z[kk];

                        // access the diameter and charge (if needed)
                        Scalar dk = Scalar(0.0);
                        Scalar qk = Scalar(0.0);
                        if (evaluator::needsDiameter())
                            dk = arrays.diameter[kk];
                        if (evaluator::needsCharge())
                            qk = arrays.charge[kk];

                        // apply periodic boundary conditions
                        if (dx_ik >= box.xhi)
                            dx_ik -= Lx;
                        else if (dx_ik < box.xlo)
                            dx_ik += Lx;

                        if (dy_ik >= box.yhi)
                            dy_ik -= Ly;
                        else if (dy_ik < box.ylo)
                            dy_ik += Ly;

                        if (dz_ik >= box.zhi)
                            dz_ik -= Lz;
                        else if (dz_ik < box.zlo)
                            dz_ik += Lz;

                        // compute rik_sq
                        Scalar rik_sq = dx_ik * dx_ik + dy_ik * dy_ik + dz_ik * dz_ik;

                        // compute the bond angle (if needed)
                        Scalar cos_th = Scalar(0.0);
                        if (evaluator::needsAngle())
                            cos_th = (dx_ij * dx_ik + dy_ij * dy_ik + dz_ij * dz_ik) / sqrt(rij_sq * rik_sq);

                        // set up the evaluator
                        eval.setRik(rik_sq);
                        if (evaluator::needsAngle())
                            eval.setAngle(cos_th);
                        if (evaluator::needsDiameter())
                            eval.setDiameter(di, dj, dk);
                        if (evaluator::needsCharge())
                            eval.setCharge(qi, qj, qk);

                        // compute the total force and energy
                        Scalar4 force_divr_ij = make_scalar4(0.0, 0.0, 0.0, 0.0);
                        Scalar4 force_divr_ik = make_scalar4(0.0, 0.0, 0.0, 0.0);
                        eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                        // compute the virial coefficients
                        Scalar virial_coeff_ij = Scalar(1.0/6.0) * rij_sq;
                        Scalar virial_coeff_ik = Scalar(1.0/6.0) * rik_sq;

                        // add the force and virial to particle i
                        // (FLOPS: 17)
                        fxi += force_divr_ij.x * dx_ij + force_divr_ik.x * dx_ik;
                        fyi += force_divr_ij.x * dy_ij + force_divr_ik.x * dy_ik;
                        fzi += force_divr_ij.x * dz_ij + force_divr_ik.x * dz_ik;
                        viriali += virial_coeff_ij * force_divr_ij.x + virial_coeff_ik * force_divr_ik.x;

                        // add the force and virial to particle j (FLOPS: 17)
                        fxj += force_divr_ij.y * dx_ij + force_divr_ik.y * dx_ik;
                        fyj += force_divr_ij.y * dy_ij + force_divr_ik.y * dy_ik;
                        fzj += force_divr_ij.y * dz_ij + force_divr_ik.y * dz_ik;
                        virialj += virial_coeff_ij * force_divr_ij.y + virial_coeff_ik * force_divr_ik.y;

                        // add the force and virial to particle k
                        fxk += force_divr_ij.z * dx_ij + force_divr_ik.z * dx_ik;
                        fyk += force_divr_ij.z * dy_ij + force_divr_ik.z * dy_ik;
                        fzk += force_divr_ij.z * dz_ij + force_divr_ik.z * dz_ik;
                        virialk += virial_coeff_ij * force_divr_ij.z + virial_coeff_ik * force_divr_ik.z;

                        // increment the force and virial for particle k
                        unsigned int mem_idx = m_index_thread_partial(kk, tid);
                        m_fdata_partial[mem_idx].x += fxk;
                        m_fdata_partial[mem_idx].y += fyk;
                        m_fdata_partial[mem_idx].z += fzk;
                        m_virial_partial[mem_idx] += virialk;
                    }
                }
            }
            // increment the force, potential energy, and virial for particle j
            unsigned int mem_idx = m_index_thread_partial(jj, tid);
            m_fdata_partial[mem_idx].x += fxj;
            m_fdata_partial[mem_idx].y += fyj;
            m_fdata_partial[mem_idx].z += fzj;
            m_fdata_partial[mem_idx].w += pej;
            m_virial_partial[mem_idx] += virialj;
        }
        // finally, increment the force, potential energy, and virial for particle i
        unsigned int mem_idx = m_index_thread_partial(i,tid);
        m_fdata_partial[mem_idx].x += fxi;
        m_fdata_partial[mem_idx].y += fyi;
        m_fdata_partial[mem_idx].z += fzi;
        m_fdata_partial[mem_idx].w += pei;
        m_virial_partial[mem_idx] += viriali;
    }
#pragma omp barrier

    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int)arrays.nparticles; i++)
    {
        // assign result from thread 0
        h_force.data[i].x  = m_fdata_partial[i].x;
        h_force.data[i].y = m_fdata_partial[i].y;
        h_force.data[i].z = m_fdata_partial[i].z;
        h_force.data[i].w = m_fdata_partial[i].w;
        h_virial.data[i] = m_virial_partial[i];

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
        {
            unsigned int mem_idx = m_index_thread_partial(i,thread);
            h_force.data[i].x += m_fdata_partial[mem_idx].x;
            h_force.data[i].y += m_fdata_partial[mem_idx].y;
            h_force.data[i].z += m_fdata_partial[mem_idx].z;
            h_force.data[i].w += m_fdata_partial[mem_idx].w;
            h_virial.data[i] += m_virial_partial[mem_idx];
        }
        #endif
    }
} // end omp parallel
//    outfile.close();

    m_pdata->release();

    if (m_prof) m_prof->pop();
}

//! Export this triplet potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialTriplet class template.
*/
template < class T > void export_PotentialTriplet(const std::string& name)
    {
    boost::python::scope in_pair =
        boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
                  .def("setParams", &T::setParams)
                  .def("setRcut", &T::setRcut)
                  .def("setRon", &T::setRon)
                  .def("setShiftMode", &T::setShiftMode)
                  ;

    boost::python::enum_<typename T::energyShiftMode>("energyShiftMode")
        .value("no_shift", T::no_shift)
        .value("shift", T::shift)
        .value("xplor", T::xplor)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#endif

