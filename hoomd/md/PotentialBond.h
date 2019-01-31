// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <memory>
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"

#include <vector>

/*! \file PotentialBond.h
    \brief Declares PotentialBond
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __POTENTIALBOND_H__
#define __POTENTIALBOND_H__

/*! Bond potential with evaluator support

    \ingroup computes
*/
template < class evaluator >
class PotentialBond : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Constructs the compute
        PotentialBond(std::shared_ptr<SystemDefinition> sysdef,
                      const std::string& log_suffix="");

        //! Destructor
        virtual ~PotentialBond();

        //! Set the parameters
        virtual void setParams(unsigned int type, const param_type &param);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

    protected:
        GPUArray<param_type> m_params;              //!< Bond parameters per type
        std::shared_ptr<BondData> m_bond_data;    //!< Bond data to use in computing bonds
        std::string m_log_name;                     //!< Cached log name
        std::string m_prof_name;                    //!< Cached profiler name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param log_suffix Name given to this instance of the force
*/
template< class evaluator >
PotentialBond< evaluator >::PotentialBond(std::shared_ptr<SystemDefinition> sysdef,
                      const std::string& log_suffix)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialBond<" << evaluator::getName() << ">" << std::endl;
    assert(m_pdata);

    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    m_log_name = std::string("bond_") + evaluator::getName() + std::string("_energy") + log_suffix;
    m_prof_name = std::string("Bond ") + evaluator::getName();

    // allocate the parameters
    GPUArray<param_type> params(m_bond_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

template< class evaluator >
PotentialBond< evaluator >::~PotentialBond()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialBond<" << evaluator::getName() << ">" << std::endl;
    }

/*! \param type Type of the bond to set parameters for
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator >
void PotentialBond< evaluator >::setParams(unsigned int type, const param_type& param)
    {
    // make sure the type is valid
    if (type >= m_bond_data->getNTypes())
        {
        this->m_exec_conf->msg->error() << "Invalid bond type specified" << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialBond");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = param;
    }

/*! PotentialBond provides
    - \c bond_"name"_energy
*/
template< class evaluator >
std::vector< std::string > PotentialBond< evaluator >::getProvidedLogQuantities()
    {
    std::vector<std::string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialBond< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": " << quantity << " is not a valid log quantity" << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
template< class evaluator >
void PotentialBond< evaluator >::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_prof_name);

    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::readwrite);

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);


    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_diameter.data);
    assert(h_charge.data);

    // Zero data for force calculation
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain length)
    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    Scalar bond_virial[6];
    for (unsigned int i = 0; i< 6; i++)
        bond_virial[i]=Scalar(0.0);

    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);
    ArrayHandle<typeval_t> h_typeval(m_bond_data->getTypeValArray(), access_location::host, access_mode::read);

    unsigned int max_local = m_pdata->getN() + m_pdata->getNGhosts();

    // for each of the bonds
    const unsigned int size = (unsigned int)m_bond_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename BondData::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag()+1);
        assert(bond.tag[1] < m_pdata->getMaximumTag()+1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];

        // throw an error if this bond is incomplete
        if (idx_a >= max_local || idx_b >= max_local)
            {
            this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": bond " <<
                bond.tag[0] << " " << bond.tag[1] << " incomplete." << std::endl << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }

        // calculate d\vec{r}
        // (MEM TRANSFER: 6 Scalars / FLOPS: 3)
        Scalar3 posa = make_scalar3(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        Scalar3 posb = make_scalar3(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);

        Scalar3 dx = posb - posa;

        // access diameter (if needed)
        Scalar diameter_a = Scalar(0.0);
        Scalar diameter_b = Scalar(0.0);
        if (evaluator::needsDiameter())
            {
            diameter_a = h_diameter.data[idx_a];
            diameter_b = h_diameter.data[idx_b];
            }

        // access charge (if needed)
        Scalar charge_a = Scalar(0.0);
        Scalar charge_b = Scalar(0.0);
        if (evaluator::needsCharge())
            {
            charge_a = h_charge.data[idx_a];
            charge_b = h_charge.data[idx_b];
            }

        // if the vector crosses the box, pull it back
        dx = box.minImage(dx);

        // calculate r_ab squared
        Scalar rsq = dot(dx,dx);

        // get parameters for this bond type
        param_type param = h_params.data[h_typeval.data[i].type];

        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        evaluator eval(rsq, param);
        if (evaluator::needsDiameter())
            eval.setDiameter(diameter_a,diameter_b);
        if (evaluator::needsCharge())
            eval.setCharge(charge_a,charge_b);

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        // Bond energy must be halved
        bond_eng *= Scalar(0.5);

        if (evaluated)
            {
            // calculate virial
            if (compute_virial)
                {
                Scalar force_div2r = Scalar(1.0/2.0)*force_divr;
                bond_virial[0] = dx.x * dx.x * force_div2r; // xx
                bond_virial[1] = dx.x * dx.y * force_div2r; // xy
                bond_virial[2] = dx.x * dx.z * force_div2r; // xz
                bond_virial[3] = dx.y * dx.y * force_div2r; // yy
                bond_virial[4] = dx.y * dx.z * force_div2r; // yz
                bond_virial[5] = dx.z * dx.z * force_div2r; // zz
                }

            // add the force to the particles (only for non-ghost particles)
            if (idx_b < m_pdata->getN())
                {
                h_force.data[idx_b].x += force_divr * dx.x;
                h_force.data[idx_b].y += force_divr * dx.y;
                h_force.data[idx_b].z += force_divr * dx.z;
                h_force.data[idx_b].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i*m_virial_pitch+idx_b]  += bond_virial[i];
                }

            if (idx_a < m_pdata->getN())
                {
                h_force.data[idx_a].x -= force_divr * dx.x;
                h_force.data[idx_a].y -= force_divr * dx.y;
                h_force.data[idx_a].z -= force_divr * dx.z;
                h_force.data[idx_a].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i*m_virial_pitch+idx_a]  += bond_virial[i];
                }
            }
        else
            {
            this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": bond out of bounds" << std::endl << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template < class evaluator >
CommFlags PotentialBond< evaluator >::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    flags[comm_flag::tag] = 1;

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    if (evaluator::needsDiameter())
        flags[comm_flag::diameter] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

//! Exports the PotentialBond class to python
/*! \param name Name of the class in the exported python module
    \tparam T class type to export. \b Must be an instantiated PotentialBOnd class template.
*/
template < class T > void export_PotentialBond(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(),pybind11::base<ForceCompute>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, const std::string& > ())
        .def("setParams", &T::setParams)
        ;
    }

#endif
