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

// $Id: PotentialBond.h 2904 2010-03-23 17:10:10Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/computes/PotentialBond.h $
// Maintainer: phillicl

#include <boost/shared_ptr.hpp>

#include <boost/python.hpp>
using namespace boost::python;

#include "ForceCompute.h"
#include "BondData.h"
#include "GPUArray.h"

#include <vector>

/*! \file PotentialBond.h
    \brief Declares PotentialBond
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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
        PotentialBond(boost::shared_ptr<SystemDefinition> sysdef,
                      const std::string& log_suffix="");

        //! Destructor
        ~PotentialBond() { };

        //! Set the parameters
        virtual void setParams(unsigned int type, const param_type &param);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    protected:
        GPUArray<param_type> m_params;              //!< Bond parameters per type
        boost::shared_ptr<BondData> m_bond_data;    //!< Bond data to use in computing bonds
        std::string m_log_name;                     //!< Cached log name
        std::string m_prof_name;                    //!< Cached profiler name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param log_suffix Name given to this instance of the force
*/
template< class evaluator >
PotentialBond< evaluator >::PotentialBond(boost::shared_ptr<SystemDefinition> sysdef,
                      const std::string& log_suffix)
    : ForceCompute(sysdef)
    {
    assert(m_pdata);

    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    m_log_name = std::string("bond_") + evaluator::getName() + std::string("_energy") + log_suffix;
    m_prof_name = std::string("Pair ") + evaluator::getName();

    // allocate the parameters
    GPUArray<param_type> params(m_bond_data->getNBondTypes(), exec_conf);
    m_params.swap(params);
    }

/*! \param type Type of the bond to set parameters for
    \param param Parmeter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator >
void PotentialBond< evaluator >::setParams(unsigned int type, const param_type& param)
    {
    // make sure the type is valid
    if (type >= m_bond_data->getNBondTypes())
        {
        cout << endl << "***Error! Invalid bond type specified" << endl << endl;
        throw runtime_error("Error setting parameters in PotentialBond");
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
    vector<string> list;
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
        std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for PotentialPair"
                  << std::endl << endl;
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

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);


    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_diameter.data);
    assert(h_charge.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);

    // precalculate box lengths
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    Scalar Lx2 = Lx / Scalar(2.0);
    Scalar Ly2 = Ly / Scalar(2.0);
    Scalar Lz2 = Lz / Scalar(2.0);

    // for each of the bonds
    const unsigned int size = (unsigned int)m_bond_data->getNumBonds();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const Bond& bond = m_bond_data->getBond(i);
        assert(bond.a < m_pdata->getN());
        assert(bond.b < m_pdata->getN());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.a];
        unsigned int idx_b = h_rtag.data[bond.b];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());

        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL)
            {
            cerr << endl << "***Error! Found incomplete bond. Possibly the bond extension is greater than 1/2 the"
                 << endl << "          domain size in any direction. Try reducing the number of sub-divisions"
                 << endl << "          or increasing the bond stiffness."
                 << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }

        // calculate d\vec{r}
        // (MEM TRANSFER: 6 Scalars / FLOPS: 3)
        Scalar dx = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
        Scalar dy = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
        Scalar dz = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

        // access diameter (if needed)
        Scalar diameter_a = Scalar(0.0);
        Scalar diameter_b = Scalar(0.0);
        if (evaluator::needsDiameter())
            {
            diameter_a = h_diameter.data[idx_a];
            diameter_b = h_diameter.data[idx_b];
            }

        // acesss charge (if needed)
        Scalar charge_a = Scalar(0.0);
        Scalar charge_b = Scalar(0.0);
        if (evaluator::needsCharge())
            {
            charge_a = h_charge.data[idx_a];
            charge_b = h_charge.data[idx_b];
            }

        // if the vector crosses the box, pull it back
        // (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done))
        if (dx >= Lx2)
            dx -= Lx;
        else if (dx < -Lx2)
            dx += Lx;

        if (dy >= Ly2)
            dy -= Ly;
        else if (dy < -Ly2)
            dy += Ly;

        if (dz >= Lz2)
            dz -= Lz;
        else if (dz < -Lz2)
            dz += Lz;

        // sanity check
        assert(dx >= -Lx2 && dx < Lx2);
        assert(dy >= -Ly2 && dy < Ly2);
        assert(dz >= -Lz2 && dz < Lz2);

        // calculate r_ab squared
        Scalar rsq = dx*dx+dy*dy+dz*dz;

        // get parameters for this bond type
        param_type param = h_params.data[bond.type];

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
        bond_eng *= 0.5f;

        if (evaluated)
            {
            // calculate virial
            Scalar bond_virial[6];
            Scalar force_div2r = Scalar(1.0/2.0)*force_divr;
            bond_virial[0] = dx * dx * force_div2r; // xx
            bond_virial[1] = dx * dy * force_div2r; // xy
            bond_virial[2] = dx * dz * force_div2r; // xz
            bond_virial[3] = dy * dy * force_div2r; // yy
            bond_virial[4] = dy * dz * force_div2r; // yz
            bond_virial[5] = dz * dz * force_div2r; // zz

            // add the force to the particles (only for non-ghost particles)
            if (idx_b < m_pdata->getN())
                {
                h_force.data[idx_b].x += force_divr * dx;
                h_force.data[idx_b].y += force_divr * dy;
                h_force.data[idx_b].z += force_divr * dz;
                h_force.data[idx_b].w += bond_eng;
                for (unsigned int i = 0; i < 6; i++)
                    h_virial.data[i*m_virial_pitch+idx_b]  += bond_virial[i];
                }

            if (idx_a < m_pdata->getN())
                {
                h_force.data[idx_a].x -= force_divr * dx;
                h_force.data[idx_a].y -= force_divr * dy;
                h_force.data[idx_a].z -= force_divr * dz;
                h_force.data[idx_a].w += bond_eng;
                for (unsigned int i = 0; i < 6; i++)
                    h_virial.data[i*m_virial_pitch+idx_a]  += bond_virial[i];
                }
            }
        else
            {
            cerr << endl << "***Error! << " << evaluator::getName() << " bond out of bounds" << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }

    if (m_prof) m_prof->pop();
    }

//! Exports the PotentialBond class to python
/*! \param name Name of the class in the exported python module
    \tparam T class type to export. \b Must be an instatiated PotentialBOnd class template.
*/
template < class T > void export_PotentialBond(const std::string& name)
    {
    class_<T, boost::shared_ptr<T>, bases<ForceCompute>, boost::noncopyable >
        (name.c_str(), init< boost::shared_ptr<SystemDefinition>, const std::string& > ())
        .def("setParams", &T::setParams)
        ;
    }

#endif

