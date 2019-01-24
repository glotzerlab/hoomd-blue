// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "CGCMMForceCompute.h"

namespace py = pybind11;
#include <stdexcept>

/*! \file CGCMMForceCompute.cc
    \brief Defines the CGCMMForceCompute class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param r_cut Cutoff radius beyond which the force is 0
    \post memory is allocated and all parameters ljX are set to 0.0
*/
CGCMMForceCompute::CGCMMForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<NeighborList> nlist,
                                     Scalar r_cut)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut)
    {
    m_exec_conf->msg->notice(5) << "Constructing CGCMMForceCompute" << endl;

    assert(m_pdata);
    assert(m_nlist);

    if (r_cut < 0.0)
        {
        m_exec_conf->msg->error() << "pair.cgcmm: Negative r_cut makes no sense" << endl;
        throw runtime_error("Error initializing CGCMMForceCompute");
        }

    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // allocate storage for lj12, lj9, lj6, and lj4 parameters
    m_lj12 = new Scalar[m_ntypes*m_ntypes];
    m_lj9 = new Scalar[m_ntypes*m_ntypes];
    m_lj6 = new Scalar[m_ntypes*m_ntypes];
    m_lj4 = new Scalar[m_ntypes*m_ntypes];

    assert(m_lj12);
    assert(m_lj9);
    assert(m_lj6);
    assert(m_lj4);

    memset((void*)m_lj12, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj9,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj6,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj4,  0, sizeof(Scalar)*m_ntypes*m_ntypes);

    // connect to the ParticleData to receive notifications when the number of types changes
    m_pdata->getNumTypesChangeSignal().connect<CGCMMForceCompute, &CGCMMForceCompute::slotNumTypesChange>(this);
    }

void CGCMMForceCompute::slotNumTypesChange()
    {
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);


    // re-allocate storage for lj12, lj9, lj6, and lj4 parameters
    delete[] m_lj12;
    delete[] m_lj9;
    delete[] m_lj6;
    delete[] m_lj4;

    m_lj12 = new Scalar[m_ntypes*m_ntypes];
    m_lj9 = new Scalar[m_ntypes*m_ntypes];
    m_lj6 = new Scalar[m_ntypes*m_ntypes];
    m_lj4 = new Scalar[m_ntypes*m_ntypes];

    assert(m_lj12);
    assert(m_lj9);
    assert(m_lj6);
    assert(m_lj4);

    memset((void*)m_lj12, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj9,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj6,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj4,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    }


CGCMMForceCompute::~CGCMMForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CGCMMForceCompute" << endl;

    m_pdata->getNumTypesChangeSignal().disconnect<CGCMMForceCompute, &CGCMMForceCompute::slotNumTypesChange>(this);

    // deallocate our memory
    delete[] m_lj12;
    delete[] m_lj9;
    delete[] m_lj6;
    delete[] m_lj4;
    m_lj12 = NULL;
    m_lj9 = NULL;
    m_lj6 = NULL;
    m_lj4 = NULL;
    }


/*! \post The parameters \a lj12 through \a lj4 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
    \note \a lj? are low level parameters used in the calculation. In order to specify
    these for a 12-4 and 9-6 lennard jones formula (with alpha), they should be set to the following.

        12-4
    - \a lj12 = 2.598076 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = 0.0
    - \a lj4 = -alpha * 2.598076 * epsilon * pow(sigma,4.0)

        9-6
    - \a lj12 = 0.0
    - \a lj9 = 6.75 * epsilon * pow(sigma,9.0);
    - \a lj6 = -alpha * 6.75 * epsilon * pow(sigma,6.0)
    - \a lj4 = 0.0

       12-6
    - \a lj12 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = -alpha * 4.0 * epsilon * pow(sigma,4.0)
    - \a lj4 = 0.0

    Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
    is no need to call this function for symmetric pairs. Any pairs that this function is not called
    for will have lj12 through lj4 set to 0.0.

    \param typ1 Specifies one type of the pair
    \param typ2 Specifies the second type of the pair
    \param lj12 1/r^12 term
    \param lj9  1/r^9 term
    \param lj6  1/r^6 term
    \param lj4  1/r^4 term
*/
void CGCMMForceCompute::setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4)
    {
    if (typ1 >= m_ntypes || typ2 >= m_ntypes)
        {
        m_exec_conf->msg->error() << "pair.cgcmm: Trying to set params for a non existent type! " << typ1 << "," << typ2 << endl;
        throw runtime_error("Error setting parameters in CGCMMForceCompute");
        }

    // set lj12 in both symmetric positions in the matrix
    m_lj12[typ1*m_ntypes + typ2] = lj12;
    m_lj12[typ2*m_ntypes + typ1] = lj12;

    // set lj9 in both symmetric positions in the matrix
    m_lj9[typ1*m_ntypes + typ2] = lj9;
    m_lj9[typ2*m_ntypes + typ1] = lj9;

    // set lj6 in both symmetric positions in the matrix
    m_lj6[typ1*m_ntypes + typ2] = lj6;
    m_lj6[typ2*m_ntypes + typ1] = lj6;

    // set lj4 in both symmetric positions in the matrix
    m_lj4[typ1*m_ntypes + typ2] = lj4;
    m_lj4[typ2*m_ntypes + typ1] = lj4;
    }

/*! CGCMMForceCompute provides
    - \c cgcmm_energy
*/
std::vector< std::string > CGCMMForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pair_cgcmm_energy");
    return list;
    }

Scalar CGCMMForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pair_cgcmm_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair.cgcmm: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The CGCMM forces are computed for the given timestep. The neighborlist's
    compute method is called to ensure that it is up to date.

    \param timestep specifies the current time step of the simulation
*/
void CGCMMForceCompute::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("CGCMM pair");

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    const unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // access the particle data
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    // sanity check
    assert(h_pos.data != NULL);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // create a temporary copy of r_cut squared
    Scalar r_cut_sq = m_r_cut * m_r_cut;

    // tally up the number of forces calculated
    int64_t n_calc = 0;

    // for each particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access the lj12 and lj9 rows for the current particle type
        Scalar * __restrict__ lj12_row = &(m_lj12[typei*m_ntypes]);
        Scalar * __restrict__ lj9_row = &(m_lj9[typei*m_ntypes]);
        Scalar * __restrict__ lj6_row = &(m_lj6[typei*m_ntypes]);
        Scalar * __restrict__ lj4_row = &(m_lj4[typei*m_ntypes]);

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar viriali[6];
        for (int k = 0; k < 6; k++)
            viriali[k] = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;

            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int k = h_nlist.data[head_i + j];
            // sanity check
            assert(k < m_pdata->getN());

            // calculate dr (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
            dx = box.minImage(dx);

            // start computing the force
            // calculate r squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // only compute the force if the particles are closer than the cutoff (FLOPS: 1)
            if (rsq < r_cut_sq)
                {
                // compute the force magnitude/r in forcemag_divr (FLOPS: 14)
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r3inv = r2inv / sqrt(rsq);
                Scalar r6inv = r3inv * r3inv;
                Scalar forcemag_divr = r6inv * (r2inv * (Scalar(12.0)*lj12_row[typej]*r6inv + Scalar(9.0)*r3inv*lj9_row[typej]
                                                         + Scalar(6.0)*lj6_row[typej]) + Scalar(4.0)*lj4_row[typej]);

                // compute the pair energy and virial (FLOPS: 6)
                Scalar pair_virial[6];
                pair_virial[0] = Scalar(0.5) * dx.x * dx.x * forcemag_divr;
                pair_virial[1] = Scalar(0.5) * dx.x * dx.y * forcemag_divr;
                pair_virial[2] = Scalar(0.5) * dx.x * dx.z * forcemag_divr;
                pair_virial[3] = Scalar(0.5) * dx.y * dx.y * forcemag_divr;
                pair_virial[4] = Scalar(0.5) * dx.y * dx.z * forcemag_divr;
                pair_virial[5] = Scalar(0.5) * dx.z * dx.z * forcemag_divr;

                Scalar pair_eng = Scalar(0.5) * (r6inv * (lj12_row[typej] * r6inv + lj9_row[typej] * r3inv + lj6_row[typej]) + lj4_row[typej] * r2inv * r2inv);

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx*forcemag_divr;
                pei += pair_eng;
                for (unsigned int l = 0; l < 6; l++)
                    viriali[l] += pair_virial[l];

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                    {
                    h_force.data[k].x -= dx.x*forcemag_divr;
                    h_force.data[k].y -= dx.y*forcemag_divr;
                    h_force.data[k].z -= dx.z*forcemag_divr;
                    h_force.data[k].w += pair_eng;
                    for (unsigned int l = 0; l < 6; l++)
                        h_virial.data[l*virial_pitch+k] += pair_virial[l];
                    }
                }

            }

        // finally, increment the force, potential energy and virial for particle i
        // (MEM TRANSFER: 10 scalars / FLOPS: 5)
        h_force.data[i].x  += fi.x;
        h_force.data[i].y  += fi.y;
        h_force.data[i].z  += fi.z;
        h_force.data[i].w  += pei;
        for (int l = 0; l < 6; l++)
            h_virial.data[l*virial_pitch+i] += viriali[l];
        }

    int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+14+6+8);
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
    if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }

void export_CGCMMForceCompute(py::module& m)
    {
    py::class_<CGCMMForceCompute, std::shared_ptr<CGCMMForceCompute> >(m, "CGCMMForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar >())
    .def("setParams", &CGCMMForceCompute::setParams)
    ;
    }
