// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "EAMForceCompute.h"

#include <vector>
using namespace std;
#include <stdexcept>
namespace py = pybind11;

/*! \file EAMForceCompute.cc
    \brief Defines the EAMForceCompute class
*/

/*! \param sysdef System to compute forces on
    \param filename Name of EAM potential file to load
    \param type_of_file Undocumented parameter
*/
EAMForceCompute::EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing EAMForceCompute" << endl;

#ifndef SINGLE_PRECISION
    m_exec_conf->msg->error() << "EAM is not supported in double precision" << endl;
    throw runtime_error("Error initializing");
#endif

    assert(m_pdata);

    loadFile(filename, type_of_file);
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // connect to the ParticleData to receive notifications when the number of particle types changes
    m_pdata->getNumTypesChangeSignal().connect<EAMForceCompute, &EAMForceCompute::slotNumTypesChange>(this);
    }

EAMForceCompute::~EAMForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying EAMForceCompute" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<EAMForceCompute, &EAMForceCompute::slotNumTypesChange>(this);
    }

/*
type_of_file = 0 => EAM/Alloy
type_of_file = 1 => EAM/FS
*/
void EAMForceCompute::loadFile(char *filename, int type_of_file)
    {
    unsigned int tmp_int, type, i, j, k;
    double  tmp_mass, tmp;
    char tmp_str[5];

    const unsigned int MAX_TYPE_NUMBER = 10;
    const unsigned int MAX_POINT_NUMBER = 1000000;

    // Open potential file
    FILE *fp;
    fp = fopen(filename,"r");
    if (fp == NULL)
        {
        m_exec_conf->msg->error() << "pair.eam: Can not load EAM file" << endl;
        throw runtime_error("Error loading file");
        }
    for(i = 0; i < 3; i++) while(fgetc(fp) != '\n');
    m_ntypes = 0;
    int n = fscanf(fp, "%d", &m_ntypes);
    if (n != 1) throw runtime_error("Error parsing eam file");

    if(m_ntypes < 1 || m_ntypes > MAX_TYPE_NUMBER )
        {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Type number is greater than " << MAX_TYPE_NUMBER << endl;
        throw runtime_error("Error loading file");
        }
    // temporary array to count used types
    std::vector<bool> types_set(m_pdata->getNTypes(), false);
    //Load names of types.
    for(i = 0; i < m_ntypes; i++)
        {
        n = fscanf(fp, "%2s", tmp_str);
        if (n != 1) throw runtime_error("Error parsing eam file");

        names.push_back(tmp_str);
        unsigned int tid = m_pdata->getTypeByName(tmp_str);
        types.push_back(tid);
        types_set[tid] = true;
        }

    //Check that all types of atopms in xml file have description in potential file
    unsigned int count_types_set = 0;
    for (i = 0; i < m_pdata->getNTypes(); i++)
        {
        if (types_set[i])
            count_types_set++;
        }
    if(m_pdata->getNTypes() != count_types_set)
        {
        m_exec_conf->msg->error() << "pair.eam: not all atom types are defined in EAM potential file!!!" << endl;
        throw runtime_error("Error loading file");
        }

    //Load parameters.
    n = fscanf(fp,"%d", &nrho);
    if (n != 1) throw runtime_error("Error parsing eam file");

    n = fscanf(fp,"%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    drho = tmp;
    rdrho = (Scalar)(1.0 / drho);
    n = fscanf(fp,"%d", &nr);
    if (n != 1) throw runtime_error("Error parsing eam file");

    n = fscanf(fp,"%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    dr = tmp;
    rdr = (Scalar)(1.0 / dr);
    n = fscanf(fp,"%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    m_r_cut = tmp;
    if (nrho < 1 || nr < 1 || nrho > MAX_POINT_NUMBER || nr > MAX_POINT_NUMBER)
        {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Point number is greater than " << MAX_POINT_NUMBER << endl;
        throw runtime_error("Error loading file");
        }
    //Resize arrays for tables
    embeddingFunction.resize(nrho * m_ntypes);
    electronDensity.resize( nr * m_ntypes * m_ntypes);
    pairPotential.resize( (int)(0.5 * nr * (m_ntypes + 1) * m_ntypes) );
    derivativeEmbeddingFunction.resize(nrho * m_ntypes);
    derivativeElectronDensity.resize(nr * m_ntypes * m_ntypes);
    derivativePairPotential.resize((int)(0.5 * nr * (m_ntypes + 1) * m_ntypes));
    int res = 0;
    for(type = 0 ; type < m_ntypes; type++)
        {
        n = fscanf(fp, "%d %lg %lg %3s ", &tmp_int, &tmp_mass, &tmp, tmp_str);
        if (n != 4) throw runtime_error("Error parsing eam file");

        mass.push_back(tmp_mass);

        //Read F's array

        for(i = 0 ; i < nrho; i++)
            {
            res = fscanf(fp, "%lg", &tmp);
            embeddingFunction[types[type] * nrho + i] = (Scalar)tmp;
            }
        //Read Rho's arrays
        //If FS we need read N arrays
        //If Alloy we ned read 1 array, and then duplicate N-1 times.
        unsigned int count = 1;
        if(type_of_file == 1) count = m_ntypes;
        //Read
        for(j = 0; j < count; j++)
            {
            for(i = 0 ; i < nr; i++)
                {
                res = fscanf(fp, "%lg", &tmp);
                electronDensity[types[type] * m_ntypes * nr + j * nr + i] = (Scalar)tmp;
                }
            }

        for(j = 1; j <= m_ntypes - count; j++)
            {
            for(i = 0 ; i < nr; i++)
                {
                electronDensity[types[type] * m_ntypes * nr + j * nr + i] = electronDensity[i];
                }


            }
        }

    if(res == EOF || res == 0)
        {
        m_exec_conf->msg->error() << "pair.eam: EAM file is truncated " << endl;
        throw runtime_error("Error loading file");
        }
    //Read V(r)'s arrays
    for (k = 0; k < m_ntypes; k++)
        {
        for(j = 0; j <= k; j++)
            {
            for(i = 0 ; i < nr; i++)
                {
                res = fscanf(fp, "%lg", &tmp);
                pairPotential[(unsigned int)ceil(0.5 *(2 * m_ntypes - types[k] -1) * types[k] + types[j]) * nr + i].x = (Scalar)tmp;

                }
            }

        }

    fclose(fp);





    //Cumpute derivative of Embedding Function and Electron Density.
    for(type = 0 ; type < m_ntypes; type++)
        {
        for(i = 0 ; i < nrho - 1; i++)
            {
            derivativeEmbeddingFunction[i + types[type] * nrho] =
                (embeddingFunction[i + 1 + types[type] * nrho] - embeddingFunction[i + types[type] * nrho]) / drho;
            }
        for(j = 0; j < m_ntypes; j++)
            {
            for(i = 0 ; i < nr - 1; i++)
                {
                derivativeElectronDensity[types[type] * m_ntypes * nr +  j * nr + i ] =
                    (electronDensity[types[type] * m_ntypes * nr +  j * nr + i + 1] -
                    electronDensity[types[type] * m_ntypes * nr +  j * nr + i]) / dr;
                }
            }

        }


    //Cumpute derivative of Pair Potential.

    for (k = 0; k < m_ntypes; k ++)
        {
        for(j = 0; j <= k; j++)
            {
            for(i = 0 ; i < nr; i++)
                {
                if((i + 1)%nr == 0) continue;
                pairPotential[(unsigned int)ceil(0.5 *(2 * m_ntypes - types[k] -1) * types[k])  + types[j] * nr + i].y =
                (pairPotential[(unsigned int)ceil(0.5 *(2 * m_ntypes - types[k] -1) * types[k]) + types[j] * nr + i + 1].x - pairPotential[(unsigned int)ceil(0.5 *(2 * m_ntypes - types[k] -1) * types[k]) + types[j] * nr + i].x) / dr;

                }
            }

        }

    }
std::vector< std::string > EAMForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pair_eam_energy");
    return list;
    }

Scalar EAMForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pair_eam_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair.eam: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The lennard jones forces are computed for the given timestep. The neighborlist's
     compute method is called to ensure that it is up to date.

    \param timestep specifies the current time step of the simulation
*/
void EAMForceCompute::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("EAM pair");


    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list
    assert(m_nlist);
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // create a temporary copy of r_cut sqaured
    Scalar r_cut_sq = m_r_cut * m_r_cut;

    // tally up the number of forces calculated
    int64_t n_calc = 0;


    // for each particle
    vector<Scalar> atomElectronDensity;
    atomElectronDensity.resize(m_pdata->getN());
    vector<Scalar> atomDerivativeEmbeddingFunction;
    atomDerivativeEmbeddingFunction.resize(m_pdata->getN());
    vector<Scalar> atomEmbeddingFunction;
    atomEmbeddingFunction.resize(m_pdata->getN());
    unsigned int ntypes = m_pdata->getNTypes();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];

        // sanity check
        assert(typei < m_pdata->getNTypes());

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
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // start computing the force
            // calculate r squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);;
            // only compute the force if the particles are closer than the cuttoff (FLOPS: 1)
            if (rsq < r_cut_sq)
                {
                 Scalar position_scalar = sqrt(rsq) * rdr;
                 Scalar position = position_scalar;
                 unsigned int r_index = (unsigned int)position;
                 r_index = min(r_index,nr);
                 position -= r_index;
                 atomElectronDensity[i] += electronDensity[r_index + nr * (typei * ntypes + typej)] + derivativeElectronDensity[r_index + nr * (typei * ntypes + typej)] * position * dr;
                 if(third_law)
                    {
                    atomElectronDensity[k] += electronDensity[r_index + nr * (typej * ntypes + typei)]
                        + derivativeElectronDensity[r_index + nr * (typej * ntypes + typei)] * position * dr;
                    }
                }
            }
        }

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        Scalar position = atomElectronDensity[i] * rdrho;
        unsigned int r_index = (unsigned int)position;
        r_index = min(r_index,nrho);
        position -= (Scalar)r_index;
        atomDerivativeEmbeddingFunction[i] = derivativeEmbeddingFunction[r_index + typei * nrho];

        h_force.data[i].w += embeddingFunction[r_index + typei * nrho] + derivativeEmbeddingFunction[r_index + typei * nrho] * position * drho;
        }

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
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
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // start computing the force
            // calculate r squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            if (rsq >= r_cut_sq) continue;
            Scalar r = sqrt(rsq);
            Scalar inverseR = 1.0 / r;
            Scalar position = r * rdr;
            unsigned int r_index = (unsigned int)position;
            position = position - (Scalar)r_index;
            int shift = (typei>=typej)?(int)(0.5 * (2 * ntypes - typej -1)*typej + typei) * nr:(int)(0.5 * (2 * ntypes - typei -1)*typei + typej) * nr;
            //r_index = min(r_index,nr - 1);
            Scalar pair_eng = (pairPotential[r_index + shift].x +
                pairPotential[r_index + shift].y * position * dr) * inverseR;
            Scalar derivativePhi = (pairPotential[r_index + shift].y - pair_eng) * inverseR;
            Scalar derivativeRhoI = derivativeElectronDensity[r_index + typei * nr];
            Scalar derivativeRhoJ = derivativeElectronDensity[r_index + typej * nr];
            Scalar fullDerivativePhi = atomDerivativeEmbeddingFunction[i] * derivativeRhoJ +
                atomDerivativeEmbeddingFunction[k] * derivativeRhoI + derivativePhi;
            Scalar pairForce = - fullDerivativePhi * inverseR;
            // are the virial and potential energy correctly calculated
            // with respect to double counting?
            viriali[0] += dx.x*dx.x * pairForce;
            viriali[1] += dx.x*dx.y * pairForce;
            viriali[2] += dx.x*dx.z * pairForce;
            viriali[3] += dx.y*dx.y * pairForce;
            viriali[4] += dx.y*dx.z * pairForce;
            viriali[5] += dx.z*dx.z * pairForce;
            fxi += dx.x * pairForce;
            fyi += dx.y * pairForce;
            fzi += dx.z * pairForce;
            pei += pair_eng;

            if (third_law)
                {
                h_force.data[k].x -= dx.x * pairForce;
                h_force.data[k].y -= dx.y * pairForce;
                h_force.data[k].z -= dx.z * pairForce;
                }
            }
        h_force.data[i].x += fxi;
        h_force.data[i].y += fyi;
        h_force.data[i].z += fzi;
        h_force.data[i].w += pei;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+i] += viriali[k];
        }

    int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+9+6+8);
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
    if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }

void EAMForceCompute::set_neighbor_list(std::shared_ptr<NeighborList> nlist)
    {
    m_nlist = nlist;
    assert(m_nlist);
    }
Scalar EAMForceCompute::get_r_cut()
    {
    return m_r_cut;
    }
void export_EAMForceCompute(py::module& m)
    {
    py::class_<EAMForceCompute, std::shared_ptr<EAMForceCompute> >(m, "EAMForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, char*, int>())
    .def("set_neighbor_list", &EAMForceCompute::set_neighbor_list)
    .def("get_r_cut", &EAMForceCompute::get_r_cut)
    ;
    }
