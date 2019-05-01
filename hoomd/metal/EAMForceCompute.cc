// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: Lin Yang, Alex Travesset
// Previous Maintainer: Morozov

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
 \param type_of_file EAM/Alloy=0, EAM/FS=1
 */
EAMForceCompute::EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file) :
        ForceCompute(sysdef)
    {

    m_exec_conf->msg->notice(5) << "Constructing EAMForceCompute" << endl;

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

void EAMForceCompute::loadFile(char *filename, int type_of_file)
    {
    unsigned int tmp_int, type, i, j, k;
    double tmp_mass, tmp;
    char tmp_str[5];

    const int MAX_TYPE_NUMBER = 10;
    const int MAX_POINT_NUMBER = 1000000;

    // Open potential file
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL)
        {
        m_exec_conf->msg->error() << "pair.eam: Can not load EAM file" << endl;
        throw runtime_error("Error loading file");
        }
    for (i = 0; i < 3; i++)
        while (fgetc(fp) != '\n')
            ;
    m_ntypes = 0;
    int n = fscanf(fp, "%d", &m_ntypes);
    if (n != 1)
        throw runtime_error("Error parsing eam file");

    if (m_ntypes < 1 || m_ntypes > MAX_TYPE_NUMBER)
        {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Type number is greater than "
                << MAX_TYPE_NUMBER << endl;
        throw runtime_error("Error loading file");
        }

    // temporary array to count used types
    std::vector<bool> types_set(m_pdata->getNTypes(), false);
    //Load names of types.
    for (i = 0; i < m_ntypes; i++)
        {
        n = fscanf(fp, "%2s", tmp_str);
        if (n != 1)
            throw runtime_error("Error parsing eam file");
        names.push_back(tmp_str);
        unsigned int tid = m_pdata->getTypeByName(tmp_str);
        types.push_back(tid);
        types_set[tid] = true;
        }

    //Check that all types of atoms in xml file have description in potential file
    unsigned int count_types_set = 0;
    for (i = 0; i < m_pdata->getNTypes(); i++)
        {
        if (types_set[i])
            count_types_set++;
        }
    if (m_pdata->getNTypes() != count_types_set)
        {
        m_exec_conf->msg->error() << "pair.eam: not all atom types are defined in EAM potential file!!!" << endl;
        throw runtime_error("Error loading file");
        }

    //Load parameters.
    n = fscanf(fp, "%d", &nrho);
    if (n != 1)
        throw runtime_error("Error parsing eam file");

    n = fscanf(fp, "%lg", &tmp);
    if (n != 1)
        throw runtime_error("Error parsing eam file");
    drho = (Scalar) tmp;
    rdrho = (Scalar) (1.0 / drho);

    n = fscanf(fp, "%d", &nr);
    if (n != 1)
        throw runtime_error("Error parsing eam file");

    n = fscanf(fp, "%lg", &tmp);
    if (n != 1)
        throw runtime_error("Error parsing eam file");
    dr = (Scalar) tmp;
    rdr = (Scalar) (1.0 / dr);

    n = fscanf(fp, "%lg", &tmp);
    if (n != 1)
        throw runtime_error("Error parsing eam file");
    m_r_cut = (Scalar) tmp;

    if (nrho < 1 || nr < 1 || nrho > MAX_POINT_NUMBER || nr > MAX_POINT_NUMBER)
        {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Point number is greater than "
                << MAX_POINT_NUMBER << endl;
        throw runtime_error("Error loading file");
        }

    //allocate potential data storage
    GPUArray<Scalar4> t_F(nrho * m_ntypes, m_exec_conf);
    m_F.swap(t_F);
    ArrayHandle<Scalar4> h_F(m_F, access_location::host, access_mode::readwrite);

    GPUArray<Scalar4> t_rho(nr * m_ntypes * m_ntypes, m_exec_conf);
    m_rho.swap(t_rho);
    ArrayHandle<Scalar4> h_rho(m_rho, access_location::host, access_mode::readwrite);

    GPUArray<Scalar4> t_rphi((int) (0.5 * nr * (m_ntypes + 1) * m_ntypes), m_exec_conf);
    m_rphi.swap(t_rphi);
    ArrayHandle<Scalar4> h_rphi(m_rphi, access_location::host, access_mode::readwrite);

    GPUArray<Scalar4> t_dF(nrho * m_ntypes, m_exec_conf);
    m_dF.swap(t_dF);
    ArrayHandle<Scalar4> h_dF(m_dF, access_location::host, access_mode::readwrite);

    GPUArray<Scalar4> t_drho(nr * m_ntypes * m_ntypes, m_exec_conf);
    m_drho.swap(t_drho);
    ArrayHandle<Scalar4> h_drho(m_drho, access_location::host, access_mode::readwrite);

    GPUArray<Scalar4> t_drphi((int) (0.5 * nr * (m_ntypes + 1) * m_ntypes), m_exec_conf);
    m_drphi.swap(t_drphi);
    ArrayHandle<Scalar4> h_drphi(m_drphi, access_location::host, access_mode::readwrite);

    int res = 0;
    for (type = 0; type < m_ntypes; type++)
        {
        n = fscanf(fp, "%d %lg %lg %3s ", &tmp_int, &tmp_mass, &tmp, tmp_str);
        if (n != 4)
            throw runtime_error("Error parsing eam file");

        nproton.push_back(tmp_int);
        mass.push_back(tmp_mass);
        lconst.push_back(tmp);
        atomcomment.push_back(tmp_str);

        //Read F's array
        for (i = 0; i < nrho; i++)
            {
            res = fscanf(fp, "%lg", &tmp);
            h_F.data[types[type] * nrho + i].w = (Scalar) tmp;
            }

        //Read rho's arrays
        //If FS we need read N arrays
        //If Alloy we need read 1 array, and then duplicate N-1 times.
        if (type_of_file == 1)
            {
            for (j = 0; j < m_ntypes; j++)
                {
                for (i = 0; i < nr; i++)
                    {
                    res = fscanf(fp, "%lg", &tmp);
                    h_rho.data[types[type] * m_ntypes * nr + types[j] * nr + i].w = (Scalar) tmp;
                    }
                }
            }
        else
            {
            for (i = 0; i < nr; i++)
                {
                res = fscanf(fp, "%lg", &tmp);
                h_rho.data[types[type] * m_ntypes * nr + i].w = (Scalar) tmp;
                for (j = 1; j < m_ntypes; j++)
                    {
                    h_rho.data[types[type] * m_ntypes * nr + j * nr + i].w =
                            h_rho.data[types[type] * m_ntypes * nr + i].w;
                    }
                }
            }
        }

    if (res == EOF || res == 0)
        {
        m_exec_conf->msg->error() << "pair.eam: EAM file is truncated " << endl;
        throw runtime_error("Error loading file");
        }

    //Read r*phi(r)'s arrays
    for (k = 0; k < m_ntypes; k++)
        {
        for (j = 0; j <= k; j++)
            {
            for (i = 0; i < nr; i++)
                {
                res = fscanf(fp, "%lg", &tmp);
                h_rphi.data[(int) (0.5 * nr * (types[k] + 1) * types[k]) + types[j] * nr + i].w = (Scalar) tmp;
                }
            }
        }

    fclose(fp);

    // Compute interpolation coefficients
    interpolation(nrho * m_ntypes, nrho, drho, &h_F, &h_dF);
    interpolation(nr * m_ntypes * m_ntypes, nr, dr, &h_rho, &h_drho);
    interpolation((int) (0.5 * nr * (m_ntypes + 1) * m_ntypes), nr, dr, &h_rphi, &h_drphi);

    }

/*! compute cubic interpolation coefficients
 \param num_all Total number of data points
 \param num_per Number of data points per chunk
 \param delta Interval distance between data points
 \param f Data need to be interpolated
 \param df Derivative data to be recorded
 */
void EAMForceCompute::interpolation(int num_all, int num_per, Scalar delta, ArrayHandle<Scalar4> *f,
        ArrayHandle<Scalar4> *df)
    {
    int m, n;
    int num_block;
    int start, end;
    num_block = num_all / num_per;
    for (n = 0; n < num_block; n++)
        {
        start = num_per * n;
        end = num_per * (n + 1) - 1;
        f->data[start].z = f->data[start + 1].w - f->data[start].w;
        f->data[start + 1].z = 0.5 * f->data[start + 2].w - f->data[start].w;
        f->data[end - 1].z = 0.5 * f->data[end].w - f->data[end - 2].w;
        f->data[end].z = f->data[end].w - f->data[end - 1].w;
        for (int m = 2; m < num_per - 2; m++)
            {
            f->data[start + m].z = (f->data[start + m - 2].w - f->data[start + m + 2].w
                    + 8.0 * (f->data[start + m + 1].w - f->data[start + m - 1].w)) / 12.0;
            }
        for (int m = 0; m < num_per - 1; m++)
            {
            f->data[start + m].y = 3.0 * (f->data[start + m + 1].w - f->data[start + m].w) - 2.0 * f->data[start + m].z
                    - f->data[start + m + 1].z;
            f->data[start + m].x = f->data[start + m].z + f->data[start + m + 1].z
                    - 2.0 * (f->data[start + m + 1].w - f->data[start + m].w);
            }
        f->data[end].y = 0.0;
        f->data[end].x = 0.0;
        }
    for (m = 0; m < num_all; m++)
        {
        df->data[m].w = f->data[m].w;
        df->data[m].z = f->data[m].z / delta;
        df->data[m].y = 2.0 * f->data[m].y / delta;
        df->data[m].x = 3.0 * f->data[m].x / delta;
        }
    }

std::vector<std::string> EAMForceCompute::getProvidedLogQuantities()
    {
    vector < string > list;
    list.push_back("pair_eam_energy");
    return list;
    }

Scalar EAMForceCompute::getLogValue(const std::string &quantity, unsigned int timestep)
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

/*! \post The EAM forces are computed for the given timestep. The neighborlist's
 compute method is called to ensure that it is up to date.
 \param timestep specifies the current time step of the simulation
 */
void EAMForceCompute::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof)
        m_prof->push("EAM pair");

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
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // access potential table
    ArrayHandle<Scalar4> h_F(m_F, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_dF(m_dF, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_rho(m_rho, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_drho(m_drho, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_rphi(m_rphi, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_drphi(m_drphi, access_location::host, access_mode::read);

    // index and remainder
    Scalar position;  // look up position, scalar
    unsigned int int_position;  // look up index for position, integer
    unsigned int idxs; // look up index in F, rho, rphi array, considering shift, integer
    Scalar remainder;  // look up remainder in array, integer
    Scalar4 v, dv;  // value, d(value)

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_F.data);
    assert(h_dF.data);
    assert(h_rho.data);
    assert(h_drho.data);
    assert(h_rphi.data);
    assert(h_drphi.data);

    // Zero data for force calculation.
    memset((void *) h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void *) h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim &box = m_pdata->getBox();

    // create a temporary copy of r_cut squared
    Scalar r_cut_sq = m_r_cut * m_r_cut;

    // sum up the number of forces calculated
    int64_t n_calc = 0;

    // parameters for each particle
    vector<Scalar> atomElectronDensity;
    atomElectronDensity.resize(m_pdata->getN());
    vector<Scalar> atomDerivativeEmbeddingFunction;
    atomDerivativeEmbeddingFunction.resize(m_pdata->getN());
    vector<Scalar> atomEmbeddingFunction;
    atomEmbeddingFunction.resize(m_pdata->getN());
    unsigned int ntypes = m_pdata->getNTypes();

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int) h_n_neigh.data[i];

        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;

            // access the index of this neighbor
            unsigned int k = h_nlist.data[head_i + j];
            // sanity check
            assert(k < m_pdata->getN());

            // calculate dr
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // start computing the force
            // calculate r squared
            Scalar rsq = dot(dx, dx);
            ;
            // only compute the force if the particles are closer than the cut-off
            if (rsq < r_cut_sq)
                {
                // calculate position r for rho(r)
                position = sqrt(rsq) * rdr;
                int_position = (unsigned int) position;
                int_position = min(int_position, nr - 1);
                remainder = position - int_position;
                // calculate P = sum{rho}
                idxs = int_position + nr * (typej * ntypes + typei);
                v = h_rho.data[idxs];
                atomElectronDensity[i] += v.w + v.z * remainder + v.y * remainder * remainder
                        + v.x * remainder * remainder * remainder;
                // if third_law, pair it
                if (third_law)
                    {
                    idxs = int_position + nr * (typei * ntypes + typej);
                    v = h_rho.data[idxs];
                    atomElectronDensity[k] += v.w + v.z * remainder + v.y * remainder * remainder
                            + v.x * remainder * remainder * remainder;
                    }
                }
            }
        }

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // calculate position rho for F(rho)
        position = atomElectronDensity[i] * rdrho;
        int_position = (unsigned int) position;
        int_position = min(int_position, nrho - 1);
        remainder = position - int_position;

        idxs = int_position + typei * nrho;
        v = h_F.data[idxs];
        dv = h_dF.data[idxs];
        // compute dF / dP
        atomDerivativeEmbeddingFunction[i] = dv.z + dv.y * remainder + dv.x * remainder * remainder;
        // compute embedded energy F(P), sum up each particle
        h_force.data[i].w += v.w + v.z * remainder + v.y * remainder * remainder
                + v.x * remainder * remainder * remainder;

        }

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type
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
        const unsigned int size = (unsigned int) h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;

            // access the index of this neighbor
            unsigned int k = h_nlist.data[head_i + j];
            // sanity check
            assert(k < m_pdata->getN());

            // calculate \Delta r
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // start computing the force
            // calculate r squared
            Scalar rsq = dot(dx, dx);

            // calculate position r for phi(r)
            if (rsq >= r_cut_sq)
                continue;
            Scalar r = sqrt(rsq);
            Scalar inverseR = 1.0 / r;
            position = r * rdr;
            int_position = (unsigned int) position;
            int_position = min(int_position, nr - 1);
            remainder = position - int_position;
            // calculate the shift position for type ij
            int shift =
                    (typei >= typej) ?
                            (int) (0.5 * (2 * ntypes - typej - 1) * typej + typei) * nr :
                            (int) (0.5 * (2 * ntypes - typei - 1) * typei + typej) * nr;

            idxs = int_position + shift;
            v = h_rphi.data[idxs];
            dv = h_drphi.data[idxs];
            // pair_eng = phi
            Scalar pair_eng = (v.w + v.z * remainder + v.y * remainder * remainder
                    + v.x * remainder * remainder * remainder) * inverseR;
            // derivativePhi = (phi + r * dphi/dr - phi) * 1/r = dphi / dr
            Scalar derivativePhi = (dv.z + dv.y * remainder + dv.x * remainder * remainder - pair_eng) * inverseR;
            // derivativeRhoI = drho / dr of i
            idxs = int_position + typei * ntypes * nr + typej * nr;
            dv = h_drho.data[idxs];
            Scalar derivativeRhoI = dv.z + dv.y * remainder + dv.x * remainder * remainder;
            // derivativeRhoJ = drho / dr of j
            idxs = int_position + typej * ntypes * nr + typei * nr;
            dv = h_drho.data[idxs];
            Scalar derivativeRhoJ = dv.z + dv.y * remainder + dv.x * remainder * remainder;
            // fullDerivativePhi = dF/dP * drho / dr for j + dF/dP * drho / dr for j + phi
            Scalar fullDerivativePhi = atomDerivativeEmbeddingFunction[i] * derivativeRhoJ
                    + atomDerivativeEmbeddingFunction[k] * derivativeRhoI + derivativePhi;
            // compute forces
            Scalar pairForce = -fullDerivativePhi * inverseR;
            viriali[0] += dx.x * dx.x * pairForce;
            viriali[1] += dx.x * dx.y * pairForce;
            viriali[2] += dx.x * dx.z * pairForce;
            viriali[3] += dx.y * dx.y * pairForce;
            viriali[4] += dx.y * dx.z * pairForce;
            viriali[5] += dx.z * dx.z * pairForce;
            fxi += dx.x * pairForce;
            fyi += dx.y * pairForce;
            fzi += dx.z * pairForce;
            pei += pair_eng * 0.5;

            if (third_law)
                {
                h_force.data[k].x -= dx.x * pairForce;
                h_force.data[k].y -= dx.y * pairForce;
                h_force.data[k].z -= dx.z * pairForce;
                h_force.data[k].w += pair_eng * 0.5;
                }
            }
        h_force.data[i].x += fxi;
        h_force.data[i].y += fyi;
        h_force.data[i].z += fzi;
        h_force.data[i].w += pei;
        for (int k = 0; k < 6; k++)
            h_virial.data[k * virial_pitch + i] += viriali[k];
        }

    int64_t flops = m_pdata->getN() * 5 + n_calc * (3 + 5 + 9 + 1 + 9 + 6 + 8);
    if (third_law)
        flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5 + 4 + 10) * sizeof(Scalar) + n_calc * (1 + 3 + 1) * sizeof(Scalar);
    if (third_law)
        mem_transfer += n_calc * 10 * sizeof(Scalar);
    if (m_prof)
        m_prof->pop(flops, mem_transfer);
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

void export_EAMForceCompute(py::module &m)
    {
    py::class_<EAMForceCompute, std::shared_ptr<EAMForceCompute> >(m, "EAMForceCompute", py::base<ForceCompute>()).def(
            py::init<std::shared_ptr<SystemDefinition>, char *, int>()).def("set_neighbor_list",
            &EAMForceCompute::set_neighbor_list).def("get_r_cut", &EAMForceCompute::get_r_cut);
    }
