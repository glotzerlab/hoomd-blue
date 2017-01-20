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
    \param type_of_file EAM/Alloy=0, EAM/FS=1
*/
EAMForceCompute::EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file, int ifinter, int setnrho, int setnr)
        : ForceCompute(sysdef) {
    m_exec_conf->msg->notice(5) << "Constructing EAMForceCompute" << endl;

#ifndef SINGLE_PRECISION
    m_exec_conf->msg->error() << "EAM is not supported in double precision" << endl;
    throw runtime_error("Error initializing");
#endif

    assert(m_pdata);

    loadFile(filename, type_of_file, ifinter, setnrho, setnr);
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // connect to the ParticleData to receive notifications when the number of particle types changes
    m_pdata->getNumTypesChangeSignal().connect<EAMForceCompute, &EAMForceCompute::slotNumTypesChange>(this);
}

EAMForceCompute::~EAMForceCompute() {
    m_exec_conf->msg->notice(5) << "Destroying EAMForceCompute" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<EAMForceCompute, &EAMForceCompute::slotNumTypesChange>(this);
}

/*
type_of_file = 0 => EAM/Alloy
type_of_file = 1 => EAM/FS
*/
void EAMForceCompute::loadFile(char *filename, int type_of_file, int ifinter, int setnrho, int setnr) {
    unsigned int tmp_int, type, i, j, k;
    double tmp_mass, tmp;
    char tmp_str[5];

    const int MAX_TYPE_NUMBER = 10;
    const int MAX_POINT_NUMBER = 1000000;

    // Open potential file
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL) {
        m_exec_conf->msg->error() << "pair.eam: Can not load EAM file" << endl;
        throw runtime_error("Error loading file");
    }
    for (i = 0; i < 3; i++) while (fgetc(fp) != '\n');
    m_ntypes = 0;
    int n = fscanf(fp, "%d", &m_ntypes);
    if (n != 1) throw runtime_error("Error parsing eam file");

    if (m_ntypes < 1 || m_ntypes > MAX_TYPE_NUMBER) {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Type number is greater than "
                                  << MAX_TYPE_NUMBER << endl;
        throw runtime_error("Error loading file");
    }
    // temporary array to count used types
    std::vector<bool> types_set(m_pdata->getNTypes(), false);
    //Load names of types.
    for (i = 0; i < m_ntypes; i++) {
        n = fscanf(fp, "%2s", tmp_str);
        if (n != 1) throw runtime_error("Error parsing eam file");

        names.push_back(tmp_str);
        unsigned int tid = m_pdata->getTypeByName(tmp_str);
        types.push_back(tid);
        types_set[tid] = true;
    }

    //Check that all types of atoms in xml file have description in potential file
    unsigned int count_types_set = 0;
    for (i = 0; i < m_pdata->getNTypes(); i++) {
        if (types_set[i])
            count_types_set++;
    }
    if (m_pdata->getNTypes() != count_types_set) {
        m_exec_conf->msg->error() << "pair.eam: not all atom types are defined in EAM potential file!!!" << endl;
        throw runtime_error("Error loading file");
    }

    //Load parameters.
    n = fscanf(fp, "%d", &rawnrho);
    if (n != 1) throw runtime_error("Error parsing eam file");

    n = fscanf(fp, "%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    rawdrho = tmp;
    rawrdrho = (Scalar) (1.0 / rawdrho);
    n = fscanf(fp, "%d", &rawnr);
    if (n != 1) throw runtime_error("Error parsing eam file");

    n = fscanf(fp, "%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    rawdr = tmp;
    rawrdr = (Scalar) (1.0 / rawdr);
    n = fscanf(fp, "%lg", &tmp);
    if (n != 1) throw runtime_error("Error parsing eam file");

    m_r_cut = tmp;
    if (rawnrho < 1 || rawnr < 1 || rawnrho > MAX_POINT_NUMBER || rawnr > MAX_POINT_NUMBER) {
        m_exec_conf->msg->error() << "pair.eam: Invalid EAM file format: Point number is greater than "
                                  << MAX_POINT_NUMBER << endl;
        throw runtime_error("Error loading file");
    }
    //Resize arrays for tables
    rawembeddingFunction.resize(rawnrho * m_ntypes);
    rawelectronDensity.resize(rawnr * m_ntypes * m_ntypes);
    rawpairPotential.resize((int) (0.5 * rawnr * (m_ntypes + 1) * m_ntypes));
    int res = 0;
    for (type = 0; type < m_ntypes; type++) {
        n = fscanf(fp, "%d %lg %lg %3s ", &tmp_int, &tmp_mass, &tmp, tmp_str);
        if (n != 4) throw runtime_error("Error parsing eam file");

        nproton.push_back(tmp_int);
        mass.push_back(tmp_mass);
        lconst.push_back(tmp);
        atomcomment.push_back(tmp_str);

        //Read F's array

        for (i = 0; i < rawnrho; i++) {
            res = fscanf(fp, "%lg", &tmp);
            rawembeddingFunction[types[type] * rawnrho + i] = (Scalar) tmp;
        }
        //Read rho's arrays
        //If FS we need read N arrays
        //If Alloy we need read 1 array, and then duplicate N-1 times.
        if (type_of_file == 1) {
            for (j = 0; j < m_ntypes; j++) {
                for (i = 0; i < rawnr; i++) {
                    res = fscanf(fp, "%lg", &tmp);
                    rawelectronDensity[types[type] * m_ntypes * rawnr + types[j] * rawnr + i] = (Scalar) tmp;
                }
            }
        } else {
            for (i = 0; i < rawnr; i++) {
                res = fscanf(fp, "%lg", &tmp);
                rawelectronDensity[types[type] * m_ntypes * rawnr + i] = (Scalar) tmp;
                for (j = 1; j < m_ntypes; j++) {
                    rawelectronDensity[types[type] * m_ntypes * rawnr + j * rawnr + i] = rawelectronDensity[
                            types[type] * m_ntypes * rawnr + i];
                }
            }
        }
    }

    if (res == EOF || res == 0) {
        m_exec_conf->msg->error() << "pair.eam: EAM file is truncated " << endl;
        throw runtime_error("Error loading file");
    }
    //Read r*phi(r)'s arrays
    for (k = 0; k < m_ntypes; k++) {
        for (j = 0; j <= k; j++) {
            for (i = 0; i < rawnr; i++) {
                res = fscanf(fp, "%lg", &tmp);
                rawpairPotential[(int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + i] = (Scalar) tmp;
            }
        }
    }

    fclose(fp);

    // TODO: interpolation
    /* begin */

    double ratiorho, ratior;

    if (ifinter == 1) {
        if (setnrho > 10000 || rawnrho > 10000) {
            std::cout << "Number of tabulated electron density values was set too large," << std::endl;
            std::cout << "Reset it to Nrho = 10000, which should be sufficient as reference states." << std::endl;
            nrho = 10000;
        }
        else {
            nrho = setnrho;
        }

        if (setnr > 10000 || rawnr > 10000) {
            std::cout << "Number of tabulated distance density values was set too large," << std::endl;
            std::cout << "Reset it to Nr = 10000, which should be sufficient as reference states." << std::endl;
            nr = 10000;
        }
        else {
            nr = setnr;
        }
    }
    else {
        nrho = rawnrho;
        nr = rawnr;
    }


    ratiorho = (double) rawnrho / (double) nrho;
    drho = rawdrho * ratiorho;
    rdrho = (Scalar) (1.0 / drho);

    ratior = (double) rawnr / (double) nr;
    dr = rawdr * ratior;
    rdr = (Scalar) (1.0 / dr);

    embeddingFunction.resize(nrho * m_ntypes);
    electronDensity.resize(nr * m_ntypes * m_ntypes);
    pairPotential.resize((int) (0.5 * nr * (m_ntypes + 1) * m_ntypes));
    derivativeEmbeddingFunction.resize(nrho * m_ntypes);
    derivativeElectronDensity.resize(nr * m_ntypes * m_ntypes);
    derivativePairPotential.resize((int) (0.5 * nr * (m_ntypes + 1) * m_ntypes));
    iemb.resize(7, std::vector< Scalar >(rawnrho * m_ntypes));
    irho.resize(7, std::vector< Scalar >( rawnr * m_ntypes * m_ntypes ));
    irphi.resize(7, std::vector< Scalar >((int)(0.5 * rawnr * (m_ntypes + 1) * m_ntypes)));

    // Compute interpolation coefficients
    interpolate(rawnrho * m_ntypes, rawnrho, rawdrho, &rawembeddingFunction, &iemb);
    interpolate(rawnr * m_ntypes * m_ntypes, rawnr, rawdr, &rawelectronDensity, &irho);
    interpolate((int)(0.5 * rawnr * (m_ntypes + 1) * m_ntypes), rawnr, rawdr, &rawpairPotential, &irphi);

    // Interpolation
    double position;  // position
    unsigned int idxold;  // index relative to old array
    double crho;  // current rho
    double cr; // current r
    for (type = 0; type < m_ntypes; type++) {
        // embeddingFunction: F
        for (i = 0; i < nrho; i++) {
            crho = drho * i;
            position = crho * rawrdrho;
            idxold = (unsigned int) position ;
            position -= (double) idxold;
            embeddingFunction[type * nrho + i] =
                    iemb.at(6).at(idxold + type * rawnrho) +
                    iemb.at(5).at(idxold + type * rawnrho) * position +
                    iemb.at(4).at(idxold + type * rawnrho) * position * position +
                    iemb.at(3).at(idxold + type * rawnrho) * position * position * position;
            derivativeEmbeddingFunction[type * nrho + i] =
                    iemb.at(2).at(idxold + type * rawnrho) +
                    iemb.at(1).at(idxold + type * rawnrho) * position +
                    iemb.at(0).at(idxold + type * rawnrho) * position * position;
        }
        // electronDensity: rho
        for (j = 0; j < m_ntypes; j++) {
            for (i = 0; i < nr; i++) {
                cr = dr * i;
                position = cr * rawrdr;
                idxold = (unsigned int) position ;
                position -= (Scalar) idxold;
                electronDensity[i + type * m_ntypes * nr + j * nr] =
                        irho.at(6).at(idxold + type * m_ntypes * rawnr + j * rawnr) +
                        irho.at(5).at(idxold + type * m_ntypes * rawnr + j * rawnr) * position +
                        irho.at(4).at(idxold + type * m_ntypes * rawnr + j * rawnr) * position * position +
                        irho.at(3).at(idxold + type * m_ntypes * rawnr + j * rawnr) * position * position * position;
                derivativeEmbeddingFunction[i + type * m_ntypes * nr + j * nr] =
                        irho.at(2).at(idxold + type * m_ntypes * rawnr + j * rawnr) +
                        irho.at(1).at(idxold + type * m_ntypes * rawnr + j * rawnr) * position +
                        irho.at(0).at(idxold + type * m_ntypes * rawnr + j * rawnr) * position * position;
            }
        }
    }
    // pairPotential: rphi
    for (i = 0; i < nr; i++) {
        cr = dr * i;
        position = cr * rawrdr;
        idxold = (unsigned int) position;
        position -= (Scalar) idxold;
        for (k = 0; k < m_ntypes; k++) {
            for (j = 0; j <= k; j++) {
                pairPotential[(int) (0.5 * nr * (types[k] + 1) * types[k]) + types[j] * nr + i] =
                        irphi.at(6).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) +
                        irphi.at(5).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) *
                        position +
                        irphi.at(4).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) *
                        position * position +
                        irphi.at(3).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) *
                        position * position * position;
                derivativePairPotential[(int) (0.5 * nr * (types[k] + 1) * types[k]) + types[j] * nr + i] =
                        irphi.at(2).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) +
                        irphi.at(1).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) *
                        position +
                        irphi.at(0).at((int) (0.5 * rawnr * (types[k] + 1) * types[k]) + types[j] * rawnr + idxold) *
                        position * position;
            }
        }
    }

    FILE *f = fopen("/home/lyang/Repository/eam-potential/pot.eam", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    /* print title text */
    fprintf(f, "%s:\n", "Interpolated potential from");
    fprintf(f, "\"%s\"\n", filename);
    if (type_of_file == 0) {
        fprintf(f, "%s\n", "EAM/Alloy");
    }
    else {
        fprintf(f, "%s\n", "EAM/FS");
    }
    /* print type */
    fprintf(f, "%d %2s %2s\n", m_ntypes, names[0].c_str(), names[1].c_str());
    /* print new EAM global parameter */
    fprintf(f, "%d %lg %d %lg %lg\n", nrho, drho, nr, dr, m_r_cut);
    /* print EAM */
    for (type = 0; type < m_ntypes; type++) {
        // subtitle
        fprintf(f, "%d %lg %lg %3s\n", nproton[type], mass[type], lconst[type], atomcomment[type].c_str());
        // embeddingFunction: F(rho)
        for (i=0; i<nrho; i++) {
            fprintf(f, "%lg ", embeddingFunction[types[type] * nrho + i]);
            if ((i+1)%5 == 0) {
                fprintf(f, "\n");
            }
        }
        // electronDensity: rho(r)
        if (type_of_file == 1) {
            for (j = 0; j < m_ntypes; j++) {
                for (i = 0; i < nr; i++) {
                    fprintf(f, "%lg ", electronDensity[types[type] * m_ntypes * nr + types[j] * nr + i]);
                    if ((i+1)%5 == 0) {
                        fprintf(f, "\n");
                    }
                }
            }
        } else {
            for (i = 0; i < nr; i++) {
                fprintf(f, "%lg ", electronDensity[types[type] * m_ntypes * nr + i]);
                if ((i+1)%5 == 0) {
                    fprintf(f, "\n");
                }
            }
        }
    }
    // pariPotential: r*phi(r)
    for (k = 0; k < m_ntypes; k++) {
        for (j = 0; j <= k; j++) {
            for (i = 0; i < nr; i++) {
                fprintf(f, "%lg ", pairPotential[(int) (0.5 * nr * (types[k] + 1) * types[k]) + types[j] * nr + i]);
                if ((i+1)%5 == 0) {
                    fprintf(f, "\n");
                }
            }
        }
    }
    fclose(f);

    // free vectors
    irho.clear();
    irphi.clear();
    iemb.clear();

    /* end */

}

// interpolation function
void EAMForceCompute::interpolate(int num_all, int num_per, Scalar delta,
                                             std::vector< Scalar >* f, std::vector< std::vector< Scalar > >* spline) {
    int m, n;
    int num_block;
    int start, end;
    num_block = num_all / num_per;
    for (m=0; m<num_all; m++) {
        spline->at(6).at(m) = f->at(m);
    }
    for (n=0; n<num_block; n++) {
        start = num_per * n;
        end = num_per * (n+1) - 1;
        spline->at(5).at(start) = spline->at(6).at(start+1) - spline->at(6).at(start);
        spline->at(5).at(start+1) = 0.5 * (spline->at(6).at(start+2) - spline->at(6).at(start));
        spline->at(5).at(end-1) = 0.5 * (spline->at(6).at(end) - spline->at(6).at(end-2));
        spline->at(5).at(end) = spline->at(6).at(end) - spline->at(6).at(end-1);
        for (int m = 2; m < num_per-2; m++) {
            spline->at(5).at(start+m) = ((spline->at(6).at(start+m-2)-spline->at(6).at(start+m+2))
                                         + 8.0*(spline->at(6).at(start+m+1)-spline->at(6).at(start+m-1))) / 12.0;
        }
        for (int m = 0; m < num_per-1; m++) {
            spline->at(4).at(start+m) = 3.0*(spline->at(6).at(start+m+1)-spline->at(6).at(start+m)) -
                                        2.0*spline->at(5).at(start+m) - spline->at(5).at(start+m+1);
            spline->at(3).at(start+m) = spline->at(5).at(start+m) + spline->at(5).at(start+m+1) -
                                        2.0*(spline->at(6).at(start+m+1)-spline->at(6).at(start+m));
        }
        spline->at(4).at(end) = 0.0;
        spline->at(3).at(end) = 0.0;
    }
    for (m = 0; m < num_all; m++) {
        spline->at(2).at(m) = spline->at(5).at(m)/delta;
        spline->at(1).at(m) = 2.0*spline->at(4).at(m)/delta;
        spline->at(0).at(m) = 3.0*spline->at(3).at(m)/delta;
    }
}

std::vector<std::string> EAMForceCompute::getProvidedLogQuantities() {
    vector<string> list;
    list.push_back("pair_eam_energy");
    return list;
}

Scalar EAMForceCompute::getLogValue(const std::string &quantity, unsigned int timestep) {
    if (quantity == string("pair_eam_energy")) {
        compute(timestep);
        return calcEnergySum();
    } else {
        m_exec_conf->msg->error() << "pair.eam: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
    }
}

/*! \post The EAM forces are computed for the given timestep. The neighborlist's
     compute method is called to ensure that it is up to date.

    \param timestep specifies the current time step of the simulation
*/
void EAMForceCompute::computeForces(unsigned int timestep) {
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
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // Zero data for force calculation.
    memset((void *) h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void *) h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim &box = m_pdata->getBox();

    // create a temporary copy of r_cut squared
    Scalar r_cut_sq = m_r_cut * m_r_cut;

    // sum up the number of forces calculated
    int64_t n_calc = 0;

    // for each particle
    vector<Scalar> atomElectronDensity;
    atomElectronDensity.resize(m_pdata->getN());
    vector<Scalar> atomDerivativeEmbeddingFunction;
    atomDerivativeEmbeddingFunction.resize(m_pdata->getN());
    vector<Scalar> atomEmbeddingFunction;
    atomEmbeddingFunction.resize(m_pdata->getN());
    unsigned int ntypes = m_pdata->getNTypes();
    for (unsigned int i = 0; i < m_pdata->getN(); i++) {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int) h_n_neigh.data[i];

        for (unsigned int j = 0; j < size; j++) {
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
            // only compute the force if the particles are closer than the cut-off (FLOPS: 1)
            if (rsq < r_cut_sq) {
                Scalar position_scalar = sqrt(rsq) * rdr;
                Scalar position = position_scalar;
                unsigned int r_index = (unsigned int) position;
                r_index = min(r_index, nr - 1);
                position -= r_index;
                atomElectronDensity[i] += electronDensity[r_index + nr * (typej * ntypes + typei)]
                                          + derivativeElectronDensity[r_index + nr * (typej * ntypes + typei)] *
                                            position * dr;
                if (third_law) {
                    atomElectronDensity[k] += electronDensity[r_index + nr * (typei * ntypes + typej)]
                                              + derivativeElectronDensity[r_index + nr * (typei * ntypes + typej)] *
                                                position * dr;
                }
            }
        }
    }

    for (unsigned int i = 0; i < m_pdata->getN(); i++) {
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        Scalar position = atomElectronDensity[i] * rdrho;
        unsigned int r_index = (unsigned int) position;
        r_index = min(r_index, nrho - 1);
        position -= (Scalar) r_index;
        atomDerivativeEmbeddingFunction[i] = derivativeEmbeddingFunction[r_index + typei * nrho];

        h_force.data[i].w += embeddingFunction[r_index + typei * nrho] +
                             derivativeEmbeddingFunction[r_index + typei * nrho] * position * drho;
    }

    for (unsigned int i = 0; i < m_pdata->getN(); i++) {
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
        const unsigned int size = (unsigned int) h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++) {
            // increment our calculation counter
            n_calc++;

            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int k = h_nlist.data[head_i + j];
            // sanity check
            assert(k < m_pdata->getN());

            // calculate \Delta r (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
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
            unsigned int r_index = (unsigned int) position;
            r_index = min(r_index, nr - 1);
            position = position - (Scalar) r_index;
            int shift = (typei >= typej) ? (int) (0.5 * (2 * ntypes - typej - 1) * typej + typei) * nr :
                        (int) (0.5 * (2 * ntypes - typei - 1) * typei + typej) * nr;
            Scalar pair_eng = (pairPotential[r_index + shift] +
                        derivativePairPotential[r_index + shift] * position * dr) * inverseR;
            Scalar derivativePhi = (derivativePairPotential[r_index + shift] - pair_eng) * inverseR;
            Scalar derivativeRhoI = derivativeElectronDensity[r_index + typei * ntypes * nr + typej * nr];
            Scalar derivativeRhoJ = derivativeElectronDensity[r_index + typej * ntypes * nr + typei * nr];
            Scalar fullDerivativePhi = atomDerivativeEmbeddingFunction[i] * derivativeRhoJ +
                                       atomDerivativeEmbeddingFunction[k] * derivativeRhoI + derivativePhi;
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

            if (third_law) {
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
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5 + 4 + 10) * sizeof(Scalar) + n_calc * (1 + 3 + 1) * sizeof(Scalar);
    if (third_law) mem_transfer += n_calc * 10 * sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
}

void EAMForceCompute::set_neighbor_list(std::shared_ptr<NeighborList> nlist) {
    m_nlist = nlist;
    assert(m_nlist);
}

Scalar EAMForceCompute::get_r_cut() {
    return m_r_cut;
}

void export_EAMForceCompute(py::module &m) {
    py::class_<EAMForceCompute, std::shared_ptr<EAMForceCompute> >(m, "EAMForceCompute", py::base<ForceCompute>())
            .def(py::init<std::shared_ptr<SystemDefinition>, char *, int, int, int, int>())
            .def("set_neighbor_list", &EAMForceCompute::set_neighbor_list)
            .def("get_r_cut", &EAMForceCompute::get_r_cut);
}