// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "TablePotential.h"

namespace py = pybind11;

#include <stdexcept>

/*! \file TablePotential.cc
    \brief Defines the TablePotential class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TablePotential::TablePotential(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<NeighborList> nlist,
                               unsigned int table_width,
                               const std::string& log_suffix)
        : ForceCompute(sysdef), m_nlist(nlist), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing TablePotential" << endl;

    // sanity checks
    assert(m_pdata);
    assert(m_nlist);

    if (table_width == 0)
        {
        m_exec_conf->msg->error() << "pair.table: Table width of 0 is invalid" << endl;
        throw runtime_error("Error initializing TablePotential");
        }

    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // allocate storage for the tables and parameters
    Index2DUpperTriangular table_index(m_ntypes);
    GlobalArray<Scalar2> tables(m_table_width, table_index.getNumElements(), m_exec_conf);
    m_tables.swap(tables);
    TAG_ALLOCATION(m_tables);

    GlobalArray<Scalar4> params(table_index.getNumElements(), m_exec_conf);
    m_params.swap(params);
    TAG_ALLOCATION(m_params);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_tables.get(), m_tables.getNumElements()*sizeof(Scalar2), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_params.get(), m_params.getNumElements()*sizeof(Scalar4), cudaMemAdviseSetReadMostly, 0);

        // prefetch
        auto& gpu_map = m_exec_conf->getGPUIds();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // prefetch data on all GPUs
            cudaMemPrefetchAsync(m_tables.get(), sizeof(Scalar2)*m_tables.getNumElements(), gpu_map[idev]);
            cudaMemPrefetchAsync(m_params.get(), sizeof(Scalar4)*m_params.getNumElements(), gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
    #endif

    assert(!m_tables.isNull());
    assert(!m_params.isNull());

    m_log_name = std::string("pair_table_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the number of types changes
    m_pdata->getNumTypesChangeSignal().connect<TablePotential, &TablePotential::slotNumTypesChange>(this);
    }

TablePotential::~TablePotential()
    {
    m_exec_conf->msg->notice(5) << "Destroying TablePotential" << endl;

    m_pdata->getNumTypesChangeSignal().disconnect<TablePotential, &TablePotential::slotNumTypesChange>(this);
    }

void TablePotential::slotNumTypesChange()
    {
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // skip the reallocation if the number of types does not change
    // this keeps old parameters when restoring a snapshot
    // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
    if (m_ntypes*(m_ntypes+1)/2 == m_params.getNumElements())
        return;

    // allocate storage for the tables and parameters
    Index2DUpperTriangular table_index(m_ntypes);
    GlobalArray<Scalar2> tables(m_table_width, table_index.getNumElements(), m_exec_conf);
    m_tables.swap(tables);
    TAG_ALLOCATION(m_tables);

    GlobalArray<Scalar4> params(table_index.getNumElements(), m_exec_conf);
    m_params.swap(params);
    TAG_ALLOCATION(m_params);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_tables.get(), m_tables.getNumElements()*sizeof(Scalar2), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_params.get(), m_params.getNumElements()*sizeof(Scalar4), cudaMemAdviseSetReadMostly, 0);

        // prefetch
        auto& gpu_map = m_exec_conf->getGPUIds();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // prefetch data on all GPUs
            cudaMemPrefetchAsync(m_tables.get(), sizeof(Scalar2)*m_tables.getNumElements(), gpu_map[idev]);
            cudaMemPrefetchAsync(m_params.get(), sizeof(Scalar4)*m_params.getNumElements(), gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
    #endif

    assert(!m_tables.isNull());
    assert(!m_params.isNull());
    }

/*! \param typ1 First particle type index in the pair to set
    \param typ2 Second particle type index in the pair to set
    \param V Table for the potential V
    \param F Table for the potential F (must be - dV / dr)
    \param rmin Minimum r in the potential
    \param rmax Maximum r in the potential
    \post Values from \a V and \a F are copied into the internal storage for type pair (typ1, typ2)
    \note There is no need to call this again for typ2,typ1
    \note See TablePotential for a detailed definition of rmin and rmax
*/
void TablePotential::setTable(unsigned int typ1,
                              unsigned int typ2,
                              const std::vector<Scalar> &V,
                              const std::vector<Scalar> &F,
                              Scalar rmin,
                              Scalar rmax)
    {
    // helpers to compute indices
    unsigned int cur_table_index = Index2DUpperTriangular(m_ntypes)(typ1, typ2);
    Index2D table_value(m_table_width);

    // access the arrays
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);

    // range check on the parameters
    if (rmin < 0 || rmax < 0 || rmax <= rmin)
        {
        m_exec_conf->msg->error() << "pair.table rmin, rmax (" << rmin << "," << rmax
             << ") is invalid" << endl;
        throw runtime_error("Error initializing TablePotential");
        }

    if (V.size() != m_table_width || F.size() != m_table_width)
        {
        m_exec_conf->msg->error() << "pair.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TablePotential");
        }

    // fill out the parameters
    h_params.data[cur_table_index].x = rmin;
    h_params.data[cur_table_index].y = rmax;
    h_params.data[cur_table_index].z = (rmax - rmin) / Scalar(m_table_width - 1);

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[table_value(i, cur_table_index)].x = V[i];
        h_tables.data[table_value(i, cur_table_index)].y = F[i];
        }
    }

/*! TablePotential provides
    - \c pair_table_energy
*/
std::vector< std::string > TablePotential::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

Scalar TablePotential::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair.table: " << quantity << " is not a valid log quantity for TablePotential" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation
*/
void TablePotential::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("Table pair");

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

    // index calculation helpers
    Index2DUpperTriangular table_index(m_ntypes);
    Index2D table_value(m_table_width);

    // for each particle
    for (int i = 0; i < (int) m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0,0,0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // access the index of this neighbor
            unsigned int k = h_nlist.data[head_i + j];
            // sanity check
            assert(k < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // access needed parameters
            unsigned int cur_table_index = table_index(typei, typej);
            Scalar4 params = h_params.data[cur_table_index];
            Scalar rmin = params.x;
            Scalar rmax = params.y;
            Scalar delta_r = params.z;

            // start computing the force
            Scalar rsq = dot(dx, dx);
            Scalar r = sqrt(rsq);

            // only compute the force if the particles are within the region defined by V
            if (r < rmax && r >= rmin)
                {
                // precomputed term
                Scalar value_f = (r - rmin) / delta_r;

                // compute index into the table and read in values
                unsigned int value_i = (unsigned int)floor(value_f);
                Scalar2 VF0 = h_tables.data[table_value(value_i, cur_table_index)];
                Scalar2 VF1 = h_tables.data[table_value(value_i+1, cur_table_index)];
                // unpack the data
                Scalar V0 = VF0.x;
                Scalar V1 = VF1.x;
                Scalar F0 = VF0.y;
                Scalar F1 = VF1.y;

                // compute the linear interpolation coefficient
                Scalar f = value_f - Scalar(value_i);

                // interpolate to get V and F;
                Scalar V = V0 + f * (V1 - V0);
                Scalar F = F0 + f * (F1 - F0);

                // convert to standard variables used by the other pair computes in HOOMD-blue
                Scalar forcemag_divr = Scalar(0.0);
                if (r > Scalar(0.0))
                    forcemag_divr = F / r;
                Scalar pair_eng = Scalar(0.5) * V;

                // compute the virial
                Scalar forcemag_div2r = Scalar(0.5) * forcemag_divr;
                virialxxi += forcemag_div2r*dx.x*dx.x;
                virialxyi += forcemag_div2r*dx.x*dx.y;
                virialxzi += forcemag_div2r*dx.x*dx.z;
                virialyyi += forcemag_div2r*dx.y*dx.y;
                virialyzi += forcemag_div2r*dx.y*dx.z;
                virialzzi += forcemag_div2r*dx.z*dx.z;

                // add the force, potential energy and virial to the particle i
                fi += dx*forcemag_divr;
                pei += pair_eng;

                // add the force to particle j if we are using the third law
                // only add force to local particles
                if (third_law && k < m_pdata->getN())
                    {
                    unsigned int mem_idx = k;
                    h_force.data[mem_idx].x -= dx.x*forcemag_divr;
                    h_force.data[mem_idx].y -= dx.y*forcemag_divr;
                    h_force.data[mem_idx].z -= dx.z*forcemag_divr;
                    h_force.data[mem_idx].w += pair_eng;
                    h_virial.data[0*m_virial_pitch+mem_idx] += forcemag_div2r * dx.x * dx.x;
                    h_virial.data[1*m_virial_pitch+mem_idx] += forcemag_div2r * dx.x * dx.y;
                    h_virial.data[2*m_virial_pitch+mem_idx] += forcemag_div2r * dx.x * dx.z;
                    h_virial.data[3*m_virial_pitch+mem_idx] += forcemag_div2r * dx.y * dx.y;
                    h_virial.data[4*m_virial_pitch+mem_idx] += forcemag_div2r * dx.y * dx.z;
                    h_virial.data[5*m_virial_pitch+mem_idx] += forcemag_div2r * dx.z * dx.z;
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        h_virial.data[0*m_virial_pitch+mem_idx] += virialxxi;
        h_virial.data[1*m_virial_pitch+mem_idx] += virialxyi;
        h_virial.data[2*m_virial_pitch+mem_idx] += virialxzi;
        h_virial.data[3*m_virial_pitch+mem_idx] += virialyyi;
        h_virial.data[4*m_virial_pitch+mem_idx] += virialyzi;
        h_virial.data[5*m_virial_pitch+mem_idx] += virialzzi;
        }

    if (m_prof) m_prof->pop();
    }

//! Exports the TablePotential class to python
void export_TablePotential(py::module& m)
    {
    py::class_<TablePotential, std::shared_ptr<TablePotential> >(m, "TablePotential", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, unsigned int, const std::string& >())
    .def("setTable", &TablePotential::setTable)
    ;
    }
