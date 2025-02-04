// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TableDihedralForceCompute.h"
#include "hoomd/VectorMath.h"

#include <stdexcept>

/*! \file TableDihedralForceCompute.cc
    \brief Defines the TableDihedralForceCompute class
*/

using namespace std;

// SMALL a relatively small number
#define SMALL 0.001f

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param table_width Width the tables will be in memory
*/
TableDihedralForceCompute::TableDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                     unsigned int table_width)
    : ForceCompute(sysdef), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing TableDihedralForceCompute" << endl;

    assert(m_pdata);

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    if (table_width == 0)
        {
        throw runtime_error("Dihedral table must have width greater than 0.");
        }

    // allocate storage for the tables and parameters
    GPUArray<Scalar2> tables(m_table_width, m_dihedral_data->getNTypes(), m_exec_conf);
    m_tables.swap(tables);
    assert(!m_tables.isNull());

    // helper to compute indices
    Index2D table_value((unsigned int)m_tables.getPitch(),
                        (unsigned int)m_dihedral_data->getNTypes());
    m_table_value = table_value;
    }

TableDihedralForceCompute::~TableDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying TableDihedralForceCompute" << endl;
    }

/*! \param type Type of the dihedral to set parameters for
    \param V Table for the potential V
    \param T Table for the potential T (must be - dV / dphi)
    \post Values from \a V and \a T are copied into the internal storage for type pair (type)
*/
void TableDihedralForceCompute::setTable(unsigned int type,
                                         const std::vector<Scalar>& V,
                                         const std::vector<Scalar>& T)
    {
    // make sure the type is valid
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    // access the arrays
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);

    if (V.size() != m_table_width || T.size() != m_table_width)
        {
        m_exec_conf->msg->error()
            << "dihedral.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TableDihedralForceCompute");
        }

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[m_table_value(i, type)].x = V[i];
        h_tables.data[m_table_value(i, type)].y = T[i];
        }
    }

void TableDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto type_id = m_dihedral_data->getTypeByName(type);

    const auto V_py = params["U"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
    const auto T_py = params["tau"].cast<pybind11::array_t<Scalar>>().unchecked<1>();

    std::vector<Scalar> V(V_py.size());
    std::vector<Scalar> T(T_py.size());

    std::copy(V_py.data(0), V_py.data(0) + V_py.size(), V.data());
    std::copy(T_py.data(0), T_py.data(0) + T_py.size(), T.data());

    setTable(type_id, V, T);
    }

/// Get the parameters for a particular type.
pybind11::dict TableDihedralForceCompute::getParams(std::string type)
    {
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);

    auto type_id = m_dihedral_data->getTypeByName(type);
    pybind11::dict params;

    auto V = pybind11::array_t<Scalar>(m_table_width);
    auto V_unchecked = V.mutable_unchecked<1>();
    auto T = pybind11::array_t<Scalar>(m_table_width);
    auto T_unchecked = T.mutable_unchecked<1>();

    for (unsigned int i = 0; i < m_table_width; i++)
        {
        V_unchecked(i) = h_tables.data[m_table_value(i, type_id)].x;
        T_unchecked(i) = h_tables.data[m_table_value(i, type_id)].y;
        }

    params["U"] = V;
    params["tau"] = T;

    return params;
    }

/*! \post The table based forces are computed for the given timestep.
\param timestep specifies the current time step of the simulation
*/
void TableDihedralForceCompute::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    size_t virial_pitch = m_virial.getPitch();

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);

    // for each of the dihedrals
    const unsigned int size = (unsigned int)m_dihedral_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const DihedralData::members_t& dihedral = m_dihedral_data->getMembersByIndex(i);
        assert(dihedral.tag[0] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[1] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[2] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[3] <= m_pdata->getMaximumTag());

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[dihedral.tag[0]];
        unsigned int idx_b = h_rtag.data[dihedral.tag[1]];
        unsigned int idx_c = h_rtag.data[dihedral.tag[2]];
        unsigned int idx_d = h_rtag.data[dihedral.tag[3]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.harmonic: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1]
                << " " << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x; // vb1x
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y; // vb1y
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z; // vb1z

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x; // vb2x
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y; // vb2y
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z; // vb2z

        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;

        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x; // vb3x
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y; // vb3y
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z; // vb3z

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);
        dcbm = box.minImage(dcbm);

        // c0 calculation
        Scalar sb1 = 1.0 / (dab.x * dab.x + dab.y * dab.y + dab.z * dab.z);
        Scalar sb3 = 1.0 / (ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z);

        Scalar rb1 = fast::sqrt(sb1);
        Scalar rb3 = fast::sqrt(sb3);

        Scalar c0 = (dab.x * ddc.x + dab.y * ddc.y + dab.z * ddc.z) * rb1 * rb3;

        // 1st and 2nd angle

        Scalar b1mag2 = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar b1mag = fast::sqrt(b1mag2);
        Scalar b2mag2 = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z;
        Scalar b2mag = fast::sqrt(b2mag2);
        Scalar b3mag2 = ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z;
        Scalar b3mag = fast::sqrt(b3mag2);

        Scalar ctmp = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z;
        Scalar r12c1 = 1.0 / (b1mag * b2mag);
        Scalar c1mag = ctmp * r12c1;

        ctmp = dcbm.x * ddc.x + dcbm.y * ddc.y + dcbm.z * ddc.z;
        Scalar r12c2 = 1.0 / (b2mag * b3mag);
        Scalar c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        Scalar sin2 = 1.0 - c1mag * c1mag;
        if (sin2 < 0.0)
            sin2 = 0.0;
        Scalar sc1 = fast::sqrt(sin2);
        if (sc1 < SMALL)
            sc1 = SMALL;
        sc1 = 1.0 / sc1;

        sin2 = 1.0 - c2mag * c2mag;
        if (sin2 < 0.0)
            sin2 = 0.0;
        Scalar sc2 = fast::sqrt(sin2);
        if (sc2 < SMALL)
            sc2 = SMALL;
        sc2 = 1.0 / sc2;

        Scalar s12 = sc1 * sc2;
        Scalar c = (c0 + c1mag * c2mag) * s12;

        if (c > 1.0)
            c = 1.0;
        if (c < -1.0)
            c = -1.0;

        // determinant
        Scalar det = dot(dab,
                         make_scalar3(ddc.y * dcb.z - ddc.z * dcb.y,
                                      ddc.z * dcb.x - ddc.x * dcb.z,
                                      ddc.x * dcb.y - ddc.y * dcb.x));
        // phi
        Scalar phi = acos(c);
        if (det < 0)
            phi = -phi;

        // precomputed term
        Scalar delta_phi = Scalar(2.0 * M_PI) / Scalar(m_table_width - 1);
        Scalar value_f = (Scalar(M_PI) + phi) / delta_phi;

        // compute index into the table and read in values

        /// Here we use the table!!
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
        unsigned int value_i = (unsigned int)value_f;
        Scalar2 VT0 = h_tables.data[m_table_value(value_i, dihedral_type)];
        Scalar2 VT1 = h_tables.data[m_table_value(value_i + 1, dihedral_type)];
        // unpack the data
        Scalar V0 = VT0.x;
        Scalar V1 = VT1.x;
        Scalar T0 = VT0.y;
        Scalar T1 = VT1.y;

        // compute the linear interpolation coefficient
        Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and T;
        Scalar V = V0 + f * (V1 - V0);
        Scalar T = T0 + f * (T1 - T0);

        // from Blondel and Karplus 1995
        vec3<Scalar> A = cross(vec3<Scalar>(dab), vec3<Scalar>(dcbm));
        Scalar Asq = dot(A, A);

        vec3<Scalar> B = cross(vec3<Scalar>(ddc), vec3<Scalar>(dcbm));
        Scalar Bsq = dot(B, B);

        Scalar3 f_a = -T * vec_to_scalar3(b2mag / Asq * A);
        Scalar3 f_b
            = -f_a
              + T / b2mag * vec_to_scalar3(dot(dab, dcbm) / Asq * A - dot(ddc, dcbm) / Bsq * B);
        Scalar3 f_c = T
                      * vec_to_scalar3(dot(ddc, dcbm) / Bsq / b2mag * B
                                       - dot(dab, dcbm) / Asq / b2mag * A - b2mag / Bsq * B);
        Scalar3 f_d = T * b2mag / Bsq * vec_to_scalar3(B);

        // Now, apply the force to each individual atom a,b,c,d
        // and accumulate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        Scalar dihedral_eng
            = V * Scalar(0.25); // the .125 term comes from distributing over the four particles

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = (1. / 4.) * (dab.x * f_a.x + dcb.x * f_c.x + (ddc.x + dcb.x) * f_d.x);
        dihedral_virial[1] = (1. / 4.) * (dab.y * f_a.x + dcb.y * f_c.x + (ddc.y + dcb.y) * f_d.x);
        dihedral_virial[2] = (1. / 4.) * (dab.z * f_a.x + dcb.z * f_c.x + (ddc.z + dcb.z) * f_d.x);
        dihedral_virial[3] = (1. / 4.) * (dab.y * f_a.y + dcb.y * f_c.y + (ddc.y + dcb.y) * f_d.y);
        dihedral_virial[4] = (1. / 4.) * (dab.z * f_a.y + dcb.z * f_c.y + (ddc.z + dcb.z) * f_d.y);
        dihedral_virial[5] = (1. / 4.) * (dab.z * f_a.z + dcb.z * f_c.z + (ddc.z + dcb.z) * f_d.z);

        h_force.data[idx_a].x += f_a.x;
        h_force.data[idx_a].y += f_a.y;
        h_force.data[idx_a].z += f_a.z;
        h_force.data[idx_a].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_a] += dihedral_virial[k];

        h_force.data[idx_b].x += f_b.x;
        h_force.data[idx_b].y += f_b.y;
        h_force.data[idx_b].z += f_b.z;
        h_force.data[idx_b].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_b] += dihedral_virial[k];

        h_force.data[idx_c].x += f_c.x;
        h_force.data[idx_c].y += f_c.y;
        h_force.data[idx_c].z += f_c.z;
        h_force.data[idx_c].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_c] += dihedral_virial[k];

        h_force.data[idx_d].x += f_d.x;
        h_force.data[idx_d].y += f_d.y;
        h_force.data[idx_d].z += f_d.z;
        h_force.data[idx_d].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_d] += dihedral_virial[k];
        }
    }

namespace detail
    {
//! Exports the TableDihedralForceCompute class to python
void export_TableDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<TableDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<TableDihedralForceCompute>>(m, "TableDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def("setTable", &TableDihedralForceCompute::setTable)
        .def_property_readonly("width", &TableDihedralForceCompute::getWidth)
        .def("setParams", &TableDihedralForceCompute::setParamsPython)
        .def("getParams", &TableDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
