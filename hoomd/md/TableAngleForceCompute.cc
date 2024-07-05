// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TableAngleForceCompute.h"

#include <stdexcept>

/*! \file TableAngleForceCompute.cc
    \brief Defines the TableAngleForceCompute class
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
TableAngleForceCompute::TableAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                               unsigned int table_width)
    : ForceCompute(sysdef), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing TableAngleForceCompute" << endl;

    assert(m_pdata);

    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_angle_data->getNTypes() == 0)
        {
        throw runtime_error("No angle types defined.");
        }

    if (table_width == 0)
        {
        throw runtime_error("Angle table must have width greater than 0.");
        }

    // allocate storage for the tables and parameters
    GPUArray<Scalar2> tables(m_table_width, m_angle_data->getNTypes(), m_exec_conf);
    m_tables.swap(tables);
    assert(!m_tables.isNull());

    // helper to compute indices
    Index2D table_value((unsigned int)m_tables.getPitch(), (unsigned int)m_angle_data->getNTypes());
    m_table_value = table_value;
    }

TableAngleForceCompute::~TableAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying TableAngleForceCompute" << endl;
    }

/*! \param type Type of the angle to set parameters for
    \param V Table for the potential V
    \param T Table for the torque T (must be - dV / dtheta)
    \post Values from \a V and \a T are copied into the internal storage for type pair (type)
*/
void TableAngleForceCompute::setTable(unsigned int type,
                                      const std::vector<Scalar>& V,
                                      const std::vector<Scalar>& T)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }

    // access the arrays
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);

    if (V.size() != m_table_width || T.size() != m_table_width)
        {
        m_exec_conf->msg->error()
            << "angle.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TableAngleForceCompute");
        }

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[m_table_value(i, type)].x = V[i];
        h_tables.data[m_table_value(i, type)].y = T[i];
        }
    }

void TableAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto type_id = m_angle_data->getTypeByName(type);

    const auto V_py = params["U"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
    const auto T_py = params["tau"].cast<pybind11::array_t<Scalar>>().unchecked<1>();

    std::vector<Scalar> V(V_py.size());
    std::vector<Scalar> T(T_py.size());

    std::copy(V_py.data(0), V_py.data(0) + V_py.size(), V.data());
    std::copy(T_py.data(0), T_py.data(0) + T_py.size(), T.data());

    setTable(type_id, V, T);
    }

/// Get the parameters for a particular type.
pybind11::dict TableAngleForceCompute::getParams(std::string type)
    {
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);

    auto type_id = m_angle_data->getTypeByName(type);
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
void TableAngleForceCompute::computeForces(uint64_t timestep)
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
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);

    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[2] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];
        unsigned int idx_c = h_rtag.data[angle.tag[2]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "angle.table: angle " << angle.tag[0] << " " << angle.tag[1] << " "
                << angle.tag[2] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x; // used for the 1-3 JL interaction
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        Scalar delta_th = Scalar(M_PI) / Scalar(m_table_width - 1);

        // start computing the force
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z;
        Scalar rcb = sqrt(rsqcb);

        // cosine of theta
        Scalar c_abbc = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z;
        c_abbc /= rab * rcb;

        if (c_abbc > 1.0)
            c_abbc = 1.0;
        if (c_abbc < -1.0)
            c_abbc = -1.0;

        // 1/sine of theta
        Scalar s_abbc = sqrt(1.0 - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = 1.0 / s_abbc;

        // theta
        Scalar theta = acos(c_abbc);

        // precomputed term
        Scalar value_f = theta / delta_th;

        // compute index into the table and read in values

        /// Here we use the table!!
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        unsigned int value_i = (unsigned int)(slow::floor(value_f));
        Scalar2 VT0 = h_tables.data[m_table_value(value_i, angle_type)];
        Scalar2 VT1 = h_tables.data[m_table_value(value_i + 1, angle_type)];
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

        Scalar a = T * s_abbc;
        Scalar a11 = a * c_abbc / rsqab;
        Scalar a12 = -a / (rab * rcb);
        Scalar a22 = a * c_abbc / rsqcb;

        Scalar fab[3], fcb[3];

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        Scalar angle_eng = V * Scalar(1.0 / 3.0);

        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // symmetrized version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // only apply force to local atoms
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += fab[0];
            h_force.data[idx_a].y += fab[1];
            h_force.data[idx_a].z += fab[2];
            h_force.data[idx_a].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += angle_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= fab[0] + fcb[0];
            h_force.data[idx_b].y -= fab[1] + fcb[1];
            h_force.data[idx_b].z -= fab[2] + fcb[2];
            h_force.data[idx_b].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += angle_virial[j];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += fcb[0];
            h_force.data[idx_c].y += fcb[1];
            h_force.data[idx_c].z += fcb[2];
            h_force.data[idx_c].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += angle_virial[j];
            }
        }
    }

namespace detail
    {
//! Exports the TableAngleForceCompute class to python
void export_TableAngleForceCompute(pybind11::module& m)
    {
    pybind11::class_<TableAngleForceCompute, ForceCompute, std::shared_ptr<TableAngleForceCompute>>(
        m,
        "TableAngleForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def_property_readonly("width", &TableAngleForceCompute::getWidth)
        .def("setParams", &TableAngleForceCompute::setParamsPython)
        .def("getParams", &TableAngleForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
