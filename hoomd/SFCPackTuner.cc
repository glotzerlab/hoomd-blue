// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SFCPackTuner.cc
    \brief Defines the SFCPackTuner class
*/

#include "SFCPackTuner.h"
#include "Communicator.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/*! \param sysdef System to perform sorts on
 */
SFCPackTuner::SFCPackTuner(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Trigger> trigger)
    : Tuner(sysdef, trigger), m_last_grid(0), m_last_dim(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing SFCPackTuner" << endl;

    // perform lots of sanity checks
    assert(m_pdata);

    m_sort_order.resize(m_pdata->getMaxN());
    m_particle_bins.resize(m_pdata->getMaxN());

    // set the default grid
    // Grid dimension must always be a power of 2 and determines the memory usage for
    // m_traversal_order To prevent massive overruns of the memory, always use 256 for 3d and 4096
    // for 2d
    if (m_sysdef->getNDimensions() == 2)
        m_grid = 4096;
    else
        m_grid = 256;

    // register reallocate method with particle data maximum particle number change signal
    m_pdata->getMaxParticleNumberChangeSignal().connect<SFCPackTuner, &SFCPackTuner::reallocate>(
        this);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

/*! reallocate the internal arrays
 */
void SFCPackTuner::reallocate()
    {
    m_sort_order.resize(m_pdata->getMaxN());
    m_particle_bins.resize(m_pdata->getMaxN());
    }

/*! Destructor
 */
SFCPackTuner::~SFCPackTuner()
    {
    m_exec_conf->msg->notice(5) << "Destroying SFCPackTuner" << endl;
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<SFCPackTuner, &SFCPackTuner::reallocate>(
        this);
    }

/*! Performs the sort.
    \note In an updater list, this sort should be done first, before anyone else
    gets ahold of the particle data

    \param timestep Current timestep of the simulation
 */
void SFCPackTuner::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(6) << "SFCPackTuner: particle sort" << std::endl;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // make sure all particles that need to be local are
        m_comm->forceMigrate();
        m_comm->communicate(timestep);

        // remove all ghost particles
        m_pdata->removeAllGhostParticles();
        }
#endif

    // figure out the sort order we need to apply
    if (m_sysdef->getNDimensions() == 2)
        getSortedOrder2D();
    else
        getSortedOrder3D();

    // apply that sort order to the particles
    applySortOrder();

    // trigger sort signal (this also forces particle migration)
    m_pdata->notifyParticleSort();

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // restore ghosts
        m_comm->communicate(timestep);
        }
#endif
    }

void SFCPackTuner::applySortOrder()
    {
    assert(m_pdata);
    assert(m_sort_order.size() >= m_pdata->getN());
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                     access_location::host,
                                     access_mode::readwrite);

    // construct a temporary holding array for the sorted data
    Scalar4* scal4_tmp = new Scalar4[m_pdata->getN()];

    // sort positions and types
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal4_tmp[i] = h_pos.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_pos.data[i] = scal4_tmp[i];

    // sort velocities and mass
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal4_tmp[i] = h_vel.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_vel.data[i] = scal4_tmp[i];

    Scalar3* scal3_tmp = new Scalar3[m_pdata->getN()];
    // sort accelerations
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal3_tmp[i] = h_accel.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_accel.data[i] = scal3_tmp[i];

    Scalar* scal_tmp = new Scalar[m_pdata->getN()];
    // sort charge
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal_tmp[i] = h_charge.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_charge.data[i] = scal_tmp[i];

    // sort diameter
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal_tmp[i] = h_diameter.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_diameter.data[i] = scal_tmp[i];

    // sort angular momentum
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal4_tmp[i] = h_angmom.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_angmom.data[i] = scal4_tmp[i];

    // sort moment of inertia
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        scal3_tmp[i] = h_inertia.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_inertia.data[i] = scal3_tmp[i];
        }

        // in case anyone access it from frame to frame, sort the net virial
        {
        ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(),
                                         access_location::host,
                                         access_mode::readwrite);
        size_t virial_pitch = m_pdata->getNetVirial().getPitch();

        for (unsigned int j = 0; j < 6; j++)
            {
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal_tmp[i] = h_net_virial.data[j * virial_pitch + m_sort_order[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_net_virial.data[j * virial_pitch + i] = scal_tmp[i];
            }
        }

        // sort net force, net torque, and orientation
        {
        ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                         access_location::host,
                                         access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            scal4_tmp[i] = h_net_force.data[m_sort_order[i]];
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            h_net_force.data[i] = scal4_tmp[i];
        }

        {
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::host,
                                          access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            scal4_tmp[i] = h_net_torque.data[m_sort_order[i]];
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            h_net_torque.data[i] = scal4_tmp[i];
        }

        {
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            scal4_tmp[i] = h_orientation.data[m_sort_order[i]];
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            h_orientation.data[i] = scal4_tmp[i];
        }

    // sort image
    int3* int3_tmp = new int3[m_pdata->getN()];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        int3_tmp[i] = h_image.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_image.data[i] = int3_tmp[i];

    // sort body
    unsigned int* uint_tmp = new unsigned int[m_pdata->getN()];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        uint_tmp[i] = h_body.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_body.data[i] = uint_tmp[i];

    // sort global tag
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        uint_tmp[i] = h_tag.data[m_sort_order[i]];
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        h_tag.data[i] = uint_tmp[i];

    // rebuild global rtag
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_rtag.data[h_tag.data[i]] = i;
        }

    delete[] scal_tmp;
    delete[] scal4_tmp;
    delete[] scal3_tmp;
    delete[] uint_tmp;
    delete[] int3_tmp;
    }

namespace detail
    {
//! x walking table for the hilbert curve
static int istep[] = {0, 0, 0, 0, 1, 1, 1, 1};
//! y walking table for the hilbert curve
static int jstep[] = {0, 0, 1, 1, 1, 1, 0, 0};
//! z walking table for the hilbert curve
static int kstep[] = {0, 1, 1, 0, 0, 1, 1, 0};

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 1
    \param in Input sequence
*/
static void permute1(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[3];
    result[2] = in[4];
    result[3] = in[7];
    result[4] = in[6];
    result[5] = in[5];
    result[6] = in[2];
    result[7] = in[1];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 2
    \param in Input sequence
*/
static void permute2(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[7];
    result[2] = in[6];
    result[3] = in[1];
    result[4] = in[2];
    result[5] = in[5];
    result[6] = in[4];
    result[7] = in[3];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 3
    \param in Input sequence
*/
static void permute3(unsigned int result[8], const unsigned int in[8])
    {
    permute2(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 4
    \param in Input sequence
*/
static void permute4(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[2];
    result[1] = in[3];
    result[2] = in[0];
    result[3] = in[1];
    result[4] = in[6];
    result[5] = in[7];
    result[6] = in[4];
    result[7] = in[5];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 5
    \param in Input sequence
*/
static void permute5(unsigned int result[8], const unsigned int in[8])
    {
    permute4(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 6
    \param in Input sequence
*/
static void permute6(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[4];
    result[1] = in[3];
    result[2] = in[2];
    result[3] = in[5];
    result[4] = in[6];
    result[5] = in[1];
    result[6] = in[0];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 7
    \param in Input sequence
*/
static void permute7(unsigned int result[8], const unsigned int in[8])
    {
    permute6(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 8
    \param in Input sequence
*/
static void permute8(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[6];
    result[1] = in[5];
    result[2] = in[2];
    result[3] = in[1];
    result[4] = in[0];
    result[5] = in[3];
    result[6] = in[4];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule \a p-1
    \param in Input sequence
    \param p permutation rule to apply
*/
void permute(unsigned int result[8], const unsigned int in[8], int p)
    {
    switch (p)
        {
    case 0:
        permute1(result, in);
        break;
    case 1:
        permute2(result, in);
        break;
    case 2:
        permute3(result, in);
        break;
    case 3:
        permute4(result, in);
        break;
    case 4:
        permute5(result, in);
        break;
    case 5:
        permute6(result, in);
        break;
    case 6:
        permute7(result, in);
        break;
    case 7:
        permute8(result, in);
        break;
    default:
        assert(false);
        }
    }

    } // end namespace detail

//! recursive function for generating hilbert curve traversal order
/*! \param i Current x coordinate in grid
    \param j Current y coordinate in grid
    \param k Current z coordinate in grid
    \param w Number of grid cells wide at the current recursion level
    \param Mx Width of the entire grid (it is cubic, same width in all 3 directions)
    \param cell_order Current permutation order to traverse cells along
    \param traversal_order Traversal order to build up
    \pre \a traversal_order.size() == 0
    \pre Initial call should be with \a i = \a j = \a k = 0, \a w = \a Mx, \a cell_order =
   (0,1,2,3,4,5,6,7,8) \post traversal order contains the grid index (i*Mx*Mx + j*Mx + k) of each
   grid point listed in the order of the hilbert curve
*/
void SFCPackTuner::generateTraversalOrder(int i,
                                          int j,
                                          int k,
                                          int w,
                                          int Mx,
                                          unsigned int cell_order[8],
                                          vector<unsigned int>& traversal_order)
    {
    if (w == 1)
        {
        // handle base case
        traversal_order.push_back(i * Mx * Mx + j * Mx + k);
        }
    else
        {
        // handle arbitrary case, split the box into 8 sub boxes
        w = w / 2;

        // we ned to handle each sub box in the order defined by cell order
        for (int m = 0; m < 8; m++)
            {
            unsigned int cur_cell = cell_order[m];
            int ic = i + w * detail::istep[cur_cell];
            int jc = j + w * detail::jstep[cur_cell];
            int kc = k + w * detail::kstep[cur_cell];

            unsigned int child_cell_order[8];
            detail::permute(child_cell_order, cell_order, m);
            generateTraversalOrder(ic, jc, kc, w, Mx, child_cell_order, traversal_order);
            }
        }
    }

void SFCPackTuner::getSortedOrder2D()
    {
    // start by checking the saneness of some member variables
    assert(m_pdata);
    assert(m_sort_order.size() >= m_pdata->getN());

    // make even bin dimensions
    const BoxDim& box = m_pdata->getBox();

        // put the particles in the bins
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);

        // for each particle
        for (unsigned int n = 0; n < m_pdata->getN(); n++)
            {
            // find the bin each particle belongs in
            Scalar3 p = make_scalar3(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z);
            Scalar3 f = box.makeFraction(p, make_scalar3(0.0, 0.0, 0.0));
            int ib = (unsigned int)(f.x * m_grid) % m_grid;
            int jb = (unsigned int)(f.y * m_grid) % m_grid;

            // if the particle is slightly outside, move back into grid
            if (ib < 0)
                ib = 0;
            if (ib >= (int)m_grid)
                ib = m_grid - 1;

            if (jb < 0)
                jb = 0;
            if (jb >= (int)m_grid)
                jb = m_grid - 1;

            // record its bin
            unsigned int bin = ib * m_grid + jb;

            m_particle_bins[n] = std::pair<unsigned int, unsigned int>(bin, n);
            }
        }

    // sort the tuples
    sort(m_particle_bins.begin(), m_particle_bins.begin() + m_pdata->getN());

    // translate the sorted order
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        m_sort_order[j] = m_particle_bins[j].second;
        }
    }

void SFCPackTuner::getSortedOrder3D()
    {
    // start by checking the saneness of some member variables
    assert(m_pdata);
    assert(m_sort_order.size() >= m_pdata->getN());

    // make even bin dimensions
    const BoxDim& box = m_pdata->getBox();

    // reallocate memory arrays if m_grid changed
    // also regenerate the traversal order
    if (m_last_grid != m_grid || m_last_dim != 3)
        {
        if (m_grid > 256)
            {
            unsigned int mb = m_grid * m_grid * m_grid * 4 / 1024 / 1024;
            m_exec_conf->msg->warning()
                << "sorter is about to allocate a very large amount of memory (" << mb << "MB)"
                << " and may crash." << endl;
            m_exec_conf->msg->warning() << "            Reduce the amount of memory allocated to "
                                           "prevent this by decreasing the "
                                        << endl;
            m_exec_conf->msg->warning() << "            grid dimension (i.e. "
                                           "sorter.set_params(grid=128) ) or by disabling it "
                                        << endl;
            m_exec_conf->msg->warning()
                << "            ( sorter.disable() ) before beginning the run()." << endl;
            }

        // generate the traversal order
        GPUArray<unsigned int> traversal_order(m_grid * m_grid * m_grid, m_exec_conf);
        m_traversal_order.swap(traversal_order);

        vector<unsigned int> reverse_order(m_grid * m_grid * m_grid);
        reverse_order.clear();

        // we need to start the hilbert curve with a seed order 0,1,2,3,4,5,6,7
        unsigned int cell_order[8];
        for (unsigned int i = 0; i < 8; i++)
            cell_order[i] = i;
        generateTraversalOrder(0, 0, 0, m_grid, m_grid, cell_order, reverse_order);

        // access traversal order
        ArrayHandle<unsigned int> h_traversal_order(m_traversal_order,
                                                    access_location::host,
                                                    access_mode::overwrite);

        for (unsigned int i = 0; i < m_grid * m_grid * m_grid; i++)
            h_traversal_order.data[reverse_order[i]] = i;

        // write the traversal order out to a file for testing/presentations
        // writeTraversalOrder("hilbert.mol2", reverse_order);

        m_last_grid = m_grid;
        // store the last system dimension computed so we can be mindful if that ever changes
        m_last_dim = m_sysdef->getNDimensions();
        }

    // sanity checks
    assert(m_particle_bins.size() >= m_pdata->getN());
    assert(m_traversal_order.getNumElements() == m_grid * m_grid * m_grid);

    // put the particles in the bins
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // access traversal order
    ArrayHandle<unsigned int> h_traversal_order(m_traversal_order,
                                                access_location::host,
                                                access_mode::read);

    // for each particle
    for (unsigned int n = 0; n < m_pdata->getN(); n++)
        {
        Scalar3 p = make_scalar3(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z);
        Scalar3 f = box.makeFraction(p, make_scalar3(0.0, 0.0, 0.0));
        int ib = (unsigned int)(f.x * m_grid) % m_grid;
        int jb = (unsigned int)(f.y * m_grid) % m_grid;
        int kb = (unsigned int)(f.z * m_grid) % m_grid;

        // if the particle is slightly outside, move back into grid
        if (ib < 0)
            ib = 0;
        if (ib >= (int)m_grid)
            ib = m_grid - 1;

        if (jb < 0)
            jb = 0;
        if (jb >= (int)m_grid)
            jb = m_grid - 1;

        if (kb < 0)
            kb = 0;
        if (kb >= (int)m_grid)
            kb = m_grid - 1;

        // record its bin
        unsigned int bin = ib * (m_grid * m_grid) + jb * m_grid + kb;

        m_particle_bins[n] = std::pair<unsigned int, unsigned int>(h_traversal_order.data[bin], n);
        }

    // sort the tuples
    sort(m_particle_bins.begin(), m_particle_bins.begin() + m_pdata->getN());

    // translate the sorted order
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        m_sort_order[j] = m_particle_bins[j].second;
        }
    }

void SFCPackTuner::writeTraversalOrder(const std::string& fname,
                                       const vector<unsigned int>& reverse_order)
    {
    m_exec_conf->msg->notice(2) << "sorter: Writing space filling curve traversal order to "
                                << fname << endl;
    ofstream f(fname.c_str());
    f << "@<TRIPOS>MOLECULE" << endl;
    f << "Generated by HOOMD" << endl;
    f << m_traversal_order.getNumElements() << " " << m_traversal_order.getNumElements() - 1
      << endl;
    f << "NO_CHARGES" << endl;

    f << "@<TRIPOS>ATOM" << endl;
    m_exec_conf->msg->notice(2) << "sorter: Writing " << m_grid << "^3 grid cells" << endl;

    for (unsigned int i = 0; i < reverse_order.size(); i++)
        {
        unsigned int idx = reverse_order[i];
        unsigned int ib = idx / (m_grid * m_grid);
        unsigned int jb = (idx - ib * m_grid * m_grid) / m_grid;
        unsigned int kb = (idx - ib * m_grid * m_grid - jb * m_grid);

        f << i + 1 << " B " << ib << " " << jb << " " << kb << " " << "B" << endl;
        idx++;
        }

    f << "@<TRIPOS>BOND" << endl;
    for (unsigned int i = 0; i < m_traversal_order.getNumElements() - 1; i++)
        {
        f << i + 1 << " " << i + 1 << " " << i + 2 << " 1" << endl;
        }
    }

namespace detail
    {
void export_SFCPackTuner(pybind11::module& m)
    {
    pybind11::class_<SFCPackTuner, Tuner, std::shared_ptr<SFCPackTuner>>(m, "SFCPackTuner")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def_property("grid", &SFCPackTuner::getGrid, &SFCPackTuner::setGridPython);
    }

    } // end namespace detail

    } // end namespace hoomd
