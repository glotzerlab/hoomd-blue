// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/mpcd/ParticleData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

class VirtualCallback
    {
    public:
    //! Constructor
    /*!
     * \param pdata MPCD particle data
     */
    VirtualCallback(std::shared_ptr<mpcd::ParticleData> pdata) : m_pdata(pdata), m_signal(false)
        {
        if (m_pdata)
            m_pdata->getNumVirtualSignal().connect<VirtualCallback, &VirtualCallback::slot>(this);
        }

    //! Destructor
    ~VirtualCallback()
        {
        if (m_pdata)
            m_pdata->getNumVirtualSignal().disconnect<VirtualCallback, &VirtualCallback::slot>(
                this);
        }

    bool operator()() const
        {
        return m_signal;
        }

    //! Reset the callback signal
    void reset()
        {
        m_signal = false;
        }

    private:
    std::shared_ptr<mpcd::ParticleData> m_pdata; //!< MPCD particle data
    bool m_signal;                               //!< If true, callback has been done

    //! Set the callback signal to true
    void slot()
        {
        m_signal = true;
        }
    };

//! Basic test case for virtual particle functionality
UP_TEST(virtual_add_remove_test)
    {
    auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU);

    // default initialize an empty snapshot in the reference box
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(2.0);
    snap->particle_data.type_mapping.push_back("A");

    // 1 particle system
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
    snap->mpcd_data.velocity[0] = vec3<Scalar>(1.0, 2.0, 3.0);

    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();

    // one particle at first, no virtual particles
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getNGlobal(), 1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);

    // add some virtual particles, and assume a realloc happened
    pdata->addVirtualParticles(2);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getNGlobal(), 1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 2);

    UP_ASSERT(pdata->getPositions().getNumElements() >= 3);
    UP_ASSERT(pdata->getAltPositions().getNumElements() >= 3);
    UP_ASSERT(pdata->getVelocities().getNumElements() >= 3);
    UP_ASSERT(pdata->getAltVelocities().getNumElements() >= 3);
    UP_ASSERT(pdata->getTags().getNumElements() >= 3);
    UP_ASSERT(pdata->getAltTags().getNumElements() >= 3);

    // ensure virtual particles are popped off
    pdata->removeVirtualParticles();
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getNGlobal(), 1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);

    // add a signaling subscriber, and make sure it is notified of changes
    VirtualCallback cb(pdata);
    UP_ASSERT(!cb());

    pdata->addVirtualParticles(1);
    UP_ASSERT(cb());

    cb.reset();
    pdata->removeVirtualParticles();
    UP_ASSERT(cb());

    cb.reset();
    pdata->removeVirtualParticles();
    UP_ASSERT(!cb());
    }
