// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jproc

#include "IntegrationMethodTwoStep.h"

#ifndef __ALCHEMOSTAT_TWO_STEP__
#define __ALCHEMOSTAT_TWO_STEP__

class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~AlchemostatTwoStep() {}

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF();

        virtual unsigned int getRotationalNDOF();

        virtual void randomizeVelocities(unsigned int timestep);

        //! Reinitialize the integration variables if needed (implemented in the actual subclasses)
        virtual void initializeIntegratorVariables() {}

    protected:
        std::shared_ptr<AlchParticles> m_alchParticles;    //!< A vector of all alchemical particles
        unsigned int m_nTStep;          //!< Trotter factorization power

// TODO: general templating possible for two step methods?

// Very similar to force compute, but only associated with one global "alchemical particle"
class AlchForceCompute: public Compute
    {
    public:
        //! Constructor
        AlchForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~AlchForceCompute();

        //! Store the timestep size TODO: necessary?
        virtual void setDeltaT(Scalar dt)
            {
            m_deltaT = dt;
            }
        
        #ifdef ENABLE_MPI
        //! Pre-compute the forces
        /*! This method is called in MPI simulations BEFORE the particles are migrated
         * and can be used to overlap computation with communication
         */
        virtual void preCompute(unsigned int timestep){}
        #endif
    
        //! Computes the forces
        virtual void compute(unsigned int timestep);

        //! Benchmark the force compute
        virtual double benchmark(unsigned int num_iters);

        //! Easy access to the force on a single particle
        // Scalar getForce(unsigned int tag);

        //! Get the array of computed forces
        GlobalArray<Scalar>& getForceArray()
            {
            return m_force;
            }
        protected:
                GlobalArray<Scalar> m_force;            //!< 
                // TODO: Make sure there's no future case where we'd need more information such as second derivatives which would not be per particle?
        
        //! Actually perform the computation of the forces
        /*! This is pure virtual here. Sub-classes must implement this function. It will be called by
            the base class compute() when the forces need to be computed.
            \param timestep Current time step
        */
        virtual void computeForces(unsigned int timestep){}
    }

// NOTE: Should maybe be a struct and part of system definition?
// TODO: AlchemyData and groups
class AlchParticles:
    {
    public:
        AlchParticles(std::shared_ptr<SystemDefinition> sysdef);
        // TODO: Mathods for adding and removing
        unsigned int getNum()
            {
            return m_size;
            }

    protected:
        std::vector<Scalar3> m_alchKineticVariables; //!< x,y,z are the velocity, net force, and mass of the alchemical particle
        GlobalVector<Scalar> m_alchValues; //!< position of the alchemical particle
        std::vector< std::vector< std::shared_ptr<AlchForceCompute> > > m_alchForces; //!<Per alchemical particle forces
        unsigned int m_size;
        // TODO: Figure out best way to link to a specific interaction property, would be best if possible to do for pairs, external, etc
    }

// TODO

#endif // #ifndef __ALCHEMOSTAT_TWO_STEP__
