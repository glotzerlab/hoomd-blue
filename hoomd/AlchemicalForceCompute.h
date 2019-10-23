// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jproc

#ifndef __ALCHEMICAL_FORCE_COMPUTE__
#define __ALCHEMICAL_FORCE_COMPUTE__

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

#endif // #ifndef __ALCHEMICAL_FORCE_COMPUTE__
