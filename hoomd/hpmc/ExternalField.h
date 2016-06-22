// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _EXTERNAL_FIELD_H_
#define _EXTERNAL_FIELD_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/


#include "hoomd/Compute.h"
#include "hoomd/extern/saruprng.h" // not sure if we need this for the accept method
#include "hoomd/VectorMath.h"

#include "HPMCCounters.h"   // do we need this to keep track of the statistics?

namespace hpmc
{
class ExternalField : public Compute
    {
    public:
        ExternalField(std::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef) {}
        /*! calculateBoltzmannWeight(unsigned int timestep)
            method used to calculate the boltzmann weight contribution for the
            external field of the entire system. This is used to interface with
            global moves such as the NPT box updater. The boltzmann factor can
            be calculated by the quotient of the boltzmann weight post move and
            boltzmann weight pre move. Example:
            Scalar bw1 = external->calculateBoltzmannWeight(timestep);
            // make a move updating m_pdata and any other system info (shape, position, orientation, etc.)
            Scalar bw2 = external->calculateBoltzmannWeight(timestep);
            pacc = min(1, bw2/bw1);
        */
        virtual Scalar calculateBoltzmannWeight(unsigned int timestep) = 0;

        virtual Scalar calculateBoltzmannFactor(const Scalar4 * const  position_old,
                                                const Scalar4 * const  orientation_old,
                                                const BoxDim * const  box_old
                                            )  = 0;

        virtual bool hasVolume() {return false;}

        virtual Scalar getVolume() {return 0;}
    };
//! Compute that accepts or rejects moves accoding to some external field
/*! **Overview** <br>

    \ingroup hpmc_computes
*/
template< class Shape >
class ExternalFieldMono : public ExternalField
    {
    public:
        ExternalFieldMono(std::shared_ptr<SystemDefinition> sysdef) : ExternalField(sysdef) {}

        ~ExternalFieldMono(){}

        //! needed for Compute. currently not used.
        virtual void compute(unsigned int timestep) {}

        //! method to accept or reject the proposed move used by the integrator.
        virtual bool accept(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new, Saru& rng) = 0;

        //! method to calculate the boltzmann factor for the proposed move.
        virtual Scalar boltzmann(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new) = 0;

        virtual void reset(unsigned int timestep) {}

    protected:
        /* Nothing yet */
    };

//! Wrapper class for wrapping pure virtual methods
template<class Shape>
class ExternalFieldMonoWrap : public ExternalFieldMono<Shape>, public boost::python::wrapper< ExternalFieldMono<Shape> >
    {
    public:
        //! Constructor
        ExternalFieldMonoWrap(std::shared_ptr<SystemDefinition> sysdef) :  ExternalFieldMono<Shape>(sysdef) { }
        //! wrap the abstract method.
        void compute(unsigned int timestep)
            {
            this->get_override("compute")(timestep);
            }
        bool accept(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new, Saru& rng)
            {
            return this->get_override("accept")(index, position_old, shape_old, position_new, shape_new, rng);
            }
        Scalar boltzmann(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            return this->get_override("boltzmann")(index, position_old, shape_old, position_new, shape_new);
            }
        Scalar calculateBoltzmannWeight(unsigned int timestep)
            {
            return this->get_override("calculateBoltzmannWeight")(timestep);
            }
        Scalar calculateBoltzmannFactor(const Scalar4 * const  position_old,
                                        const Scalar4 * const  orientation_old,
                                        const BoxDim * const  box_old)
            {
            return this->get_override("calculateBoltzmannFactor")(position_old,
                                                    orientation_old,
                                                    box_old);
            }
    };

template<class Shape>
void export_ExternalFieldInterface(std::string name)
    {
    class_< ExternalFieldMonoWrap<Shape>, std::shared_ptr< ExternalFieldMonoWrap<Shape> > >
    ((name + "Interface").c_str(), init< std::shared_ptr<SystemDefinition> >());
    }

} // end namespace hpmc


#endif // end inclusion guard
