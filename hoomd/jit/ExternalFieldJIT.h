#ifndef _EXTERNAL_FIELD_ENERGY_JIT_H_
#define _EXTERNAL_FIELD_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/ExternalField.h"
#include "hoomd/BoxDim.h"

#include "ExternalFieldEvalFactory.h"

#define EXTERNAL_FIELD_JIT_LOG_NAME           "jit_energy"

//! Evaluate external field forces via runtime generated code
/*! This class enables the widest possible use-cases of external fields in HPMC with low energy barriers for users to add
    custom forces that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a particle and an external field. The actual computation is performed by code that is loaded and
    compiled at run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    LLVM execution is managed with the KaleidoscopeJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. KaleidoscopeJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. KaleidoscopeJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. ExternalFieldJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
template< class Shape>
class ExternalFieldJIT : public hpmc::ExternalFieldMono<Shape>
    {
    public:
        //! Constructor
        ExternalFieldJIT(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir) : hpmc::ExternalFieldMono<Shape>(sysdef)
            {
            // build the JIT.
            m_factory = std::shared_ptr<ExternalFieldEvalFactory>(new ExternalFieldEvalFactory(llvm_ir));

            // get the evaluator
            m_eval = m_factory->getEval();

            if (!m_eval)
                {
                exec_conf->msg->error() << m_factory->getError() << std::endl;
                throw std::runtime_error("Error compiling JIT code.");
                }
            }

        //! Evaluate the energy of the force.
        /*! \param box The system box.
            \param type Particle type.
            \param r_i Particle position
            \param q_i Particle orientation.
            \param diameter Particle diameter.
            \param charge Particle charge.
            \returns Energy due to the force
        */
        virtual float energy(const BoxDim& box,
            unsigned int type,
            const vec3<Scalar>& r_i,
            const quat<Scalar>& q_i,
            Scalar diameter,
            Scalar charge
            )
            {
            return m_eval(box, type, r_i, q_i, diameter, charge);
            }

        virtual double calculateDeltaE(const Scalar4 * const position_old_arg,
                                       const Scalar4 * const orientation_old_arg,
                                       const BoxDim * const box_old_arg
                                       )
            {
            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(), access_location::host, access_mode::read);

            const BoxDim& box_new = this->m_pdata->getGlobalBox();
            const Scalar4 * position_old = position_old_arg, * orientation_old = orientation_old_arg;
            const BoxDim * box_old = box_old_arg;

            if( !position_old )
                {
                const Scalar4 * const position_new = h_postype.data;
                position_old = position_new;
                }
            if( !orientation_old )
                {
                const Scalar4 * const orientation_new = h_orientation.data;
                orientation_old = orientation_new;
                }
            if( !box_old )
                {
                box_old = &box_new;
                }

            double dE = 0.0;
            for(size_t i = 0; i < this->m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                unsigned int typ_i = __scalar_as_int(postype_i.w);
                vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
                
                dE += energy(box_new, typ_i, pos_i, quat<Scalar>(h_orientation.data[i]), h_diameter.data[i], h_charge.data[i]);
                dE -= energy(*box_old, typ_i, vec3<Scalar>(*(position_old+i)), quat<Scalar>(*(orientation_old+i)), h_diameter.data[i], h_charge.data[i]);
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(MPI_IN_PLACE, &dE, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
                }
            #endif

            return dE;
            }

        //! method to calculate the energy difference for the proposed move.
        double energydiff(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
            Scalar4 postype_i = h_postype.data[index];
            unsigned int typ_i = __scalar_as_int(postype_i.w);

            ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(), access_location::host, access_mode::read);

            double dE = 0.0;
            dE += energy(this->m_pdata->getGlobalBox(), typ_i, position_new, shape_new.orientation, h_diameter.data[index], h_charge.data[index]);
            dE -= energy(this->m_pdata->getGlobalBox(), typ_i, position_old, shape_old.orientation, h_diameter.data[index], h_charge.data[index]);
            return dE;
            }

        //! Returns a list of log quantities this compute calculates
        std::vector< std::string > getProvidedLogQuantities()
            {
            std::vector<std::string> provided_quantities;
            provided_quantities.push_back(std::string("external_field_jit"));
            return provided_quantities;
            }

        //! Calculates the requested log value and returns it
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if ( quantity == "external_field_jit" )
                {
                ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
                ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(), access_location::host, access_mode::read);
                ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(), access_location::host, access_mode::read);

                const BoxDim& box = this->m_pdata->getGlobalBox();

                double dE = 0.0;
                for(size_t i = 0; i < this->m_pdata->getN(); i++)
                    {
                    // read in the current position and orientation
                    Scalar4 postype_i = h_postype.data[i];
                    unsigned int typ_i = __scalar_as_int(postype_i.w);
                    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

                    dE += energy(box, typ_i, pos_i, quat<Scalar>(h_orientation.data[i]), h_diameter.data[i], h_charge.data[i]);
                    }

                return dE;
                }
            else
                {
                this->m_exec_conf->msg->error() << "jit.external.user: " << quantity << " is not a valid log quantity" << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }

    protected:
        //! function pointer signature
        typedef float (*ExternalFieldEvalFnPtr)(const BoxDim& box, unsigned int type, const vec3<Scalar>& r_i, const quat<Scalar>& q_i, Scalar diameter, Scalar charge);
        std::shared_ptr<ExternalFieldEvalFactory> m_factory;       //!< The factory for the evaluator function
        ExternalFieldEvalFactory::ExternalFieldEvalFnPtr m_eval;                //!< Pointer to evaluator function inside the JIT module

    };

//! Exports the ExternalFieldJIT class to python
template< class Shape>
void export_ExternalFieldJIT(pybind11::module &m, std::string name)
    {
    pybind11::class_<ExternalFieldJIT<Shape>, std::shared_ptr<ExternalFieldJIT<Shape> > >(m, name.c_str(), pybind11::base< hpmc::ExternalFieldMono <Shape> >())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, 
                                 std::shared_ptr<ExecutionConfiguration>,
                                 const std::string& >())
            .def("energy", &ExternalFieldJIT<Shape>::energy);
    }
#endif // _EXTERNAL_FIELD_ENERGY_JIT_H_
