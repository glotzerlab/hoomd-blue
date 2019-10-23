#ifndef _PATCH_ENERGY_JIT_H_
#define _PATCH_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include "EvalFactory.h"


//! Evaluate patch energies via runtime generated code
/*! This class enables the widest possible use-cases of patch energies in HPMC with low energy barriers for users to add
    custom interactions that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a pair of particles. The actual computation is performed by code that is loaded and compiled at
    run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    This is the first use of LLVM in HOOMD and it is experimental. As additional areas are identified as
    useful applications of LLVM, we will want to factor out some of the comment elements of this code
    into a generic LLVM module class. (i.e. handle broadcasting the string and compiling it in one place,
    with specific implementations requesting the function pointers they need).

    LLVM execution is managed with the KaleidoscopeJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. KaleidoscopeJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. KaleidoscopeJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. PatchEnergyJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
class PYBIND11_EXPORT PatchEnergyJIT : public hpmc::PatchEnergy
    {
    public:
        //! Constructor
        PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut,
                       const unsigned int array_size);

        //! Get the maximum r_ij radius beyond which energies are always 0
        virtual Scalar getRCut()
            {
            return m_r_cut;
            }

        //! Get the maximum r_ij radius beyond which energies are always 0
        virtual inline Scalar getAdditiveCutoff(unsigned int type)
            {
            // this potential corresponds to a point particle
            return 0.0;
            }

        //! evaluate the energy of the patch interaction
        /*! \param r_ij Vector pointing from particle i to j
            \param type_i Integer type index of particle i
            \param d_i Diameter of particle i
            \param charge_i Charge of particle i
            \param q_i Orientation quaternion of particle i
            \param type_j Integer type index of particle j
            \param q_j Orientation quaternion of particle j
            \param d_j Diameter of particle j
            \param charge_j Charge of particle j
            \returns Energy of the patch interaction.
        */
        virtual float energy(const vec3<float>& r_ij,
            unsigned int type_i,
            const quat<float>& q_i,
            float d_i,
            float charge_i,
            unsigned int type_j,
            const quat<float>& q_j,
            float d_j,
            float charge_j)
            {
            return m_eval(r_ij, type_i, q_i, d_i, charge_i, type_j, q_j, d_j, charge_j);
            }

        static pybind11::object getAlphaNP(pybind11::object self)
            {
            auto self_cpp = self.cast<PatchEnergyJIT *>();
            return pybind11::array(self_cpp->m_alpha_size, (float*)&self_cpp->m_alpha[0], self);
            }

    protected:
        //! function pointer signature
        typedef float (*EvalFnPtr)(const vec3<float>& r_ij, unsigned int type_i, const quat<float>& q_i, float, float, unsigned int type_j, const quat<float>& q_j, float, float);
        Scalar m_r_cut;                             //!< Cutoff radius
        std::shared_ptr<EvalFactory> m_factory;       //!< The factory for the evaluator function
        EvalFactory::EvalFnPtr m_eval;                //!< Pointer to evaluator function inside the JIT module
        float * m_alpha;                            //!< Array containing adjustable elements
        unsigned int m_alpha_size;                  //!< Size of array
    };

//! Exports the PatchEnergyJIT class to python
void export_PatchEnergyJIT(pybind11::module &m);
#endif // _PATCH_ENERGY_JIT_H_
