#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"

#include "OrcLazyJIT.h"

class PatchEnergyJIT
    {
    public:
        PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& fname, Scalar r_cut);

        Scalar getRCut()
            {
            return m_r_cut;
            }

        float energy(unsigned int type_i, const quat<float>& orientation_i, const vec3<float>& pos_j, unsigned int type_j, const quat<float>& orientation_j)
            {
            return m_eval(type_i, orientation_i, pos_j, type_j, orientation_j);
            }

    private:
        typedef float (*EvalFnPtr)(unsigned int type_i, const quat<float>& orientation_i, const vec3<float>& pos_j, unsigned int type_j, const quat<float>& orientation_j);

        Scalar m_r_cut;
        std::shared_ptr<llvm::OrcLazyJIT> m_JIT;
        EvalFnPtr m_eval;
    };

//! Exports the PatchEnergyJIT class to python
void export_PatchEnergyJIT(pybind11::module &m);
