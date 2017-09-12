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
        float energy(float x, float y, float z)
            {
            return m_eval(vec3<float>(x, y, z));
            }
    private:
        typedef float (*EvalFnPtr)(const vec3<float>& v);

        Scalar m_r_cut;
        std::shared_ptr<llvm::OrcLazyJIT> m_JIT;
        EvalFnPtr m_eval;
    };

//! Exports the PatchEnergyJIT class to python
void export_PatchEnergyJIT(pybind11::module &m);
