#include <pybind11/pybind11.h>

#include "Tuner.h"

class PYBIND11_EXPORT PythonTuner : public Tuner
{
    public:
        PythonTuner(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<Trigger> trigger,
                      pybind11::object tuner);

        void update(unsigned int timestep);

        PDataFlags getRequestedPDataFlags();

        void setTuner(pybind11::object tuner);

        pybind11::object getTuner() {return m_tuner;}

    protected:
        pybind11::object m_tuner;
        PDataFlags m_flags;
};

void export_PythonTuner(pybind11::module& m);
