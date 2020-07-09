#include <pybind11/pybind11.h>

#include "Updater.h"

class PYBIND11_EXPORT PythonUpdater : public Updater
{
    public:
        PythonUpdater(std::shared_ptr<SystemDefinition> sysdef,
                      pybind11::object updater);

        void update(unsigned int timestep);

        PDataFlags getRequestedPDataFlags();

        void setUpdater(pybind11::object updater);

        pybind11::object getUpdater() {return m_updater;}

    protected:
        pybind11::object m_updater;
        PDataFlags m_flags;
};

void export_PythonUpdater(pybind11::module& m);
