

#ifndef GSD_STATE_H
#define GSD_STATE_H

#include "hoomd/SharedSignal.h"
#include "hoomd/GSDDumpWriter.h"
#include "hoomd/GSDReader.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


template<typename T>
inline void _connectGSDSignal(T* obj, std::shared_ptr<GSDDumpWriter> writer, std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&T::slotWriteGSD, obj, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    obj->addSlot(pslot);
    }

class gsd_element
{

};

#endif
