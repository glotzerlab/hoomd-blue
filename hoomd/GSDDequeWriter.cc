// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "GSDDequeWriter.h"
#include "hoomd/GSDDumpWriter.h"

namespace hoomd
    {
GSDDequeWriter::GSDDequeWriter(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<Trigger> trigger,
                               const std::string& fname,
                               std::shared_ptr<ParticleGroup> group,
                               int queue_size,
                               std::string mode)
    : GSDDumpWriter(sysdef, trigger, fname, group, mode), m_queue_size(queue_size)
    {
    }

void GSDDequeWriter::analyze(uint64_t timestep)
    {
    m_frame_queue.emplace_front();
    m_log_queue.push_front(getLogData());
    populateLocalFrame(m_frame_queue.front(), timestep);
    if (m_queue_size != -1 && m_frame_queue.size() > static_cast<size_t>(m_queue_size))
        {
        m_frame_queue.pop_back();
        m_log_queue.pop_back();
        }
    }

void GSDDequeWriter::dump()
    {
    for (auto i {static_cast<long int>(m_frame_queue.size()) - 1}; i >= 0; --i)
        {
        write(m_frame_queue[i], m_log_queue[i]);
        }
    m_frame_queue.clear();
    m_log_queue.clear();
    }

int GSDDequeWriter::getMaxQueueSize() const
    {
    return m_queue_size;
    }

void GSDDequeWriter::setMaxQueueSize(int new_max_size)
    {
    m_queue_size = new_max_size;
    if (m_queue_size == -1)
        {
        return;
        }
    while (static_cast<size_t>(m_queue_size) < m_frame_queue.size())
        {
        m_frame_queue.pop_back();
        m_log_queue.pop_back();
        }
    }

namespace detail
    {
void export_GSDDequeWriter(pybind11::module& m)
    {
    pybind11::class_<GSDDequeWriter, GSDDumpWriter, std::shared_ptr<GSDDequeWriter>>(
        m,
        "GSDDequeWriter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::string,
                            std::shared_ptr<ParticleGroup>,
                            int,
                            std::string>())
        .def_property("n_max_frames",
                      &GSDDequeWriter::getMaxQueueSize,
                      &GSDDequeWriter::setMaxQueueSize)
        .def("dump", &GSDDequeWriter::dump);
    }
    } // namespace detail
    } // namespace hoomd
