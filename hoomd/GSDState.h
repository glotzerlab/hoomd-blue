

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

// T corres
template<typename T, typename G>
class gsd_pack
    {
    public:
        typedef T data_type;
        typedef G gsd_data_type;

        gsd_pack(const T& data, unsigned int N, gsd_type typ) : m_data(data), m_N(N), m_type(type) {}

        virtual G * pack() { return (G *) &m_data; }

        virtual unsigned int N() const { return m_N; }

        virtual unsigned int type() const { return m_type; }
    protected:
        std::vector<T>  m_buffer;
        const T&        m_data;
        unsigned int    m_N;
        gsd_type        m_type;
    };

template<typename T, typename R>
class gsd_unpack
    {
    public:
        gsd_unpack( unsigned int N, gsd_type typ, unsigned int buffer_size) : m_data(data), m_N(N), m_type(type) { m_buffer.resize(buffer_size); }

        virtual void unpack()
            {
            for(unsigned int i = 0; i < m_N; i++)
                {
                m_data[i] = m_buffer[i];
                }
            }

        virtual unsigned int N() const { return m_N; }

        virtual unsigned int type() const { return m_type; }

        virtual unsigned int bytes() const { return m_buffer.size()*gsd_sizeof_type(m_type); }

        virtual void * buffer() {return &m_buffer[0]; }
    protected:
        std::vector<T>  m_buffer;
        R&              m_data
        unsigned int    m_N;
        gsd_type        m_type;
    };

class gsd_chunk
    {
    public:
        gsd_chunk(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : m_exec_conf(exec_conf), m_mpi(mpi) {}

        template<class pack>
        int write(gsd_handle& handle, const std::string& name, pack packer) // unsigned int Ntypes, const T& data, gsd_type type,
            {
            if(!m_exec_conf->isRoot())
                return 0;

            int retval = 0;
            retval |= gsd_write_chunk(&handle, name.c_str(), packer.type(), packer.N(), 1, 0, (void *)packer.pack());
            return retval;
            }

        template<class unpack>
        bool read(  std::shared_ptr<GSDReader> reader,
                    uint64_t frame,
                    const std::string& name,
                    typename unpack::data_type& data,
                    unpack unpacker) // unsigned int Ntypes, T& data, gsd_type type
            {
            bool success = true;
            if(m_exec_conf->isRoot())
                {
                success = reader->readChunk((void *) unpacker.buffer(), frame, name.c_str(), unpacker.bytes(), unpacker.N()) && success;
                }
        #ifdef ENABLE_MPI
            if(m_mpi)
                {
                bcast(d, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                }
        #endif
            if(!d.size())
                throw std::runtime_error("Error occured while attempting to restore from gsd file.");
            unpacker.unpack(data);
            return success;
            }

    protected:
        const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
        bool m_mpi;
    };


#endif
