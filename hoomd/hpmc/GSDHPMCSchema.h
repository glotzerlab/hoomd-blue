
#include "ShapeSphere.h" // check
#include "ShapeConvexPolygon.h" // check
#include "ShapeSpheropolygon.h" // check
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h" // check
#include "ShapeSpheropolyhedron.h" // check
#include "ShapeSimplePolygon.h" // check
#include "ShapeEllipsoid.h" // check
#include "ShapeFacetedEllipsoid.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"
#include "hoomd/extern/gsd.h"
#include "hoomd/GSDDumpWriter.h"
#include "hoomd/GSDReader.h"
#include "hoomd/HOOMDMPI.h"

#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifndef _GSD_HPMC_Schema_H_
#define _GSD_HPMC_Schema_H_

template<class T>
using param_array = typename std::vector<T, managed_allocator<T> >;

struct gsd_schema_hpmc_base
    {
    gsd_schema_hpmc_base(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : m_exec_conf(exec_conf), m_mpi(mpi) {}
    const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    bool m_mpi;
    };

struct gsd_schema_hpmc : public gsd_schema_hpmc_base
    {
    gsd_schema_hpmc(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}
    template<class T>
    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes, const T* const data, gsd_type type)
        {
        if(!m_exec_conf->isRoot())
            return 0;
        int retval = 0;
        retval |= gsd_write_chunk(&handle, name.c_str(), type, Ntypes, 1, 0, (void *)data);
        return retval;
        }

    template<class T>
    bool read(std::shared_ptr<GSDReader> reader, uint64_t frame, const std::string& name, unsigned int Ntypes, T* const data, gsd_type type)
        {
        bool success = true;
        std::vector<T> d;
        if(m_exec_conf->isRoot())
            {
            d.resize(Ntypes);
            success = reader->readChunk((void *) &d[0], frame, name.c_str(), Ntypes*gsd_sizeof_type(type), Ntypes) && success;
            }
    #ifdef ENABLE_MPI
        if(m_mpi)
            {
            bcast(d, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            }
    #endif
        if(!d.size())
            throw std::runtime_error("Error occurred while attempting to restore from gsd file.");
        for(unsigned int i = 0; i < Ntypes; i++)
            {
            data[i] = d[i];
            }
        return success;
        }
    };

template<class T>
struct gsd_shape_schema : public gsd_schema_hpmc_base
    {
    gsd_shape_schema(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle&, const std::string&, unsigned int, const param_array<T>& )
        {
        throw std::runtime_error("This is not implemented");
        return 0;
        }
    bool read(std::shared_ptr<GSDReader>, uint64_t, const std::string&, unsigned int, param_array<T>&)
        {
        throw std::runtime_error("This is not implemented");
        return false;
        }
    };

template<>
struct gsd_shape_schema<hpmc::sph_params>: public gsd_schema_hpmc_base
    {
    gsd_shape_schema(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes, const param_array<hpmc::sph_params>& shape)
        {
        if(!m_exec_conf->isRoot())
            return 0;
        int retval = 0;
        std::string path = name + "radius";
        std::string path_o = name + "orientable";
        std::vector<float> data(Ntypes);
        std::vector<uint8_t> orientableflag(Ntypes);
        std::transform(shape.begin(), shape.end(), data.begin(), [](const hpmc::sph_params& s)->float{return s.radius;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&data[0]);
        std::transform(shape.begin(), shape.end(), orientableflag.begin(), [](const hpmc::sph_params& s)->uint32_t{return s.isOriented;});
        retval |= gsd_write_chunk(&handle, path_o.c_str(), GSD_TYPE_UINT8, Ntypes, 1, 0, (void *)&orientableflag[0]);
        return retval;
        }

    void read(  std::shared_ptr<GSDReader> reader,
                uint64_t frame,
                const std::string& name,
                unsigned int Ntypes,
                param_array<hpmc::sph_params>& shape
            )
        {

        std::string path_o = name + "orientable";
        std::vector<float> data;
        std::string path = name + "radius";
        std::vector<uint8_t> orientableflag(Ntypes);
        bool state_read = true;
        if(m_exec_conf->isRoot())
            {
            data.resize(Ntypes, 0.0);
            orientableflag.resize(Ntypes);
            if(!reader->readChunk((void *) &data[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            if (reader->getHandle().header.schema_version <= gsd_make_version(1,2))
                {
                std::fill(orientableflag.begin(), orientableflag.end(), 0);
                }
            else if (!reader->readChunk((void *) &orientableflag[0], frame, path_o.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_UINT8), Ntypes))
                {
                state_read = false;
                }
            }

        #ifdef ENABLE_MPI
            if(m_mpi)
            {
            bcast(state_read, 0, m_exec_conf->getMPICommunicator());
            bcast(data, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            bcast(orientableflag, 0, m_exec_conf->getMPICommunicator());
            }
        #endif

        if (!state_read)
            throw std::runtime_error("Error occurred while attempting to restore from gsd file.");

        for(unsigned int i = 0; i < Ntypes; i++)
            {
            shape[i].radius = data[i];
            shape[i].isOriented = orientableflag[i];
            shape[i].ignore = 0;
            }
        }
    };

template<>
struct gsd_shape_schema<hpmc::ell_params>: public gsd_schema_hpmc_base
    {
    gsd_shape_schema(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes,const param_array<hpmc::ell_params>& shape)
        {
        if(!m_exec_conf->isRoot())
            return 0;

        int retval = 0;
        std::vector<float> data(Ntypes);
        std::string path = name + "a";
        std::transform(shape.cbegin(), shape.cend(), data.begin(), [](const hpmc::ell_params& s)->float{return s.x;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&data[0]);
        path = name + "b";
        std::transform(shape.cbegin(), shape.cend(), data.begin(), [](const hpmc::ell_params& s)->float{return s.y;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&data[0]);
        path = name + "c";
        std::transform(shape.cbegin(), shape.cend(), data.begin(), [](const hpmc::ell_params& s)->float{return s.z;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&data[0]);
        return retval;
        }

    void read(  std::shared_ptr<GSDReader> reader,
                uint64_t frame,
                const std::string& name,
                unsigned int Ntypes,
                param_array<hpmc::ell_params>& shape
            )
        {

        bool state_read = true;
        std::vector<float> a,b,c;
        if(m_exec_conf->isRoot())
            {
            a.resize(Ntypes),b.resize(Ntypes),c.resize(Ntypes);
            std::string path = name + "a";
            if (!reader->readChunk((void *)&a[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            path = name + "b";
            if (!reader->readChunk((void *)&b[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            path = name + "c";
            if (!reader->readChunk((void *)&c[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            }

        #ifdef ENABLE_MPI
            if(m_mpi)
                {
                bcast(state_read, 0, m_exec_conf->getMPICommunicator());
                bcast(a, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                bcast(b, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                bcast(c, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                }
        #endif

        if (!state_read)
            throw std::runtime_error("Error occurred while attempting to restore from gsd file.");

        for(unsigned int i = 0; i < Ntypes; i++)
            {
            shape[i].x = a[i];
            shape[i].y = b[i];
            shape[i].z = c[i];
            shape[i].ignore = 0;
            }
        }
    };

template<>
struct gsd_shape_schema< hpmc::detail::poly3d_verts > : public gsd_schema_hpmc_base
    {
    gsd_shape_schema(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes, const param_array<hpmc::detail::poly3d_verts>& shape)
        {
        if(!m_exec_conf->isRoot())
            return 0;

        std::string path;
        int retval = 0;
        std::vector<uint32_t> N(Ntypes);
        path = name + "N";
        std::transform(shape.cbegin(), shape.cend(), N.begin(), [](const hpmc::detail::poly3d_verts& s) -> uint32_t{return s.N;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_UINT32, Ntypes, 1, 0, (void *)&N[0]);
        path = name + "vertices";
        size_t count = std::accumulate(N.begin(), N.end(), 0);
        std::vector<float> data(count*Ntypes*3), sr(Ntypes);
        count = 0;
        for(unsigned int i = 0; i < Ntypes; i++)
            {
            for (unsigned int v = 0; v < shape[i].N; v++)
                {
                data[count*3+0] = float(shape[i].x[v]);
                data[count*3+1] = float(shape[i].y[v]);
                data[count*3+2] = float(shape[i].z[v]);
                count++;
                }
            }
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, count, 3, 0, (void *)&data[0]);
        path = name + "sweep_radius";
        std::transform(shape.cbegin(), shape.cend(), sr.begin(), [](const hpmc::detail::poly3d_verts& s) -> float{return s.sweep_radius;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&sr[0]);
        return retval;
        }

    void read(  std::shared_ptr<GSDReader> reader,
                uint64_t frame,
                const std::string& name,
                unsigned int Ntypes,
                param_array<hpmc::detail::poly3d_verts>& shape
            )
        {

        bool state_read = true;
        std::vector<float> vertices,sweep_radius;
        std::vector<uint32_t> N;
        uint32_t count = 0;
        std::string path;
        assert(shape.size() == Ntypes);
        if(m_exec_conf->isRoot())
            {
            N.resize(Ntypes);
            sweep_radius.resize(Ntypes);
            path = name + "N";
            if(!reader->readChunk((void *)&N[0], frame, path.c_str(),  Ntypes*gsd_sizeof_type(GSD_TYPE_UINT32), Ntypes))
                state_read = false;
            count = std::accumulate(N.begin(), N.end(), 0);
            vertices.resize(count*Ntypes*3);
            path = name + "vertices";
            if(!reader->readChunk((void *)&vertices[0], frame, path.c_str(), 3*count*gsd_sizeof_type(GSD_TYPE_FLOAT), count))
                state_read = false;
            path = name + "sweep_radius";
            if(!reader->readChunk((void *)&sweep_radius[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            }
    #ifdef ENABLE_MPI
        if(m_mpi)
            {
            bcast(state_read, 0, m_exec_conf->getMPICommunicator());
            bcast(N, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            bcast(vertices, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            bcast(sweep_radius, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            }
    #endif

        if (!state_read)
            throw std::runtime_error("Error occurred while attempting to restore from gsd file.");

        count = 0;
        for (unsigned int i = 0; i < Ntypes; i++)
            {
            float dsq = 0.0;
            hpmc::detail::poly3d_verts result(N[i], m_exec_conf->isCUDAEnabled());
            for (unsigned int v = 0; v < N[i]; v++)
                {
                result.x[v] = vertices[count*3+0];
                result.y[v] = vertices[count*3+1];
                result.z[v] = vertices[count*3+2];
                dsq = fmax(result.x[v]*result.x[v] + result.y[v]*result.y[v] + result.z[v]*result.z[v], dsq);
                count++;
                }
            result.diameter = 2.0*(sqrt(dsq)+result.sweep_radius);
            result.N = N[i];
            result.sweep_radius = sweep_radius[i];
            shape[i] = result; // Can we avoid a full copy of the data (move semantics?)
            shape[i].ignore = 0;
            }
        }
    };

template<>
struct gsd_shape_schema< hpmc::detail::poly2d_verts >: public gsd_schema_hpmc_base
    {
    gsd_shape_schema(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes, const param_array<hpmc::detail::poly2d_verts>& shape)
        {
        if(!m_exec_conf->isRoot())
            return 0;

        std::string path;
        int retval = 0;
        std::vector<uint32_t> N(Ntypes);
        path = name + "N";
        std::transform(shape.cbegin(), shape.cend(), N.begin(), [](const hpmc::detail::poly2d_verts& s) -> uint32_t{return s.N;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_UINT32, Ntypes, 1, 0, (void *)&N[0]);
        path = name + "vertices";
        std::vector<float> data(hpmc::detail::MAX_POLY2D_VERTS*Ntypes*2), sr(Ntypes); // over allocate is ok because we just wont write those extra ones
        uint32_t count = 0;
        for(unsigned int i = 0; i < Ntypes; i++)
            {
            for (unsigned int v = 0; v < shape[i].N; v++)
                {
                data[count*2+0] = float(shape[i].x[v]);
                data[count*2+1] = float(shape[i].y[v]);
                count++;
                }
            }
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, count, 2, 0, (void *)&data[0]);
        path = name + "sweep_radius";
        std::transform(shape.cbegin(), shape.cend(), sr.begin(), [](const hpmc::detail::poly2d_verts& s) -> float{return s.sweep_radius;});
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, Ntypes, 1, 0, (void *)&sr[0]);
        return retval;
        }

    void read(  std::shared_ptr<GSDReader> reader,
                uint64_t frame,
                const std::string& name,
                unsigned int Ntypes,
                param_array<hpmc::detail::poly2d_verts>& shape
            )
        {

        bool state_read = true;
        std::vector<float> vertices,sweep_radius;
        std::vector<uint32_t> N;
        uint32_t count = 0;
        std::string path;
        if(m_exec_conf->isRoot())
            {
            N.resize(Ntypes);
            vertices.resize(hpmc::detail::MAX_POLY2D_VERTS*Ntypes*2);
            sweep_radius.resize(Ntypes);
            path = name + "N";
            if(!reader->readChunk((void *)&N[0], frame, path.c_str(),  Ntypes*gsd_sizeof_type(GSD_TYPE_UINT32), Ntypes))
                state_read = false;
            count = std::accumulate(N.begin(), N.end(), 0);
            path = name + "vertices";
            if(!reader->readChunk((void *)&vertices[0], frame, path.c_str(), 2*count*gsd_sizeof_type(GSD_TYPE_FLOAT), count))
                state_read = false;
            path = name + "sweep_radius";
            if(!reader->readChunk((void *)&sweep_radius[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes))
                state_read = false;
            }
        #ifdef ENABLE_MPI
            if(m_mpi)
                {
                bcast(state_read, 0, m_exec_conf->getMPICommunicator());
                bcast(N, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                bcast(vertices, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                bcast(sweep_radius, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
                }
        #endif

        if (!state_read)
            throw std::runtime_error("Error occurred while attempting to restore from gsd file.");

        count = 0;
        for (unsigned int i = 0; i < Ntypes; i++)
            {
            float dsq = 0.0;
            for (unsigned int v = 0; v < N[i]; v++)
                {
                shape[i].x[v] = vertices[count*2+0];
                shape[i].y[v] = vertices[count*2+1];
                dsq = fmax(shape[i].x[v]*shape[i].x[v] + shape[i].y[v]*shape[i].y[v], dsq);
                count++;
                }
            shape[i].diameter = 2.0*(sqrt(dsq)+shape[i].sweep_radius);
            shape[i].N = N[i];
            shape[i].sweep_radius = sweep_radius[i];
            shape[i].ignore = 0;
            }
        }
    };

#endif
