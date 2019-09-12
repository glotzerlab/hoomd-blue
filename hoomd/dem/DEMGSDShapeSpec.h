
// inclusion guard
#ifndef _DEM_GSD_SHAPE_SPEC_
#define _DEM_GSD_SHAPE_SPEC_

#include "hoomd/extern/gsd.h"
#include "hoomd/GSDDumpWriter.h"
#include "VectorMath.h"
#include <sstream>
#include <iostream>

template <typename Real, typename Vector>
class DEMGSDShapeSpecBase
    {
    protected:

        // Method to parse shape defitinion
        std::string getShapeType(const std::vector<Vector> &verts, const Real &radius){};

        // Helper method to parse vertices into strings
        std::string parseVertices(const std::vector<Vector> &verts){};

    };

// Partial 2D specialization
template <typename Real>
class DEMGSDShapeSpecBase<Real,vec2<Real>>
    {
    protected:

        std::string getShapeType(const std::vector< vec2<Real> > &verts, const Real &radius)
            {
            std::ostringstream shapedef;
            unsigned int nverts = verts.size();
            if (nverts == 1)
                {
                shapedef << "{\"type\": \"Disk\", " << "\"diameter\": " << Real(2)*radius << "}";
                }
            else
                {
                shapedef << "{\"type\": \"Polygon\", " << "\"rounding_radius\": " << radius <<
                            ", \"vertices\": "  << parseVertices(verts) << "}";
                }
            return shapedef.str();
            }

        std::string parseVertices(const std::vector<vec2<Real>> &verts)
            {
            std::ostringstream vertstr;
            unsigned int nverts = verts.size();
            vertstr << "[";
            for (unsigned int i = 0; i < nverts-1; i++)
                {
                vertstr << "[" << verts[i].x << ", " << verts[i].y << "], ";
                }
            vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << "]" << "]";
            return vertstr.str();
            }
    };

// Partial 3D specialization
template <typename Real>
class DEMGSDShapeSpecBase<Real,vec3<Real>>
    {
    protected:

        std::string getShapeType(const std::vector<vec3<Real>> &verts, const Real &radius)
            {
            std::ostringstream shapedef;
            unsigned int nverts = verts.size();
            if (nverts == 1)
                {
                shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
                }
            else
                {
                shapedef <<  "{\"type\": \"ConvexPolyhedron\", " << "\"rounding_radius\": " << radius <<
                            ", \"vertices\": "  << parseVertices(verts) << "}";
                }
            return shapedef.str();
            }

        std::string parseVertices(const std::vector<vec3<Real>> &verts)
            {
            std::ostringstream vertstr;
            unsigned int nverts = verts.size();
            vertstr << "[";
            for (unsigned int i = 0; i < nverts-1; i++)
                {
                vertstr << "[" << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
                }
            vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << ", " << verts[nverts-1].z  << "]" << "]";
            return vertstr.str();
            }
    };

template <typename Real, typename Vector>
class DEMGSDShapeSpec : public DEMGSDShapeSpecBase<Real, Vector>
    {
    public:
        DEMGSDShapeSpec(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) :  m_exec_conf(exec_conf), m_mpi(mpi) {};

        // write shape definition to GSD file
        int write(gsd_handle& handle,
                  const std::string& name,
                  const std::vector<std::vector<Vector>> &shapes,
                  const Real &radius);

   private:
       const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
       bool m_mpi;
    };

template <typename Real, typename Vector>
int DEMGSDShapeSpec<Real,Vector>::write(gsd_handle& handle, const std::string& name, const std::vector<std::vector<Vector>> &shapes, const Real &radius)
    {
    if(!m_exec_conf->isRoot())
        return 0;

    std::vector<std::string> type_shape_mapping(shapes.size());

    int max_len = 0;
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        {
        type_shape_mapping[i] = this->getShapeType(shapes[i], radius);
        max_len = std::max(max_len, (int)type_shape_mapping[i].size());
        }
    max_len += 1;  // for null
    m_exec_conf->msg->notice(10) << "dump.gsd: writing " << name << std::endl;
    std::vector<char> types(max_len * type_shape_mapping.size());
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        strncpy(&types[max_len*i], type_shape_mapping[i].c_str(), max_len);
    int retval = gsd_write_chunk(&handle, name.c_str(), GSD_TYPE_UINT8, type_shape_mapping.size(), max_len, 0, (void *)&types[0]);
    if (retval == -1)
        {
        m_exec_conf->msg->error() << "dump.gsd: " << strerror(errno) << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval != 0)
        {
        m_exec_conf->msg->error() << "dump.gsd: " << "Unknown error " << retval << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    return retval;
    };
#endif
