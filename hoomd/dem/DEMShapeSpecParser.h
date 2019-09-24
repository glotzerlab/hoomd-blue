// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __DEMSHAPESPECPARSER_H__
#define __DEMSHAPESPECPARSER_H__

#include "hoomd/extern/gsd.h"
#include "hoomd/GSDDumpWriter.h"
#include "VectorMath.h"
#include <sstream>
#include <iostream>

template<typename Real, typename Vector>
class Parser
    {
    protected:

        std::string getTypeShape(const std::vector<Vector> &verts, const Real &radius) const;

        std::string parseVertices(const std::vector<Vector> &verts) const;
    };

// Partial 2D specialization
template <typename Real>
class Parser<Real,vec2<Real> >
    {

    protected:

        std::string getTypeShape(const std::vector< vec2<Real> > &verts, const Real &radius) const
            {
            std::ostringstream shapedef;
            unsigned int nverts = verts.size();
            if (nverts == 1)
                {
                shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
                }
            else
                {
                shapedef << "{\"type\": \"Polygon\", " << "\"rounding_radius\": " << radius <<
                            ", \"vertices\": "  << parseVertices(verts) << "}";
                }
            return shapedef.str();
            }

        std::string parseVertices(const std::vector<vec2<Real> > &verts) const
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
class Parser<Real,vec3<Real> >
    {

    protected:

        std::string getTypeShape(const std::vector<vec3<Real> > &verts, const Real &radius) const
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

        std::string parseVertices(const std::vector<vec3<Real> > &verts) const
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

template<typename Real, typename Vector>
class DEMShapeSpecParser : public Parser<Real, Vector>
    {

    public:

        std::vector<std::string> getTypeShapeMapping(const std::vector<std::vector<Vector> > &verts, const Real &radius) const
            {
            std::vector<std::string> type_shape_mapping(verts.size());
            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
                {
                type_shape_mapping[i] = this->getTypeShape(verts[i], radius);
                }
            return type_shape_mapping;
            }
    };

#endif
