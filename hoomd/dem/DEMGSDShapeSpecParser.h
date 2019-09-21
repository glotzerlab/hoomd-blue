
// inclusion guard
#ifndef __DEM_GSD_SHAPE_SPEC_PARSER
#define __DEM_GSD_SHAPE_SPEC_PARSER

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
class Parser<Real,vec2<Real>>
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

        std::string parseVertices(const std::vector<vec2<Real>> &verts) const
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
class Parser<Real,vec3<Real>>
    {

    protected:

        std::string getTypeShape(const std::vector<vec3<Real>> &verts, const Real &radius) const
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

        std::string parseVertices(const std::vector<vec3<Real>> &verts) const
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

        std::vector<std::string> getTypeShapeMapping(const std::vector<std::vector<Vector>> &verts, const Real &radius) const
            {
            std::vector<std::string> type_shape_mapping(verts.size());
            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
                {
                type_shape_mapping[i] = this->getTypeShape(verts[i], radius);
                }
            return type_shape_mapping;
            }
    };

//template<typename Real, typename Vector>
//class DEMShapeSpecParserBase
//    {
//
//    public:
//
//        std::string getTypeShape(const std::vector<Vector> &verts, const Real &radius) const;
//
//        std::string parseVertices(const std::vector<Vector> &verts) const;
//
//        std::vector<std::string> getTypeShapeMapping(const std::vector<std::vector<Vector>> &verts, const Real &radius) const
//            {
//            std::vector<std::string> type_shape_mapping(verts.size());
//            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
//                {
//                type_shape_mapping[i] = getTypeShape(verts[i], radius);
//                }
//            return type_shape_mapping;
//            }
//    };
//
//template <typename Real, typename Vector>
//class DEMShapeSpecParser : public DEMShapeSpecParserBase<Real, Vector>
//    {
//
//    public:
//
//        std::string getTypeShape(const std::vector<Vector> &verts, const Real &radius) const;
//
//        std::string parseVertices(const std::vector<Vector> &verts) const;
//    };
//
//// Partial 2D specialization
//template <typename Real>
//class DEMShapeSpecParser<Real,vec2<Real>> : public DEMShapeSpecParserBase<Real, vec2<Real>>
//    {
//
//    public:
//
//        std::string getTypeShape(const std::vector< vec2<Real> > &verts, const Real &radius)
//            {
//            std::ostringstream shapedef;
//            unsigned int nverts = verts.size();
//            if (nverts == 1)
//                {
//                shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
//                }
//            else
//                {
//                shapedef << "{\"type\": \"Polygon\", " << "\"rounding_radius\": " << radius <<
//                            ", \"vertices\": "  << parseVertices(verts) << "}";
//                }
//            return shapedef.str();
//            }
//
//        std::string parseVertices(const std::vector<vec2<Real>> &verts)
//            {
//            std::ostringstream vertstr;
//            unsigned int nverts = verts.size();
//            vertstr << "[";
//            for (unsigned int i = 0; i < nverts-1; i++)
//                {
//                vertstr << "[" << verts[i].x << ", " << verts[i].y << "], ";
//                }
//            vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << "]" << "]";
//            return vertstr.str();
//            }
//    };
//
//// Partial 3D specialization
//template <typename Real>
//class DEMShapeSpecParser<Real,vec3<Real>> : public DEMShapeSpecParserBase<Real, vec3<Real>>
//    {
//    public:
//
//        std::string getTypeShape(const std::vector<vec3<Real>> &verts, const Real &radius)
//            {
//            std::ostringstream shapedef;
//            unsigned int nverts = verts.size();
//            if (nverts == 1)
//                {
//                shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
//                }
//            else
//                {
//                shapedef <<  "{\"type\": \"ConvexPolyhedron\", " << "\"rounding_radius\": " << radius <<
//                            ", \"vertices\": "  << parseVertices(verts) << "}";
//                }
//            return shapedef.str();
//            }
//
//        std::string parseVertices(const std::vector<vec3<Real>> &verts)
//            {
//            std::ostringstream vertstr;
//            unsigned int nverts = verts.size();
//            vertstr << "[";
//            for (unsigned int i = 0; i < nverts-1; i++)
//                {
//                vertstr << "[" << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
//                }
//            vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << ", " << verts[nverts-1].z  << "]" << "]";
//            return vertstr.str();
//            }
//
//    };
//

#endif
