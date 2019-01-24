# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R"""Various helper utilities for geometry.
"""

from collections import Counter, defaultdict, deque
from itertools import chain
import numpy as np

def _normalize(vector):
    """Returns a normalized version of a numpy vector."""
    return vector/np.sqrt(np.dot(vector, vector));

def _polygonNormal(vertices):
    """Returns the unit normal vector of a planar set of vertices."""
    return -_normalize(np.cross(vertices[1] - vertices[0], vertices[0] - vertices[-1]));

def area(vertices, factor=1.):
    """Computes the signed area of a polygon in 2 or 3D.

    Args:
        vertices (list): (x, y) or (x, y, z) coordinates for each vertex
        factor (float): Factor to scale the resulting area by

    """
    vertices = np.asarray(vertices);
    shifted = np.roll(vertices, -1, axis=0);

    crosses = np.sum(np.cross(vertices, shifted), axis=0);

    return np.abs(np.dot(crosses, _polygonNormal(vertices))*factor/2);

def spheroArea(vertices, radius=1., factor=1.):
    """Computes the area of a spheropolygon.

    Args:
        vertices (list): List of (x, y) coordinates, in right-handed (counterclockwise) order
        radius (float): Rounding radius of the disk to expand the polygon by
        factor (float): Factor to scale the resulting area by

    """
    vertices = list(vertices);

    if not len(vertices) or len(vertices) == 1:
        return np.pi*radius*radius;

    # adjust for concave vertices
    adjustment = 0.;
    shifted = vertices[1:] + [vertices[0]];
    delta = [(x2 - x1, y2 - y1) for ((x1, y1), (x2, y2)) in zip(vertices, shifted)];
    lastDelta = [delta[-1]] + delta[:-1];
    thetas = [np.arctan2(y, x) for (x, y) in delta];
    dthetas = [(theta2 - theta1)%(2*np.pi) for (theta1, theta2) in
               zip([thetas[-1]] + thetas[:-1], thetas)];

    # non-rounded component of the given polygon + sphere
    polygonSkeleton = [];

    for ((x, y), dtheta, dr1, dr2) in zip(vertices, dthetas, lastDelta, delta):

        if dtheta > np.pi: # this is a concave vertex
            # subtract the rounded segment we'll add later
            theta = dtheta - np.pi;
            adjustment += radius*radius*theta/2;

            # add a different point to the skeleton
            h = radius/np.sin(theta/2);

            bisector = _negBisector(dr1, (-dr2[0], -dr2[1]));
            point = (x + bisector[0]*h, y + bisector[1]*h);
            polygonSkeleton.append(point);

        else:
            (dr1, dr2) = _normalize(dr1), _normalize(dr2);

            polygonSkeleton.append((x + dr1[1]*radius, y - dr1[0]*radius));
            polygonSkeleton.append((x, y));
            polygonSkeleton.append((x + dr2[1]*radius, y - dr2[0]*radius));

    # Contribution from rounded corners
    sphereContribution = (sum([theta % np.pi for theta in dthetas]))/2.*radius**2;

    return (area(polygonSkeleton) + sphereContribution - adjustment)*factor;

def rmax(vertices, radius=0., factor=1.):
    """Compute the maximum distance among a set of vertices

    Args:
        vertices (list): list of (x, y) or (x, y, z) coordinates
        factor (float): Factor to scale the result by

    """
    return (np.sqrt(np.max(np.sum(np.asarray(vertices)*vertices, axis=1))) + radius)*factor;

def _fanTriangles(vertices, faces=None):
    """Create triangles by fanning out from vertices. Returns a
    generator for vertex triplets. If faces is None, assume that
    vertices are planar and indicate a polygon; otherwise, use the
    face indices given in faces."""
    vertices = np.asarray(vertices);

    if faces is None:
        if len(vertices) < 3:
            return;
        for tri in ((vertices[0], verti, vertj) for (verti, vertj) in
                    zip(vertices[1:], vertices[2:])):
            yield tri;
    else:
        for face in faces:
            for tri in ((vertices[face[0]], vertices[i], vertices[j]) for (i, j) in
                        zip(face[1:], face[2:])):
                yield tri;

def massProperties(vertices, faces=None, factor=1.):
    """Compute the mass, center of mass, and inertia tensor of a polygon or polyhedron

    Args:
        vertices (list): List of (x, y) or (x, y, z) coordinates in 2D or 3D, respectively
        faces (list): List of vertex indices for 3D polyhedra, or None for 2D. Faces should be in right-hand order.
        factor (float): Factor to scale the resulting results by

    Returns (mass, center of mass, moment of inertia tensor in (xx,
    xy, xz, yy, yz, zz) order) specified by the given list of vertices
    and faces. Note that the faces must be listed in a consistent
    order so that normals are all pointing in the correct direction
    from the face. If given a list of 2D vertices, return the same but
    for the 2D polygon specified by the vertices.

    .. warning::
        All faces should be specified in right-handed order.

    The computation for the 3D case follows "Polyhedral Mass
    Properties (Revisited) by David Eberly, available at:

    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

    """
    vertices = np.array(vertices, dtype=np.float64);

    # Specially handle 2D
    if len(vertices[0]) == 2:
        # First, calculate the center of mass and center the vertices
        shifted = list(vertices[1:]) + [vertices[0]];
        a_s = [(x1*y2 - x2*y1) for ((x1, y1), (x2, y2))
               in zip(vertices, shifted)];
        triangleCOMs = [(v0 + v1)/3 for (v0, v1) in zip(vertices, shifted)];
        COM = np.sum([a*com for (a, com) in zip(a_s, triangleCOMs)],
                     axis=0)/np.sum(a_s);
        vertices -= COM;

        shifted = list(vertices[1:]) + [vertices[0]];
        f = lambda x1, x2: x1*x1 + x1*x2 + x2*x2;
        Ixyfs = [(f(y1, y2), f(x1, x2)) for ((x1, y1), (x2, y2))
                 in zip(vertices, shifted)];

        Ix = sum(I*a for ((I, _), a) in zip(Ixyfs, a_s))/12.;
        Iy = sum(I*a for ((_, I), a) in zip(Ixyfs, a_s))/12.;

        I = np.array([Ix, 0, 0, Iy, 0, Ix + Iy]);

        return area(vertices)*factor, COM, factor*I;

    # multiplicative factors
    factors = 1./np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120]);

    # order: 1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    intg = np.zeros(10);

    for (v0, v1, v2) in _fanTriangles(vertices, faces):
        # (xi, yi, zi) = vi
        abc1 = v1 - v0;
        abc2 = v2 - v0;
        d = np.cross(abc1, abc2);

        temp0 = v0 + v1;
        f1 = temp0 + v2;
        temp1 = v0*v0;
        temp2 = temp1 + v1*temp0;
        f2 = temp2 + v2*f1;
        f3 = v0*temp1 + v1*temp2 + v2*f2;
        g0 = f2 + v0*(f1 + v0);
        g1 = f2 + v1*(f1 + v1);
        g2 = f2 + v2*(f1 + v2);

        intg[0] += d[0]*f1[0];
        intg[1:4] += d*f2;
        intg[4:7] += d*f3;
        intg[7] += d[0]*(v0[1]*g0[0] + v1[1]*g1[0] + v2[1]*g2[0]);
        intg[8] += d[1]*(v0[2]*g0[1] + v1[2]*g1[1] + v2[2]*g2[1]);
        intg[9] += d[2]*(v0[0]*g0[2] + v1[0]*g1[2] + v2[0]*g2[2]);

    intg *= factors;

    mass = intg[0];
    com = intg[1:4]/mass;

    moment = np.zeros(6);

    moment[0] = intg[5] + intg[6] - mass*np.sum(com[1:]**2);
    moment[1] = -(intg[7] - mass*com[0]*com[1]);
    moment[2] = -(intg[9] - mass*com[0]*com[2]);
    moment[3] = intg[4] + intg[6] - mass*np.sum(com[[0, 2]]**2);
    moment[4] = -(intg[8] - mass*com[1]*com[2]);
    moment[5] = intg[4] + intg[5] - mass*np.sum(com[:2]**2);

    return mass*factor, com, moment*factor;

def center(vertices, faces=None):
    """Centers shapes in 2D or 3D.

    Args:
        vertices (list): List of (x, y) or (x, y, z) coordinates in 2D or 3D, respectively
        faces (list): List of vertex indices for 3D polyhedra, or None for 2D. Faces should be in right-hand order.

    Returns a list of vertices shifted to have the center of mass of
    the given points at the origin. Shapes should be specified in
    right-handed order. If the input shape has no mass, return the
    input.

    .. warning::
        All faces should be specified in right-handed order.

    """
    (mass, COM, _) = massProperties(vertices, faces);
    if mass > 1e-6:
        return np.asarray(vertices) - COM[np.newaxis, :];
    else:
        return np.asarray(vertices);

def _negBisector(p1, p2):
    """Return the negative bisector of an angle given by points p1 and p2"""
    return -_normalize(_normalize(p1) + _normalize(p2));

def convexHull(vertices, tol=1e-6):
    """Compute the 3D convex hull of a set of vertices and merge coplanar faces.

    Args:
        vertices (list): List of (x, y, z) coordinates
        tol (float): Floating point tolerance for merging coplanar faces


    Returns an array of vertices and a list of faces (vertex
    indices) for the convex hull of the given set of vertice.

    .. note::
        This method uses scipy's quickhull wrapper and therefore requires scipy.

    """
    from scipy.spatial import cKDTree, ConvexHull;
    from scipy.sparse.csgraph import connected_components;

    hull = ConvexHull(vertices);
    # Triangles in the same face will be defined by the same linear equalities
    dist = cKDTree(hull.equations);
    trianglePairs = dist.query_pairs(tol);

    connectivity = np.zeros((len(hull.simplices), len(hull.simplices)), dtype=np.int32);

    for (i, j) in trianglePairs:
        connectivity[i, j] = connectivity[j, i] = 1;

    # connected_components returns (number of faces, cluster index for each input)
    (_, joinTarget) = connected_components(connectivity, directed=False);
    faces = defaultdict(list);
    norms = defaultdict(list);
    for (idx, target) in enumerate(joinTarget):
        faces[target].append(idx);
        norms[target] = hull.equations[idx][:3];

    # a list of sets of all vertex indices in each face
    faceVerts = [set(hull.simplices[faces[faceIndex]].flat) for faceIndex in sorted(faces)];
    # normal vector for each face
    faceNorms = [norms[faceIndex] for faceIndex in sorted(faces)];

    # polygonal faces
    polyFaces = [];
    for (norm, faceIndices) in zip(faceNorms, faceVerts):
        face = np.array(list(faceIndices), dtype=np.uint32);
        N = len(faceIndices);

        r = hull.points[face];
        rcom = np.mean(r, axis=0);

        # plane_{a, b}: basis vectors in the plane
        plane_a = r[0] - rcom;
        plane_a /= np.sqrt(np.sum(plane_a**2));
        plane_b = np.cross(norm, plane_a);

        dr = r - rcom[np.newaxis, :];

        thetas = np.arctan2(dr.dot(plane_b), dr.dot(plane_a));

        sortidx = np.argsort(thetas);

        face = face[sortidx];
        polyFaces.append(face);

    return (hull.points, polyFaces);
