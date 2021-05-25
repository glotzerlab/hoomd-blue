
# class group(object):

#     def __init__(self, *walls):
#         self.spheres = []
#         self.cylinders = []
#         self.planes = []
#         for wall in walls:
#             self.add(wall)

#     def add(self, wall):

#         if (isinstance(wall, sphere)):
#             self.spheres.append(wall)
#         elif (isinstance(wall, cylinder)):
#             self.cylinders.append(wall)
#         elif (isinstance(wall, plane)):
#             self.planes.append(wall)
#         elif (type(wall) == list):
#             for wall_el in wall:
#                 if (isinstance(wall_el, sphere)):
#                     self.spheres.append(wall_el)
#                 elif (isinstance(wall_el, cylinder)):
#                     self.cylinders.append(wall_el)
#                 elif (isinstance(wall_el, plane)):
#                     self.planes.append(wall_el)
#                 else:
#                     print("Input of type " + str(type(wall_el))
#                           + " is not allowed. Skipping invalid list element...")
#         else:
#             print("Input of type " + str(type(wall)) + " is not allowed.")

#     def del_sphere(self, *indexs):
#         for index in indexs:
#             if type(index) is int:
#                 index = [index]
#             elif type(index) is range:
#                 index = list(index)
#             index = list(set(index))
#             index.sort(reverse=True)
#             for i in index:
#                 try:
#                     del (self.spheres[i])
#                 except IndexValueError:
#                     hoomd.context.current.device.cpp_msg.error(
#                         "Specified index for deletion is not valid.\n")
#                     raise RuntimeError("del_sphere failed")

#     def del_cylinder(self, *indexs):
#         for index in indexs:
#             if type(index) is int:
#                 index = [index]
#             elif type(index) is range:
#                 index = list(index)
#             index = list(set(index))
#             index.sort(reverse=True)
#             for i in index:
#                 try:
#                     del (self.cylinders[i])
#                 except IndexValueError:
#                     hoomd.context.current.device.cpp_msg.error(
#                         "Specified index for deletion is not valid.\n")
#                     raise RuntimeError("del_cylinder failed")

#     def del_plane(self, *indexs):
#         for index in indexs:
#             if type(index) is int:
#                 index = [index]
#             elif type(index) is range:
#                 index = list(index)
#             index = list(set(index))
#             index.sort(reverse=True)
#             for i in index:
#                 try:
#                     del (self.planes[i])
#                 except IndexValueError:
#                     hoomd.context.current.device.cpp_msg.error(
#                         "Specified index for deletion is not valid.\n")
#                     raise RuntimeError("del_plane failed")

#     ## \internal
#     # \brief Returns output for print
#     def __str__(self):
#         output = "Wall_Data_Structure:\nspheres:%s{" % (len(self.spheres))
#         for index in range(len(self.spheres)):
#             output += "\n[%s:\t%s]" % (repr(index), str(self.spheres[index]))

#         output += "}\ncylinders:%s{" % (len(self.cylinders))
#         for index in range(len(self.cylinders)):
#             output += "\n[%s:\t%s]" % (repr(index), str(self.cylinders[index]))

#         output += "}\nplanes:%s{" % (len(self.planes))
#         for index in range(len(self.planes)):
#             output += "\n[%s:\t%s]" % (repr(index), str(self.planes[index]))

#         output += "}"
#         return output

class _Base_Wall(object):
    def __init__(self):
        self._attached = False
        self._cpp_obj = None

class Sphere(_Base_Wall):
    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), inside=True):
        self.r = r
        self.origin = origin
        self.inside = inside

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tInside=%s" % (str(self.r), str(
            self.origin), str(self.inside))

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'inside': %s}" % (str(
            self.r), str(self.origin), str(self.inside))

class Cylinder(_Base_Wall):
    def __init__(self,
                 r=0.0,
                 origin=(0.0, 0.0, 0.0),
                 axis=(0.0, 0.0, 1.0),
                 inside=True):
        self.r = r
        self.origin = origin
        self.axis = axis
        self.inside = inside

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tAxis=%s\tInside=%s" % (str(
            self.r), str(self.origin), str(self.axis), str(self.inside))

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'axis': %s, 'inside': %s}" % (str(
            self.r), str(self.origin), str(self.axis), str(self.inside))


class Plane(_Base_Wall):
    def __init__(self,
                 origin=(0.0, 0.0, 0.0),
                 normal=(0.0, 0.0, 1.0),
                 inside=True):
        self.origin = origin
        self.normal = normal
        self.inside = inside

    def __str__(self):
        return "Origin=%s\tNormal=%s\tInside=%s" % (str(
            self.origin), str(self.normal), str(self.inside))

    def __repr__(self):
        return "{'origin':%s, 'normal': %s, 'inside': %s}" % (str(
            self.origin), str(self.normal), str(self.inside))

