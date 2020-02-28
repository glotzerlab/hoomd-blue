import os
import sys
import glob

dir_names = glob.glob(sys.argv[1], recursive = True)
#print(dir_names)
#dir_path = os.path.dirname(os.path.abspath(__file__))
func_name_list = []

#def search_stuff(dir_path):
 # for root, dirs, files in os.walk(path):
  #  for file_ in files:
for filename in dir_names:
    #print(file_)
    some_file = open(filename, "r")
    for line in some_file:
        if "cuda" in line:
            func_name_list.append(line)
            """#print("Hello")
            idxStart = line.find("cuda", 0)
            idxEnd = line.find("(", 0)
            if not idxEnd == -1:
                if not line[idxStart:idxEnd] in func_name_list:
                    func_name_list.append(line[idxStart:idxEnd])
            else:
                idxEnd = line.find(" ", idxStart)
                if not idxEnd == -1:
                    if not line[idxStart:idxEnd] in func_name_list:
                        func_name_list.append(line[idxStart:idxEnd])
                else:
                    if not line[idxStart:] in func_name_list:
                        func_name_list.append(line[idxStart:])
"""
        #if "hip" in line:
         #   func_name_list.append(line)
"""            #print("low")
            idxStart = line.find("hip", 0)
            idxEnd = line.find("(", 0)
            if not idxEnd == -1:
                if not line[idxStart:idxEnd] in func_name_list:
                    func_name_list.append(line[idxStart:idxEnd])
            else:
                idxEnd = line.find(" ", idxStart)
                if not idxEnd == -1:
                    if not line[idxStart:idxEnd] in func_name_list:
                        func_name_list.append(line[idxStart:idxEnd])
                else:
                    if not line[idxStart:] in func_name_list:
                        func_name_list.append(line[idxStart:])
"""
#if "CUDA" in line:
#    func_name_list.append(line)
"""            idxStart = line.find("CUDA", 0)
            idxEnd = line.find("(", 0)
            if not idxEnd == -1:
                if not line[idxStart:idxEnd] in func_name_list:
                    func_name_list.append(line[idxStart:idxEnd])
            else:
                idxEnd = line.find(" ", idxStart)
                if not idxEnd == -1:
                    if not line[idxStart:idxEnd] in func_name_list:
                        func_name_list.append(line[idxStart:idxEnd])
                else:
                    if not line[idxStart:] in func_name_list:
                        func_name_list.append(line[idxStart:])
"""
#if "HIP" in line:
#    func_name_list.append(line)
"""            idxStart = line.find("HIP", 0)
            idxEnd = line.find("(", 0)
            if not idxEnd == -1:
                if not line[idxStart:idxEnd] in func_name_list:
                        func_name_list.append(line[idxStart:idxEnd])
            else:
                idxEnd = line.find(" ", idxStart)
                if not idxEnd == -1:
                    if not line[idxStart:idxEnd] in func_name_list:
                        func_name_list.append(line[idxStart:idxEnd])
                else:
                    if not line[idxStart:] in func_name_list:
                        func_name_list.append(line[idxStart:])
""" #           search_stuff(os.path.join(dir_path, directory)

#search_stuff(dir_path)
for word in func_name_list:
  print(word)
