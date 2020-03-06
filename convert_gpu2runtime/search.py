import os
import sys
import glob

#Finds minimum index given an array of indexes
#Used in "searchAndAppend"
def findMinIndex(array, lineLength):
    minIdx = lineLength
    for element in array:
        if element < minIdx and not element == -1:
            minIdx = element
    return minIdx

#Searches for functions, typenames, and macros containing a given searchword
#Appends findings to a given func_list iff they are not already in the given func_list
def searchAndAppend(searchword, line, func_list):
    if searchword in line:
        idxStart = line.find(searchword, 0)

        idxEndArray = []
        idxEndArray.append(line.find("(", idxStart))
        idxEndArray.append(line.find(")", idxStart))
        idxEndArray.append(line.find(",", idxStart))
        idxEndArray.append(line.find(" ", idxStart))
        idxEndArray.append(line.find(">", idxStart))
        idxEndArray.append(line.find("<", idxStart))
        idxEndArray.append(line.find("[", idxStart))
        idxEndArray.append(line.find("]", idxStart))

        idxEnd = findMinIndex(idxEndArray, len(line) -1)

        someFunc = line[idxStart:idxEnd]
        if not someFunc in func_list:
            func_list.append(someFunc)

        newLine = line[idxEnd:len(line) - 1]
        if searchword in newLine:
            searchAndAppend(searchword, newLine, func_list)


#MAIN
dir_names = glob.glob(sys.argv[1], recursive = True)
func_name_list = []

for filename in dir_names:
    if not "GPURuntime.h" in filename and not "replace.py" in filename and not "search.py" in filename:
        some_file = open(filename, "r")
        for line in some_file:
            searchAndAppend("cuda", line, func_name_list)
            searchAndAppend("hip", line, func_name_list)
            searchAndAppend("CUDA", line, func_name_list)
            searchAndAppend("HIP", line, func_name_list)

for word in func_name_list:
  print(word)
