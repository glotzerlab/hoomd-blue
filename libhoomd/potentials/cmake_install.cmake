# Install script for directory: /Users/jamesaan/hoomd-blue/libhoomd/potentials

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/Users/jamesaan/hoomd-install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hoomd" TYPE FILE FILES
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/AllBondPotentials.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/AllPairPotentials.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/AllTripletPotentials.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorBondFENE.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorBondHarmonic.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairDPDLJThermo.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairDPDThermo.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairEwald.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairForceShiftedLJ.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairGauss.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairLJ.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairMie.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairMoliere.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairMorse.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairSLJ.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairYukawa.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorPairZBL.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/EvaluatorTersoff.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialBond.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialBondGPU.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialPair.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialPairDPDThermo.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialPairDPDThermoGPU.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialPairGPU.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialTersoff.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialTersoffGPU.h"
    "/Users/jamesaan/hoomd-blue/libhoomd/potentials/PotentialPairDPDThermoGPU.cuh"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

