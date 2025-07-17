# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mocked module for use in documentation builds."""


class BuildInfo:

    @staticmethod
    def getVersion():
        pass

    @staticmethod
    def getCompileFlags():
        pass

    @staticmethod
    def getEnableGPU():
        pass

    @staticmethod
    def getGPUAPIVersion():
        pass

    @staticmethod
    def getGPUPlatform():
        pass

    @staticmethod
    def getCXXCompiler():
        pass

    @staticmethod
    def getEnableMPI():
        pass

    @staticmethod
    def getSourceDir():
        pass

    @staticmethod
    def getInstallDir():
        pass

    @staticmethod
    def getFloatingPointPrecision():
        pass


class Trigger:
    pass


class PeriodicTrigger:
    pass


class BeforeTrigger:
    pass


class OnTrigger:
    pass


class AfterTrigger:
    pass


class NotTrigger:
    pass


class AndTrigger:
    pass


class OrTrigger:
    pass


def make_scalar3():
    pass


def make_int3():
    pass


def make_char3():
    pass


class Variant:
    pass


class VariantConstant:
    pass


class VariantRamp:

    def __init__(*args):
        pass


class VariantCycle:
    pass


class VariantPower:
    pass


class VectorVariantBox:
    pass


class VectorVariantBoxConstant:
    pass


class VectorVariantBoxInterpolate:
    pass


class VectorVariantBoxInverseVolumeRamp:
    pass


class LocalParticleDataHost:
    pass


class LocalBondDataHost:
    pass


class LocalAngleDataHost:
    pass


class LocalDihedralDataHost:
    pass


class LocalImproperDataHost:
    pass


class LocalConstraintDataHost:
    pass


class LocalPairDataHost:
    pass


class ParticleFilter:
    pass


class ParticleFilterAll:
    pass


class ParticleFilterNull:
    pass


class ParticleFilterRigid:
    pass


class ParticleFilterSetDifference:
    pass


class ParticleFilterUnion:
    pass


class ParticleFilterIntersection:
    pass


class ParticleFilterTags:
    pass


class ParticleFilterType:
    pass


class MPIConfiguration:
    pass


class LocalForceComputeDataHost:
    pass


def abort_mpi(*args):
    pass
