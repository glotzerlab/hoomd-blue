import numpy
import math

# these series of functions are used to calculate the values for the test cases used below
def dchi_dr1(n1, n2, r1, r2, r3, r4):
    n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
    n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))
    numerator = numpy.dot(n1hat, n2hat) * numpy.cross(
        numpy.dot(n1hat, n2hat) * n1hat - n2hat,
        r2 - r3) / numpy.linalg.norm(n1)
    denominator = numpy.sqrt(1 - numpy.dot(numpy.cross(
        n1hat, n2hat), numpy.cross(n1hat, n2hat))) * numpy.linalg.norm(
            numpy.cross(n1hat, n2hat))
    return numerator / denominator


def dchi_dr2(n1, n2, r1, r2, r3, r4):
    n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
    n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))
    numerator = numpy.dot(n1hat, n2hat)*(numpy.cross(numpy.dot(n1hat,n2hat)*n2hat-n1hat, r3-r4)\
                                         /numpy.linalg.norm(n2)-numpy.cross(numpy.dot(n1hat,n2hat)*n1hat-n2hat, r1-r3)/numpy.linalg.norm(n1))
    denominator = numpy.sqrt(1 - numpy.dot(numpy.cross(
        n1hat, n2hat), numpy.cross(n1hat, n2hat))) * numpy.linalg.norm(
            numpy.cross(n1hat, n2hat))
    return numerator / denominator


def dchi_dr3(n1, n2, r1, r2, r3, r4):
    n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
    n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))
    numerator = numpy.dot(n1hat, n2hat) * (numpy.cross(numpy.dot(n1hat,n2hat) * n1hat - n2hat, r1 - r2)\
                                            / numpy.linalg.norm(n1) - numpy.cross(numpy.dot(n1hat,n2hat)*n2hat - n1hat, r2 - r4)/numpy.linalg.norm(n2))
    denominator = numpy.sqrt(1 - numpy.dot(numpy.cross(
        n1hat, n2hat), numpy.cross(n1hat, n2hat))) * numpy.linalg.norm(
            numpy.cross(n1hat, n2hat))
    return numerator / denominator


def dchi_dr4(n1, n2, r1, r2, r3, r4):
    n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
    n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))
    numerator = numpy.dot(n1hat, n2hat) * numpy.cross(
        numpy.dot(n1hat, n2hat) * n2hat - n1hat,
        r2 - r3) / numpy.linalg.norm(n2)
    denominator = numpy.sqrt(1 - numpy.dot(numpy.cross(
        n1hat, n2hat), numpy.cross(n1hat, n2hat))) * numpy.linalg.norm(
            numpy.cross(n1hat, n2hat))
    return numerator / denominator


def chi_from_pos(posa, posb, posc, posd):
    n1 = numpy.cross(posa - posb, posb - posc)
    n2 = numpy.cross(posb - posc, posc - posd)
    mag = numpy.dot(n1, n2) / numpy.linalg.norm(n1) / numpy.linalg.norm(n2)
    return math.acos(numpy.linalg.norm(mag))


def du_dchi_periodic(chi, chi0, k, n, d):
    return -k * n * d * numpy.sin(n * chi - chi0)


def du_dchi_harmonic(chi, k, chi0):
    return k * (chi - chi0)


def periodic_improper_energy(chi, k, n, d, chi0):
    return (k * (1 + d * numpy.cos(n * chi - chi0)))


def get_force_vectors(chi, n1, n2, r1, r2, r3, r4, chi0, k, d, n):
    f_matrix = numpy.zeros((4, 3))
    f_matrix[0, :] = dchi_dr1(n1, n2, r1, r2, r3, r4) * du_dchi_periodic(
        chi, chi0=chi0, k=k, d=d, n=n)
    f_matrix[1, :] = dchi_dr2(n1, n2, r1, r2, r3, r4) * du_dchi_periodic(
        chi, chi0=chi0, k=k, d=d, n=n)
    f_matrix[2, :] = dchi_dr3(n1, n2, r1, r2, r3, r4) * du_dchi_periodic(
        chi, chi0=chi0, k=k, d=d, n=n)
    f_matrix[3, :] = dchi_dr4(n1, n2, r1, r2, r3, r4) * du_dchi_periodic(
        chi, chi0=chi0, k=k, d=d, n=n)
    return f_matrix
