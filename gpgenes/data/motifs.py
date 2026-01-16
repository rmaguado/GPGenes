import random
from .primitives import Regulation, Gene


def motif_mutual_inhibition(A: Gene, B: Gene):
    A.regulators.append(Regulation(B, -random.normalvariate(1.0, 0.2)))
    B.regulators.append(Regulation(A, -random.normalvariate(1.0, 0.2)))


def motif_mutual_activation(A, B):
    A.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))


def motif_negative_feedback(A, B):
    A.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))
    B.regulators.append(Regulation(A, -random.normalvariate(1.0, 0.2)))


def motif_positive_feedback(A, B):
    A.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))


def motif_feedforward_coherent(A, B, C):
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))


def motif_feedforward_incoherent(A, B, C):
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(B, -random.normalvariate(1.0, 0.2)))


def motif_chain(A, B, C):
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))


def motif_fan_out(A, B, C):
    B.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))


def motif_convergent(A, B, C):
    C.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))
    C.regulators.append(Regulation(B, random.normalvariate(1.0, 0.2)))


def motif_autoregulation(A):
    A.regulators.append(Regulation(A, random.normalvariate(1.0, 0.2)))


MOTIFS = [
    (motif_mutual_inhibition, 2),
    (motif_mutual_activation, 2),
    (motif_negative_feedback, 2),
    (motif_positive_feedback, 2),
    (motif_feedforward_coherent, 3),
    (motif_feedforward_incoherent, 3),
    (motif_chain, 3),
    (motif_fan_out, 3),
    (motif_convergent, 3),
    (motif_autoregulation, 1),
]
