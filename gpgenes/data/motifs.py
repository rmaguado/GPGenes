import random
from .primitives import Regulation, Gene


def motif_mutual_inhibition(A: Gene, B: Gene):
    A.regulators.append(Regulation(B, -random.normalvariate(1.0, 0.2)))
    B.regulators.append(Regulation(A, -random.normalvariate(1.0, 0.2)))


def motif_mutual_activation(A, B):
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


def preset_toggle_chain(A, B, C, D, E):
    motif_mutual_inhibition(A, B)
    motif_chain(B, C, D)
    motif_convergent(C, D, E)


def preset_dual_feedforward(A, B, C, D, E):
    motif_feedforward_coherent(A, B, C)
    motif_feedforward_coherent(A, D, E)


def preset_hierarchical_control(A, B, C, D, E):
    motif_autoregulation(A)
    motif_fan_out(A, B, C)
    motif_chain(B, D, E)


def preset_signal_integrator(A, B, C, D, E):
    motif_chain(A, B, C)
    motif_chain(D, E, C)
    motif_autoregulation(C)


def preset_competition_with_output(A, B, C, D, E):
    motif_mutual_activation(A, B)
    motif_mutual_inhibition(B, C)
    motif_convergent(A, C, D)
    motif_chain(D, E, A)


MOTIFS = [
    (motif_mutual_inhibition, 2),
    (motif_mutual_activation, 2),
    (motif_feedforward_coherent, 3),
    (motif_feedforward_incoherent, 3),
    (motif_chain, 3),
    (motif_fan_out, 3),
    (motif_convergent, 3),
    (motif_autoregulation, 1),
]

PRESETS = [
    (preset_toggle_chain, 5),
    (preset_dual_feedforward, 5),
    (preset_hierarchical_control, 5),
    (preset_signal_integrator, 5),
    (preset_competition_with_output, 5),
]
