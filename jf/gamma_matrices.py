"""Chroma gamma matrices

   jmf 20130517
"""

import numpy as np

Gamma = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    [[-1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]],
    [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
    [[0, -1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
    [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    [[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
    [[0, 0, -1j, 0], [0, 0, 0, 1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    [[1j, 0, 0, 0], [0, -1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]],
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, -1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
]

# Use Kronecker product to make tensor product of spin
# matrix S with color matrix C. From np.kron(S,C) get
# structure of blocks of the second array scaled by
# elements of the first, ie
#
# [[ S00 C, S01 C, S02 C, ...],
#  [ S10 C, S11 C, S12 C, ...],
#  ...                       ]]

Is = Gamma[0]  # spin identity
G1 = Gamma[1]
G2 = Gamma[2]
G3 = Gamma[4]
G4 = Gamma[8]
G5 = Gamma[15]

Ic = np.identity(3, int)  # colour identity

G5Ic = np.kron(G5, Ic)

# Conversion for Gamma_mu = Gamma[mu2i[mu]] = Gamma[i]
# where mu is Lorentz index with convention 0123 for xyzt
mu2i = (1, 2, 4, 8)  # gamma_{x,y,z,t} =Gamma[{1,2,4,8}]

# Is gamma matrix hermitian or antihermitian?
# Gamma[i]^dagger = Gammahermiticity[i] Gamma[i]
GammaHermiticity = (1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1)
GHsign = GammaHermiticity

# Does gamma matrix commute/anticommute with G5?
G5Commutativity = (1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1)
G5sign = G5Commutativity

# What do you get when you premultiply one of the Gammas by G5?
# G5.Gamma[i] = +/- Gamma[j]
# Here are the signs and indices j for each i.
G5GiSignIndex = (
    (1, 15),
    (-1, 14),
    (1, 13),
    (-1, 12),
    (-1, 11),
    (1, 10),
    (-1, 9),
    (1, 8),
    (1, 7),
    (-1, 6),
    (1, 5),
    (-1, 4),
    (-1, 3),
    (1, 2),
    (-1, 1),
    (1, 0),
)
G5Gi = G5GiSignIndex
