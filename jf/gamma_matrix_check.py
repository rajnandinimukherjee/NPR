"""To check that the Chroma gamma matrices satisfy the following table.

   jmf 201312118, 20140528

   i  Gi     G5 Gi = sign Gj    sign   j

   0   1      G5                +1    15
   1   G1    -G1G5              -1    14
   2   G2     G5G2              +1    13
   3   G1G2  -G3G4              -1    12
   4   G3    -G3G5              -1    11
   5   G1G3   G2G4              +1    10
   6   G2G3  -G1G4              -1    9
   7   G5G4   G4                +1    8
   8   G4     G5G4              +1    7
   9   G1G4  -G2G3              -1    6
   10  G2G4   G1G3              +1    5
   11  G3G5  -G3                -1    4
   12  G3G4  -G1G2              -1    3
   13  G5G2   G2                +1    2
   14  G1G5  -G1                -1    1
   15  G5     1                 +1    0

   Also check Gammahermiticity

     Gi^dag = GHsign Gi

   Finally check that if G5 Gi = sigma Gj then

     GHsign[i] G5sign[i] = GHsign[j]

   where G5 Gi = G5sign Gi G5
"""

import numpy as np
from gamma_matrices import Gamma,GHsign,G5sign,G5Gi

# Use the Kronecker product to make the tensor product of a spin
# matrix S with a color matrix C. From np.kron(S,C) we get a
# matrix with structure of blocks of the second array scaled by
# elements of the first, ie
#
# [[ S00 C, S01 C, S02 C, ...],
#  [ S10 C, S11 C, S12 C, ...],
#  ...                       ]]

Is=Gamma[0] # spin identity
G1=Gamma[1]
G2=Gamma[2]
G3=Gamma[4]
G4=Gamma[8]
G5=Gamma[15]

Ic=np.identity(3,int) # colour identity

IsIc=np.kron(Is,Ic)
G5Ic=np.kron(G5,Ic)
G1Ic=np.kron(G1,Ic)
G2Ic=np.kron(G2,Ic)
G3Ic=np.kron(G3,Ic)
G4Ic=np.kron(G4,Ic)

chromagammas=(IsIc,G1Ic,G2Ic,np.dot(G1Ic,G2Ic),
              G3Ic,np.dot(G1Ic,G3Ic),np.dot(G2Ic,G3Ic),np.dot(G5Ic,G4Ic),
              G4Ic,np.dot(G1Ic,G4Ic),np.dot(G2Ic,G4Ic),np.dot(G3Ic,G5Ic),
              np.dot(G3Ic,G4Ic),np.dot(G5Ic,G2Ic),np.dot(G1Ic,G5Ic),G5Ic)

print('Checking G5 Gi = sigma Gj')
for i,g in enumerate(chromagammas):
  GiIc=np.kron(Gamma[i],Ic)
  sgn,j=G5Gi[i]
  print(i,np.allclose(GiIc,g),
        np.allclose(np.dot(G5Ic,GiIc),sgn*np.kron(Gamma[j],Ic)))

print('\nChecking Gi^dag = GHsign Gi')
for i,g in enumerate(chromagammas):
  GiIc=np.kron(g,Ic)
  GiIcdag=GiIc.transpose().conjugate()
  if np.allclose(GiIc,GiIcdag):
    h=1
  elif np.allclose(GiIc,-GiIcdag):
    h=-1
  else:
    print('Neither hermitian nor antihermitian')
  print('{:2d}  {:2d} {:2d}'.format(i,h,GHsign[i]))

##G5sign=(1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1)
##GHsign=Gammahermiticity
print('\nChecking GHsign[i] G5sign[i] = GHsign[j] if G5 Gi = sigma Gj')
print('Last two columns should match')
print(' i   j  ghi g5i gh5i ghj')
for i,g in enumerate(chromagammas):
  sigma,j=G5Gi[i]
  ghi=GHsign[i]
  ghj=GHsign[j]
  g5i=G5sign[i]
  print(('{:2d}  '*6).format(i,j,ghi,g5i,ghi*g5i,ghj))
  
