import numpy as np
from dscribe.descriptors import SOAP
from ase import Atoms

class Polarizability:
    """Processing of the polarizability tensor by splitting into
    isotropic and anisotropic components

    alphas: np array of polarizabilities of size N x 3 x 3 """

    def __init__(self, alphas):
        self.setSize = np.shape(alphas)[0]
        self.fullPolarizability = alphas

    def process_alpha(self):

        anisotropic = np.zeros((self.setSize,3,3))

        isotropic = (self.fullPolarizability[:,0,0] +
                  self.fullPolarizability[:,1,1] +
                  self.fullPolarizability[:,2,2]) / 3

        zeros = np.zeros(self.setSize)

        isotropicIdentity = np.array([[isotropic,zeros,zeros],[zeros,isotropic,zeros],[zeros,zeros,isotropic]])

        anisotropic = self.fullPolarizability - np.transpose(isotropicIdentity,(2,1,0))

        return isotropic, anisotropic


class Molecules:
    """Molecules class"""

    def __init__(self, atomList, geometries):
        self.atomList = atomList
        self.geometries = geometries
        self.Natoms = len(atomList)
        self.Ngeometries = np.shape(geometries)[0]

    def get_bonds(self, frame = 0, r_cutoff = 1.6):

        bondList = []

        for ij in range(self.Natoms):
            for jk in range(ij+1,self.Natoms):

                distance = np.linalg.norm(self.geometries[frame,ij,:] - self.geometries[frame,jk,:])

                if distance < r_cutoff:
                    bondList.append([ij,jk])

        return bondList

    def get_bond_features(self, representation, bonds, Nfeatures=390):

        Nbonds = len(bonds)

        features = np.zeros((Nbonds,Nfeatures,self.Ngeometries))
        RR = np.zeros((Nbonds,9,self.Ngeometries))

        for ij in range(self.Ngeometries):

            ASEatoms = Atoms(self.atomList,positions=self.geometries[ij,:,:])

            bond_centers = np.zeros((Nbonds,3))

            for bb in range(Nbonds):

                mm = bonds[bb][0]
                nn = bonds[bb][1]

                atom1 = self.geometries[ij,mm,:]
                atom2 = self.geometries[ij,nn,:]

                bond_centers[bb,:] = (atom1 + atom2)/2

                bond_norm = (atom1 - atom2)/np.linalg.norm(atom1-atom2)

                RR[bb,:,ij] = get_RR(bond_norm)

            soap_ = representation.create(ASEatoms, bond_centers)

            features[:,:,ij] = soap_[:,:]

        return features, RR

def get_RR(bond_vector):

    RR = np.zeros((3,3))

    for ii in range(3):
        for jj in range(3):

            if ii == jj:
                RR[ii,jj] = bond_vector[ii]*bond_vector[jj] - 1/3
            else:
                RR[ii,jj] = bond_vector[ii]*bond_vector[jj]


    return np.reshape(RR,(9))
