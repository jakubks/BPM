import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_scalar_kernel(features_i, features_j, gamma):

    ''' features_i is (n_bonds(I) x n_features x n_i) - TEST features
        features_j is (n_bonds(J) x n_features x n_j)
        projections is (n_bonds(I) x n_i)

    '''

    n_i = np.shape(features_i)[2]
    n_j = np.shape(features_j)[2]
    n_features = np.shape(features_i)[1]
    nbonds = np.shape(features_i)[0]

    kernel = np.zeros((n_i, n_j))

    for ii in range(n_i):
        for jj in range(n_j):

            f_i = features_i[:, :, ii] #(n_bonds(I) x n_features)
            f_j = features_j[:, :, jj] #(n_bonds(J) x n_features)

            x_i = np.repeat(f_i[:, np.newaxis, :], nbonds, axis=1) # (n_bonds(I) x n_bonds(J) x n_features)
            x_j = np.repeat(f_j[np.newaxis, :, :], nbonds, axis=0) # (n_bonds(I) x n_bonds(J) x n_features)

            kspace_dist = np.linalg.norm(x_i-x_j,axis=2) #matrix of distances in kernel space
            kappa = np.exp(-gamma * kspace_dist**2) # (n_bonds(I) x n_bonds(J))

            kernel[ii,jj] = np.sum(kappa)

    return kernel

def get_scalar_weights(features, goals, gamma, lambd = 1e-8):
    '''
    Getting the parameters alpha for KRR
    features is a 3D array of features of size (n_bonds x n_features x n_training_set)
             or a 2D array of size (n_features x n_training_set)
    energies is a 1D array with (n_training_set) size
    gamma is the Gaussian (Laplacian) parameter
    lambd is the regularization parameter
    '''

    K = get_scalar_kernel(features, features, gamma)

    n_training = np.shape(goals)[0] # number of training examples

    inverse = np.linalg.inv(K + lambd * np.identity(n_training))

    weights = np.dot(inverse,goals)

    return weights


def get_tensor_kernel(features_i, features_j, projections, gamma):

    ''' features_i is (n_bonds(I) x n_features x n_i) - TEST features
        features_j is (n_bonds(J) x n_features x n_j)
        projections is (n_bonds(I) x n_i)

    '''

    n_i = np.shape(features_i)[2]
    n_j = np.shape(features_j)[2]
    n_features = np.shape(features_i)[1]
    nbonds = np.shape(features_i)[0]

    kernel = np.zeros((n_i, n_j))

    for ii in range(n_i):
        for jj in range(n_j):

            f_i = features_i[:, :, ii] #(n_bonds(I) x n_features)
            f_j = features_j[:, :, jj] #(n_bonds(J) x n_features)

            x_i = np.repeat(f_i[:, np.newaxis, :], nbonds, axis=1) # (n_bonds(I) x n_bonds(J) x n_features)
            x_j = np.repeat(f_j[np.newaxis, :, :], nbonds, axis=0) # (n_bonds(I) x n_bonds(J) x n_features)

            kspace_dist = np.linalg.norm(x_i-x_j,axis=2) #matrix of distances in kernel space
            kappa = np.exp(-gamma * kspace_dist**2) # (n_bonds(I) x n_bonds(J))

            ppp = projections[:,ii]

            Pi = np.repeat(ppp[:,np.newaxis], nbonds, axis=1)

            kappa_x_proj = np.multiply(kappa,Pi)

            kernel[ii,jj] = np.sum(kappa_x_proj)

    return kernel


def get_tensor_weights(train_features, goals, gamma, projections, epsilon = 1e-8):

    Ntrain = np.shape(train_features)[-1]
    KKernel = np.zeros((Ntrain,Ntrain))
    VVector = np.zeros((Ntrain,1))

    for ijk in range(9):

        KK = get_tensor_kernel(train_features, train_features, projections[:,ijk,:],gamma)

        KKernel = KKernel + KK @ KK

        VVector = VVector + KK @ np.reshape(goals[:,ijk],(Ntrain,1))


    svd = np.linalg.svd(KKernel, compute_uv=False)

    rcond = epsilon / np.max(svd)

    inverse = np.linalg.pinv(KKernel,rcond = rcond)

    alpha = np.dot(inverse,VVector)

    return alpha
