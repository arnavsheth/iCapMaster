def regularize_cov_matrix(monthly_cov_mat, maxiter=1e5, tol=1e-8):
    # make matrix positive-semidefinite

    std_dev_mat = np.diag(np.sqrt(np.diag(monthly_cov_mat)))
    inv_std_dev_mat = np.linalg.inv(std_dev_mat)
    corr_mat = inv_std_dev_mat @ monthly_cov_mat @ inv_std_dev_mat

    K = len(corr_mat)
    Y = np.copy(corr_mat)
    dS = np.zeros((K, K))
    i = 0

    while i < maxiter:

        R = Y - dS
        eigval, eigvec = np.linalg.eigh(R)
        eigval[eigval < 0] = 0.0
        X = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
        dS = X - R
        Y = np.copy(X)
        np.fill_diagonal(Y, 1.0)

        if np.linalg.norm(X - Y) < tol:
            shrunk_corr = get_correlation_with_shrinkage(Y)
            monthly_cov = np.linalg.multi_dot([std_dev_mat, shrunk_corr, std_dev_mat])
            assert(np.all(np.linalg.eigvals(monthly_cov) > 0))
            return monthly_cov

        i += 1
    return None


def get_correlation_with_shrinkage(cor_mat, delta=1e-5, n=1):

    K = len(cor_mat)
    avg = (np.sum(cor_mat) - K) / (K ** 2 - K)
    target = np.full((K, K), avg)
    np.fill_diagonal(target, 1.0)
    shrinkage = n * delta
    if shrinkage >= 1:
        raise Exception(f"shrinkage value {shrinkage} is large or equal to 1")
    X = (1 - shrinkage) * cor_mat + shrinkage * target
    Y = np.copy(X)
    np.fill_diagonal(Y, 1.0)
    assert(np.all(np.linalg.eigvals(Y) > 0))

    return Y