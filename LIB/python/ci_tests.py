# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

from math import isnan

# Implements multiple conditional independence tests handling mixed variables
import numpy as np
import pandas as pd
from fcit import fcit
from pgmpy.estimators.CITests import chi_square
from python.bnutils import conditional_gaussian_likelihood
from scipy.stats import chi2


def fast_conditional_ind_test(data, x, y, z, **kwargs):
    """Fast Conditional Independence Test between two variables X and Y given set of variables Z
    Args:
        data: pandas dataframe of sampled data or numpy data matrix
        x: first node
        y: second node
        z: set of nodes for the conditioning set
    Returns:
        p value for the null hypothesis that x and y are independent
    """
    z = list(z)
    if type(data) is not np.ndarray:
        data = data.to_numpy()
    nodes = kwargs["nodes"]
    onehot_dict = kwargs["onehot_dict"]
    # extract data from data matrix
    data_x = data[:, x]
    data_y = data[:, y]
    # preprocess data for discrete variables
    if nodes[x]["type"] == "disc":
        data_x = onehot_dict[x]
    if nodes[y]["type"] == "disc":
        data_y = onehot_dict[y]
    for i in range(len(z)):
        tmp = data[:, z[i]]
        if nodes[z[i]]["type"] == "disc":
            tmp = onehot_dict[z[i]]
        if i == 0:
            data_z = tmp
            data_z = data_z.reshape(data_z.shape[0], -1)
        else:
            data_z = np.column_stack((data_z, tmp.reshape(tmp.shape[0], -1)))
    data_x = data_x.reshape(data_x.shape[0], -1)
    data_y = data_y.reshape(data_y.shape[0], -1)
    if len(z) == 0:
        p_val = fcit.test(
            data_x,
            data_y,
            discrete=(nodes[x]["type"] == "disc", nodes[x]["type"] == "disc"),
            **kwargs
        )
    else:
        data_z = data_z.reshape(data_z.shape[0], -1)
        p_val = fcit.test(
            data_x,
            data_y,
            data_z,
            discrete=(nodes[x]["type"] == "disc", nodes[x]["type"] == "disc"),
            **kwargs
        )
    # print(x, y, z, p_val)
    if isnan(p_val):
        # if pval is NaN then return 0 to add the edge to the skeleton
        # this edge can be later removed in search-and-score phase if it is incorrect
        return 0
    return p_val


def conditional_gaussian_test(df, x, y, z, **kwargs):
    """Conditional Gaussian Test between the variables X and Y given set of variables Z
    Args:
        df: pandas data frame object for the sampled data
        x: index of first node
        y: index of second node
        z: set of indices for the conditioning set
    Returns:
        p value for the null hypothesis that x and y are independent
    """
    nodes = kwargs["nodes"]
    z_disc = [i for i in z if nodes[i]["type"] == "disc"]
    z_cont = [i for i in z if nodes[i]["type"] == "cont"]

    # first check x ind y given z
    levels = [nodes[node]["num_categories"] for node in z_disc]
    # lx1 is the likelihood of x given z
    lx1, dofx1 = conditional_gaussian_likelihood(df, x, z_disc, z_cont, levels, nodes)

    z_disc_y = z_disc.copy()
    z_cont_y = z_cont.copy()
    z_disc_y.append(y) if nodes[y]["type"] == "disc" else z_cont_y.append(y)
    if nodes[y]["type"] == "disc":
        levels.append(nodes[y]["num_categories"])
    # lx0 is the likelihood of x given {yUz}
    lx0, dofx0 = conditional_gaussian_likelihood(df, x, z_disc_y, z_cont_y, levels, nodes)

    LRx = 2 * (lx0 - lx1)
    DOFx = dofx0 - dofx1 if dofx0 - dofx1 > 0 else 1
    pval_x = chi2.sf(LRx, DOFx)

    # now check y ind x given z
    levels = [nodes[node]["num_categories"] for node in z_disc]
    # ly1 is the likelihood of y given z
    ly1, dofy1 = conditional_gaussian_likelihood(df, y, z_disc, z_cont, levels, nodes)

    z_disc_x = z_disc.copy()
    z_cont_x = z_cont.copy()
    z_disc_x.append(x) if nodes[x]["type"] == "disc" else z_cont_x.append(x)
    if nodes[x]["type"] == "disc":
        levels.append(nodes[x]["num_categories"])
    # ly0 is the likelihood of y given {xUz}
    ly0, dofy0 = conditional_gaussian_likelihood(df, y, z_disc_x, z_cont_x, levels, nodes)

    LRy = 2 * (ly0 - ly1)
    DOFy = dofy0 - dofy1 if dofy0 - dofy1 > 0 else 1
    pval_y = chi2.sf(LRy, DOFy)

    return min(pval_x, pval_y)


def chi_square_test(df, X, Y, Z, **kwargs):
    """Chi-square conditional independence test between two discrete variables X and Y given set of discrete variables Z
    Args:
        data: pandas dataframe of sampled data or numpy data matrix
        x: first node
        y: second node
        z: set of nodes for the conditioning set
    Returns:
        p value for the null hypothesis that x and y are independent
    """
    # convert column names to string
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    Df = df
    Df.columns = Df.columns.astype(str)
    Z_str = [str(z) for z in Z]
    return chi_square(str(X), str(Y), Z_str, Df)[1]
