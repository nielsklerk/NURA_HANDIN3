# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

NBINS = 1000

def readfile(filename):
    """
    Helper function to read in the satellite galaxy data from the provided text files.

    Parameters
    ----------
    filename : str
        The name of the file to read in.

    Returns
    -------
    radius : ndarray
        The virial radius for all the satellites in the file.
    nhalo : int
        The number of halos in the file.
    """
    f = open(filename, "r")
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != "#":
            radius.append(float(line.split()[0]))

    radius = np.array(radius, dtype=float)
    f.close()
    return (
        radius,
        nhalo,
    )  # Return the virial radius for all the satellites in the file, and the number of halos


def n(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    # Create x and n arrays
    x = np.asarray(x)
    n = np.zeros_like(x)

    # Function is only defined for positive x
    n[x>0] = A * Nsat * (x[x>0] / b) ** (a-3) * np.exp(-(x[x>0] / b) ** c)

    return n

def romberg_integrator(func: callable, bounds: tuple, order: int = 10, err: bool = False, args: tuple = ()
) -> float:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    a, b = bounds
    h = b - a

    r = np.zeros(order)
    r[0] = 0.5 * h * (func(b, *args) + func(a, *args))
    N_p = 1
    # Create the list of starting estimates
    for i in range(1, order):
        r[i] = 0
        delta = np.copy(h)
        h *= 0.5
        x = a + h
        for _ in range(N_p):
            r[i] += func(x, *args)
            x += delta
        r[i] = 0.5 * (r[i-1] + delta * r[i])
        N_p *= 2

    N_p = 1
    # Iteratively improve the estimates
    for i in range(1, order):
        N_p *= 4
        r[:order - i] = (N_p * r[1:order - i + 1] - r[:order - i]) / (N_p - 1)

    # Returns error if needed
    if err:
        return r[0], np.abs(r[0]-r[1])  # (value, error)
    return r[0] # value

# Following the lectures, the function below provides a template for a custom minimization method.
# Depending on your choice of method, you may or may not need to add more function input parameters.
def my_minimizer(
    func: callable, bounds: tuple, tol: float = 1e-5, max_iters: int=100
) -> tuple:
    """
    Custom minimization method.

    Parameters
    ----------
    func : callable
        Function to minimize.
    x_arr : ndarray
        Array of x values to evaluate func at.
    bounds : tuple
        Tuple of (xmin, xmax) to search for minimum in.
    tol : float, optional
        Tolerance for the minimization.
        The default is 1e-5.

    Returns
    -------
    x_min : float
        Value of x at which func is minimum.
    func_min : float
        Minimum value of func.
    """
    def bracketing(bracket):
        w = (1 + np.sqrt(5)) * 0.5
        a = bracket[0] if func(bracket[0]) > func(bracket[1]) else bracket[1]
        b = bracket[0] if bracket[0] != a else bracket[1]
        c = b + (b - a) * w
        for _ in range(max_iters):
            y_a = func(a)
            y_b = func(b)
            y_c = func(c)
            if y_c > y_b:
                return [a, b, c]
            else: 
                term_1 = c * (y_b - y_a)
                term_2 = b * (y_a - y_c)
                term_3 = a * (y_c - y_b)
                d = -0.5 * (term_3 * c + term_2 * b + term_1 * a) / (term_3 + term_2 + term_1)
                y_d = func(d)
                if d > b and d < c:
                    if y_d < y_c:
                        return [b, d, c]
                    elif y_d > y_b:
                        return [a, b, d]
                    else:
                        d = c + (c - b) * w
                else:
                    if np.abs(d - b) > 100 * np.abs(c-b):
                        d = c + (c - b) * w
            a = b
            b = c
            c = d

    w = 2 - (1 + np.sqrt(5)) * 0.5
    a, b, c = bracketing(bounds)
    for _ in range(max_iters):
        if np.abs(c - b) > np.abs(b - a):
            d = b + (c - b) * w
        else:
            d = b + (a - b) * w
        
        if np.abs(c - a) < tol:
            break
    
        if func(d) < func(b):
            if (d > b and d < c) or (d > c and d < b):
                a = b
                b = d
            else:
                c = b
                b = d
        else:
            if (d > b and d < c) or (d > c and d < b):
                c = d
            else:
                a = d
    x = d if func(d) < func(b) else b
    return x, func(x)


#### Fitting ####


def chi2(params: tuple, model: callable, data: np.ndarray,) -> float:
    """
    Calculate the chi-squared for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : tuple
        The parameters to evaluate the model at.

    Returns
    -------
    float
        The chi-squared value for the given parameters and data.
    """
    m = model(*params)
    if not np.all(np.isfinite(m)):
        return np.inf
    return np.sum((m - data)**2/m)


def negative_poisson_ln_likelihood(params: tuple, 
    model: callable, data: np.ndarray
) -> float:
    """
    Calculate the Poisson negative log-likelihood for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : tuple
        The parameters to evaluate the model at.

    Returns
    -------
    float
        The Poisson negative log-likelihood value for the given parameters and data.
    """
    m = model(*params)
    if not np.all(np.isfinite(m)):
        return np.inf

    return -np.sum(data * np.log(m) -  m)  # replace by the correct value


def get_normalization_constant(a: float, b: float, c: float, Nsat: float, x_lower:float, x_upper:float) -> float:
    """
    Calculate the normalization constant A (which is a function of a,b,c) for the satellite number density profile.

    Parameters
    ----------
    a : float
        Small-scale slope.
    b : float
        Transition scale.
    c : float
        Steepness of exponential drop-off.
    Nsat : float
        Average number of satellites.

    Returns
    -------
    float
        Normalization constant A.
    """
    integrand = lambda x: 4 * np.pi * x**2 * n(x, 1, Nsat, a, b, c)
    integral = quad(
        integrand, x_lower, x_upper)[0]
    return Nsat/integral


def minimize_chi2(model: callable, data: np.ndarray, initial_params: tuple) -> tuple:
    """
    Minimize the chi-squared value for a given model and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    initial_params : tuple
        Initial guess for the parameters to minimize over.

    Returns
    -------
    best_params : tuple
        The parameters that minimize the chi-squared value.
    min_chi2 : float
        The minimum chi-squared value achieved.
    """

    # TODO: implement the minimization of chi2 using your custom method. Remember to normalize for each minimization step
    result = minimize(chi2, x0=initial_params, args=(model, data), bounds=[(0, None), (0, None), (0, None)])
    # best_a, best_b, best_c = result.x
    # chi2_min = result.fun
    # best_params = initial_params
    # min_chi2 = chi2(
    #     model, data, initial_params
    # )  # replace by the correct calculation of chi2 for the given parameters

    return result.x, result.fun


def minimize_poisson_ln_likelihood(
    model: callable, data: np.ndarray, initial_params: tuple
) -> tuple:
    """
    Minimize the Poisson negative log-likelihood for a given model and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    initial_params : tuple
        Initial guess for the parameters to minimize over.

    Returns
    -------
    best_params : tuple
        The parameters that minimize the Poisson negative log-likelihood value.
    min_ln_likelihood : float
        The minimum Poisson negative log-likelihood value achieved.
    """

    # TODO: implement the minimization of the Poisson negative log-likelihood using your custom method. Remember to normalize for each minimization step

    result = minimize(negative_poisson_ln_likelihood, x0=initial_params, args=(model, data), bounds=[(0, None), (0, None), (0, None)])

    return result.x, result.fun

def model(a, b, c, bin_edges, Nsat, x_lower, x_upper):
    N_i = []
    A = get_normalization_constant(a, b, c, Nsat, x_lower, x_upper)
    integrand = lambda x: 4 * np.pi * x**2 * n(x, A, Nsat, a, b, c)
    for i in range(len(bin_edges)-1):
        x_i, x_j = bin_edges[i], bin_edges[i+1]
        integral = quad(integrand, x_i, x_j)[0]
        N_i.append(integral)
    return np.array(N_i)

def G_score(data, model):
    return 2 * np.sum(data[data>0] * (np.log(data[data>0]) - np.log(model[data>0])))

def Q_score(G_score, dof):
    from scipy.special import gammainc
    return 1 - gammainc(dof/2, G_score/2)

def rng(N: int) -> np.ndarray:
    """
    Random number generator 

    Parameters
    ----------
    N: int
        Number of random numbers

    Returns
    -------
    np.ndarray
        Array containing N random numbers
        if N=1 returns float instead
    """
    global seed
    seed = np.uint64(seed)

    # Parameters for rng
    a = np.uint64(4294957665)
    a_1 = np.uint64(21)
    a_2 = np.uint64(35)
    a_3 = np.uint64(4)

    rnds = np.zeros(N)
    for i in range(N):

        # MWC 32-bit
        seed &= np.uint64(2**32 - 1)
        seed = a*(seed & np.uint64(2**32 - 1)) + (seed >> np.uint64(32))
        seed &= np.uint64(2**32 - 1)

        # 64-bit XOR-shift
        seed ^= (seed>>a_1)
        seed ^= (seed<<a_2)
        seed ^= (seed>>a_3)

        # Calculating the float
        u = seed / np.float64(2**64)
        rnds[i] = u
    
    if N==1:
        return rnds[0]
    return rnds


def sampler(
    dist: callable,
    min: float,
    max: float,
    Nsamples: int,
    args: tuple = (),
) -> np.ndarray:
    """
    Sample a distribution using rejection sampling

    Parameters
    dist : callable
        Distribution to sample
    min :
        Minimum value for sampling
    max : float
        Maximum value for sampling
    Nsamples : int
        Number of samples
    args : tuple, optional
        Arguments of the distribution to sample, passed as args to dist

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """
    xx = np.linspace(min, max, 10000)
    pdf = dist(xx, *args)
    cdf = np.array([np.sum(pdf[:i+1]) for i in range(len(pdf))])
    cdf /= np.max(cdf)
    # plt.figure(figsize=(6.4, 4.8))
    # plt.plot(xx, cdf)
    # plt.xscale('log')
    # plt.savefig("Plots/test2.png")
    random_y_values = rng(Nsamples)
    random_x_values = np.interp(random_y_values, cdf, xx)
    return random_x_values

# =====================================================
# ======== Main functions for each subquestion ========
# =====================================================


def do_question_1a():
    print('1a')
    return
    # ======== Question 1a: Maximization of N(x) ========
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A_1a = 256 / (5 * np.pi ** (3 / 2))
    x_lower, x_upper = 10**-4, 5

    x_max, Nx_max = my_minimizer(lambda x: -4*np.pi*x**2*n(x, A=A_1a, Nsat=Nsat, a=a, b=b, c=c), (x_lower, x_upper))
    Nx_max *= -1

    # Write the results to text files for later use in the PDF
    with open("Calculations/satellite_max_x.txt", "w") as f:
        f.write(f"{x_max:.6f}")
    with open("Calculations/satellite_max_Nx.txt", "w") as f:
        f.write(f"{Nx_max:.6f}")


def do_question_1b():
    print('1b')
    # return
    # ======== Question 1b: Fitting N(x) with chi-squared ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    N_sat = []
    min_chi2_values = []
    global best_params_chi2
    best_params_chi2 = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")
        nsat = len(radius) / nhalo
        x_lower, x_upper = (
            min(radius),
            max(radius),
        )
        bins = np.logspace(np.log10(x_lower), np.log10(x_upper), NBINS)  # choose appropriate bins
        hist = np.histogram(radius, bins=bins)[0]
        N_i = hist / nhalo

        best_params, min_chi2 = minimize_chi2(lambda a, b, c: model(a, b, c, bin_edges=bins, Nsat=nsat, x_lower=x_lower, x_upper=x_upper),
                      N_i,
                      [2.4, .25, 1.6])
        A = get_normalization_constant(*best_params, nsat, x_lower, x_upper)

        # Store N_sat, chi2 values and best-fit parameters in their arrays
        N_sat.append(nsat)
        min_chi2_values.append(min_chi2)
        best_params_chi2.append(best_params)

        # Plot the data and the best-fit model for each data file in a subplot.
        axs[datafiles.index(datafile)].stairs(
        N_i/(bins[1:] - bins[:-1]), edges=bins, fill=True
        )

        x_plot = np.linspace(x_lower, x_upper, 100)
        axs[datafiles.index(datafile)].plot(
            x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *best_params)
        )  # plot the best-fit model using the best-fit parameters found from chi-squared minimization

        # Add labels and title to the subplot
        axs[datafiles.index(datafile)].set_title(f"Data file: {datafile}")
        axs[datafiles.index(datafile)].set_xlabel("x = r / r_virial")
        axs[datafiles.index(datafile)].set_ylabel("Number of satellites")

        # log-log scaling
        axs[datafiles.index(datafile)].set_xscale("log")
        axs[datafiles.index(datafile)].set_yscale("log")

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_chi2.png")

    # Save N_sat, chi2 values and best-fit parameters for each data file to tex files for later use in the PDF
    with open("Calculations/table_fitparams_chi2.tex", "w") as f:
        rows = list(zip(N_sat, min_chi2_values, best_params_chi2))
        for idx, (N, chi2_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {N:.5f} & {chi2_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )


def do_question_1c():
    print('1c')
    # return 
    # ======== Question 1c: Fitting N(x) with Poisson ln-likelihood ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    min_poisson_llh_values = []
    global best_params_poisson
    best_params_poisson = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")
        nsat = len(radius) / nhalo
        x_lower, x_upper = (
            min(radius),
            max(radius),
        )
        bins = np.logspace(np.log10(x_lower), np.log10(x_upper), NBINS)  # choose appropriate bins
        hist = np.histogram(radius, bins=bins)[0]
        N_i = hist / nhalo
        best_params, min_poisson_llh = minimize_poisson_ln_likelihood(lambda a, b, c: model(a, b, c, bin_edges=bins, Nsat=nsat, x_lower=x_lower, x_upper=x_upper),
                      N_i,
                      [2.4, .6, 1.6])
        A = get_normalization_constant(*best_params, nsat, x_lower, x_upper)

        # Store poisson llh values and best-fit parameters in their arrays
        min_poisson_llh_values.append(min_poisson_llh)
        best_params_poisson.append(
            best_params
        )  # replace by the correct best-fit parameters (a,b,c) found from Poisson negative log-likelihood minimization

        # Plot the data and the best-fit model for each data file in a subplot.
        axs[datafiles.index(datafile)].stairs(
        N_i/(bins[1:] - bins[:-1]), edges=bins, fill=True, label="Satellite galaxies"
        )
        x_plot = np.linspace(x_lower, x_upper, 100)
        axs[datafiles.index(datafile)].plot(
            x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *best_params)
        )  # plot the best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization

        # Add labels and title to the subplot
        axs[datafiles.index(datafile)].set_title(f"Data file: {datafile}")
        axs[datafiles.index(datafile)].set_xlabel("x = r / r_virial")
        axs[datafiles.index(datafile)].set_ylabel("Number of satellites")

        # log-log scaling
        axs[datafiles.index(datafile)].set_xscale("log")
        axs[datafiles.index(datafile)].set_yscale("log")

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_poisson.png")

    # Save poisson llh values and best-fit parameters for each data file to text files for later use in the PDF
    with open("Calculations/table_fitparams_poisson.tex", "w") as f:
        rows = list(zip(min_poisson_llh_values, best_params_poisson))
        for idx, (llh_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {llh_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )


def do_question_1d():
    print('1d')
    global best_params_chi2
    global best_params_poisson
    # ======== Question 1d: Statistical tests ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    G_scores_chi2 = []
    Q_scores_chi2 = []

    G_scores_poisson = []
    Q_scores_poisson = []

    for i, datafile in enumerate(datafiles):
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")
        nsat = len(radius) / nhalo
        x_lower, x_upper = (
            min(radius),
            max(radius),
        )
        bins = np.logspace(np.log10(x_lower), np.log10(x_upper), NBINS)  # choose appropriate bins
        hist = np.histogram(radius, bins=bins)[0]

        # Use best-fit parameters from previous steps
        fitted_params_chi2 = best_params_chi2[i]# replace by the correct array
        fitted_params_poisson = best_params_poisson[i]# replace by the correct array
        model_chi2 = model(*fitted_params_chi2, bins, nsat, x_lower, x_upper)*nhalo
        model_poisson = model(*fitted_params_poisson, bins, nsat, x_lower, x_upper)*nhalo

        # TODO: implement the statistical tests to calculate G and Q scores for both chi2 and poisson fits, and store the results in their respective arrays
        G_score_chi2 = G_score(hist, model_chi2)
        Q_score_chi2 = Q_score(G_score_chi2, len(bins) - len(fitted_params_chi2) - 1)
        G_score_poisson = G_score(hist, model_poisson)
        Q_score_poisson = Q_score(G_score_poisson, len(bins) - len(fitted_params_poisson) - 1)

        # Append the G and Q scores for chi2 and poisson fits to their respective arrays
        G_scores_chi2.append(G_score_chi2)
        Q_scores_chi2.append(Q_score_chi2)
        G_scores_poisson.append(G_score_poisson)
        Q_scores_poisson.append(Q_score_poisson)

    # Save G and Q scores for chi2 and poisson fits to tex files for later use in the PDF
    with open("Calculations/statistical_test_table_rows.tex", "w") as f:
        rows = []
        for i, (G, Q) in enumerate(zip(G_scores_chi2, Q_scores_chi2), start=11):
            rows.append(f"$\\chi^2$ & m{i} & {G:.5f} & {Q:.5f}")

        for i, (G, Q) in enumerate(zip(G_scores_poisson, Q_scores_poisson), start=11):
            rows.append(f"Poisson & m{i} & {G:.5f} & {Q:.5f}")

        for idx, row in enumerate(rows):
            if idx < len(rows) - 1:
                f.write(row + " \\\\\n")
            else:
                f.write(row)


def do_question_1e():
    print('1e')
    # Set seed
    global seed
    seed = 31415926535

    global best_params_chi2
    global best_params_poisson
    # ======== Question 1e: Monte Carlo simulations ========
    # pick one of the data files to perform the Monte Carlo simulations on, e.g. m12
    datafiles = ["m11", "m12", "m13", "m14", "m15"]
    index = (
        1  # index of the data file to use for Monte Carlo simulations, e.g. 1 for m12
    )

    radius, nhalo = readfile(f"Data/satgals_{datafiles[index]}.txt")
    nsat = len(radius) / nhalo
    x_lower, x_upper = (
        min(radius),
        max(radius),
    )
    bins = np.logspace(np.log10(x_lower), np.log10(x_upper), NBINS)  # choose appropriate bins
    hist = np.histogram(radius, bins=bins)[0]
    N_i = hist / nhalo

    # Use best-fit parameters from previous steps for the original data file
    best_param_chi2 = best_params_chi2[index]  # replace by the correct array
    best_param_poisson = best_params_poisson[index]  # replace by the correct array

    p_of_x_chi2 = (
        lambda x: 4*np.pi*x**2*n(x, 1, 1, *best_param_chi2))
    p_of_x_poisson = (
        lambda x: 4*np.pi*x**2*n(x, 1, 1, *best_param_poisson))

    # Normalize probability density for rejection sampling
    _, Nx_max_chi2 = my_minimizer(lambda x: -1 * p_of_x_chi2(x), (x_lower, x_upper))
    _, Nx_max_poisson = my_minimizer(lambda x: -1 * p_of_x_poisson(x), (x_lower, x_upper))
    p_of_x_chi2_norm = lambda x: -1 * p_of_x_chi2(x) / Nx_max_chi2
    p_of_x_poisson_norm = lambda x: -1 * p_of_x_poisson(x) / Nx_max_poisson

    pseudo_chi2_params = []
    pseudo_poisson_params = []

    num_pseudo_experiments = 10  # replace by number with reasonable runtime
    for i in range(num_pseudo_experiments):
        random_samples_chi2 = sampler(p_of_x_chi2_norm, min=x_lower, max=x_upper, Nsamples=nhalo, args=())
        random_samples_poisson = sampler(p_of_x_poisson_norm, min=x_lower, max=x_upper, Nsamples=nhalo, args=())
        N_i_chi2 = np.histogram(random_samples_chi2, bins=bins)[0] / nhalo
        # print(N_i_chi2)
        # plt.figure(figsize=(6.4, 4.8))
        # plt.stairs(N_i/(bins[1:] - bins[:-1]), edges=bins, fill=True, alpha=0.5)
        # plt.stairs(N_i_chi2/(bins[1:] - bins[:-1]), edges=bins, fill=True, alpha=.5)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.savefig("Plots/test.png")
        N_i_poisson = np.histogram(random_samples_poisson, bins=bins)[0] / nhalo

        mc_param_chi2, _ = minimize_chi2(lambda a, b, c: model(a, b, c, bin_edges=bins, Nsat=nsat, x_lower=x_lower, x_upper=x_upper),
                      N_i_chi2,
                      [*best_param_chi2])
        mc_param_poisson, _ = minimize_poisson_ln_likelihood(lambda a, b, c: model(a, b, c, bin_edges=bins, Nsat=nsat, x_lower=x_lower, x_upper=x_upper),
                      N_i_poisson,
                      [*best_param_poisson])

        # Append the best-fit parameters for each pseudo-dataset to their respective arrays.
        pseudo_chi2_params.append(
            mc_param_chi2
        )  # replace by the correct best-fit parameters (a,b,c) found from chi-squared minimization for the pseudo-dataset
        pseudo_poisson_params.append(
            mc_param_poisson
        )  # replace by the correct best-fit parameters (a,b,c) found from Poisson negative log-likelihood minimization for the pseudo-dataset

    # plot the pseudo best-fit profiles, plot the original best-fit profile in another color and plot the mean in one more color

    # chi2 plot
    x_plot = np.linspace(x_lower, x_upper, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_chi2_params:
        A = get_normalization_constant(*params, nsat, x_lower, x_upper)
        plt.plot(
            x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *params)
        )  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from chi-squared minimization

    A = get_normalization_constant(*best_param_chi2, nsat, x_lower, x_upper)
    plt.plot(
        x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *best_param_chi2)
    )  # plot the original best-fit model using the best-fit parameters found from chi-squared minimization on the real data

    mean_params_chi2 = np.mean(
        pseudo_chi2_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    A = get_normalization_constant(*mean_params_chi2, nsat, x_lower, x_upper)
    plt.plot(
        x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *mean_params_chi2)
    )  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - chi2 fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-3, None)
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_chi2.png")

    # poisson plot
    x_plot = np.linspace(x_lower, x_upper, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_poisson_params:
        A = get_normalization_constant(*params, nsat, x_lower, x_upper)
        plt.plot(
            x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *params)
        )   # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from Poisson negative log-likelihood minimization
    A = get_normalization_constant(*best_param_poisson, nsat, x_lower, x_upper)
    plt.plot(
        x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *best_param_chi2)
    )  # plot the original best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization on the real data

    mean_params_poisson = np.mean(
        pseudo_poisson_params, axis=0
    )  # calculate the mean of the best-fit parameters from the pseudo-datasets
    A = get_normalization_constant(*mean_params_poisson, nsat, x_lower, x_upper)
    plt.plot(
        x_plot, 4*np.pi*x_plot**2*n(x_plot, A, nsat, *mean_params_poisson)
    )  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - Poisson fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-3, None)
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_poisson.png")


if __name__ == "__main__":
    do_question_1a()
    do_question_1b()
    do_question_1c()
    do_question_1d()
    do_question_1e()
