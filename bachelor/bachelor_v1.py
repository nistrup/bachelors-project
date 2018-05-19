from scipy.optimize import minimize
import pandas as pd
import numpy as np
import scipy.stats

def main():

    """
    Valg af datasæt:

    1: S&P sectors
    2: Industry portfolios
    3: International portfolios
    4: Mkt/SMB/HML
    5: FF 1-factor
    6: FF 4-factor
    """

    data_selection = 3

    if data_selection == 1:
        selected = ['Energy', 'Material', 'Industrials', 'Cons-Discr.', 'Cons-Staples', 'Helth-Care', 'Financials',
                    'Inf-Tech', 'Telecom', 'Utilities',
                    'SP500']  # Angiv ønskede aktiver fra Quandl eller kolonner fra CSV-ark
        file_name = 'data/10_SP500_Sectors.csv'  # navn og file-exstention på data
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        returns_data = data
    if data_selection == 2:
        selected = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other', 'Mkt']
        file_name = 'data/10_Industry_Portfolios.csv'  # navn og file-exstention på data
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        rf = pd.read_csv(file_name, delimiter=',', header=0, usecols=['RF3'])
        returns_data = np.subtract(data, rf)
    if data_selection == 3:
        selected = ['Canada', 'Japan', 'France', 'Germany', 'Italy', 'Switzerland', 'UK', 'US', 'World']
        file_name = 'data/8_Country_and_World.csv'
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        returns_data = data
    if data_selection == 4:
        selected = ['Mkt-RF', 'SMB', 'HML']
        file_name = 'data/3_MKT-SMB-HML.csv'
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        returns_data = data
    if data_selection == 5:
        selected = ['S--L', 'S--2', 'S--3', 'S--4', 'S--H', '2--L', '2--2', '2--3', '2--4', '2--H', '3--L', '3--2',
                    '3--3',
                    '3--4', '3--H', '4--L', '4--2', '4--3', '4--4', '4--H', 'Mkt-RF']
        file_name = 'data/20_Portfolios_1_Factor.csv'
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        returns_data = data
    if data_selection == 6:
        selected = ['S--L', 'S--2', 'S--3', 'S--4', 'S--H', '2--L', '2--2', '2--3', '2--4', '2--H', '3--L', '3--2',
                    '3--3',
                    '3--4', '3--H', '4--L', '4--2', '4--3', '4--4', '4--H', 'SMB', 'HML', 'MOM', 'Mkt-RF']
        file_name = 'data/20_Portfolios_4_Factor.csv'
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        returns_data = data

    global cov_matrix, returns_mean
    global N, M, a, T

    N = len(selected)
    M = 120
    T = len(data)
    a = (1 / 2) * (1 / N)
    gamma_1 = True

    b = (0, 1)
    b_g = (a, 1)
    bnds = []
    bnds_g = []

    port_returns_naive = []
    port_returns_mv_oos = []
    port_returns_mv_c = []
    port_returns_min = []
    port_returns_min_c = []
    port_returns_bs = []
    port_returns_bs_c = []
    port_returns_g_min_c = []

    stock_weights_naive = []
    stock_weights_mv_is = []
    stock_weights_mv_oos = []
    stock_weights_mv_c = []
    stock_weights_min = []
    stock_weights_min_c = []
    stock_weights_bs = []
    stock_weights_bs_c = []
    stock_weights_g_min_c = []

    def objective_min(x):
        return np.dot(x, np.dot(cov_matrix, x))

    def objective_umax(x):
        if gamma_1:
            gamma = 1
        else:
            gamma = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))
        return -1 * (np.dot(x, returns_mean) - (gamma / 2) * np.dot(x, np.dot(cov_matrix, x)))

    def objective_bs_c(x):
        if gamma_1:
            gamma = 1
        else:
            gamma = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))
        return -1 * (np.dot(x, mubs) - (gamma / 2) * np.dot(x, np.dot(cov_matrix, x)))

    def constraint(x):
        x_sum = 0
        for i in range(0, N):
            x_sum = x_sum + x[i]
        return x_sum - 1

    def zcal(a1, b1):

        cov_what = np.cov(a1, b1)

        mu_a1 = sum(a1) / len(a1)
        mu_b1 = sum(b1) / len(b1)

        sd_a1 = np.sqrt(cov_what[0, 0])
        sd_b1 = np.sqrt(cov_what[1, 1])

        element1 = 2 * ((sd_a1 ** 2) * (sd_b1 ** 2))
        element2 = -2 * (sd_a1 * sd_b1 * cov_what[0, 1])
        element3 = (1 / 2) * ((mu_a1 ** 2) * (sd_b1 ** 2))
        element4 = (1 / 2) * ((mu_b1 ** 2) * (sd_a1 ** 2))
        element5 = -1 * ((mu_a1 * mu_b1) / (sd_a1 * sd_b1)) * cov_what[0, 1] ** 2

        teta = (1 / (T - M)) * (element1 + element2 + element3 + element4 + element5)
        z = (sd_a1 * mu_b1 - sd_b1 * mu_a1) / np.sqrt(teta)

        return z

    for i in range(0, N):
        bnds.append(b)
        bnds_g.append(b_g)

    bnds = tuple(bnds)
    bnds_g = tuple(bnds_g)

    con = {'type': 'eq', 'fun': constraint}

    # ---------------------------------------- NAIVE -----------------------------------------------------

    for month in range(0, len(returns_data) - M):
        weights = np.ones(N) * (1 / N)
        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_naive.append(returns)
        stock_weights_naive.append(weights)

    sharpe_naive = (pd.DataFrame(port_returns_naive).mean() / pd.DataFrame(port_returns_naive).std()).iloc[0]

    # ---------------------------------------- NAIVE -----------------------------------------------------

    # --------------------------------- MV IN SAMPLE -----------------------------------------------------

    new_returns_data = returns_data[M:T]
    returns_monthly = new_returns_data
    returns_mean = returns_monthly.mean()
    cov_matrix = returns_monthly.cov()

    t = np.dot(np.linalg.inv(cov_matrix), returns_mean)
    n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))
    weights = t / n

    returns_mv_is = np.dot(weights, returns_mean)
    var_mv_is = np.dot(weights, np.dot(cov_matrix, weights))
    stock_weights_mv_is.append(weights)

    sharpe_mv_is = returns_mv_is / np.sqrt(var_mv_is)

    # --------------------------------- MV IN SAMPLE -----------------------------------------------------

    # --------------------------------- MV OUT OF SAMPLE -------------------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        t = np.dot(np.linalg.inv(cov_matrix), returns_mean)
        n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))
        weights = t / n

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_mv_oos.append(returns)
        stock_weights_mv_oos.append(weights)

    sharpe_mv_oos = (pd.DataFrame(port_returns_mv_oos).mean() / pd.DataFrame(port_returns_mv_oos).std()).iloc[0]

    # --------------------------- MV OUT OF SAMPLE -------------------------------------------------------

    # --------------------------------------- MV SHORTSALE CONSTRAINT ------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        wguess = np.ones(N)
        solmin = minimize(objective_umax, wguess, method='SLSQP', bounds=bnds, constraints=con)
        weights = solmin.x

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_mv_c.append(returns)
        stock_weights_mv_c.append(weights)

    sharpe_mv_c = (pd.DataFrame(port_returns_mv_c).mean() / pd.DataFrame(port_returns_mv_c).std()).iloc[0]

    # -------------------------------------- MV SHORTSALE CONSTRAINT -------------------------------------

    # -------------------------------------- MIN VARIANCE ------------------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        t = np.dot(np.linalg.inv(cov_matrix), np.ones(N))
        n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), np.ones(N)))
        weights = t / n

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_min.append(returns)
        stock_weights_min.append(weights)

    sharpe_min = (pd.DataFrame(port_returns_min).mean() / pd.DataFrame(port_returns_min).std()).iloc[0]

    # ------------------------------------------- MINIMUM VARIANCE ---------------------------------------

    # ------------------------------------------ MIN - C -------------------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        wguess = np.ones(N)
        solmin = minimize(objective_min, wguess, method='SLSQP', bounds=bnds, constraints=con)
        weights = solmin.x

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_min_c.append(returns)
        stock_weights_min_c.append(weights)

    sharpe_min_c = (pd.DataFrame(port_returns_min_c).mean() / pd.DataFrame(port_returns_min_c).std()).iloc[0]

    # ----------------------------------------------- MIN - C --------------------------------------------

    # ----------------------------------- G - MIN - C ----------------------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        wguess = np.ones(N)
        solmin = minimize(objective_min, wguess, method='SLSQP', bounds=bnds_g, constraints=con)
        weights = solmin.x

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_g_min_c.append(returns)
        stock_weights_g_min_c.append(weights)

    sharpe_g_min_c = (pd.DataFrame(port_returns_g_min_c).mean() / pd.DataFrame(port_returns_g_min_c).std()).iloc[0]

    # -------------------------- G-MIN-C -----------------------------------------------------------------

    # ------------------------------------------- BS -----------------------------------------------------

    for month in range(0, len(returns_data) - M):
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        tmin = np.dot(np.linalg.inv(cov_matrix), np.ones(N))
        nmin = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), np.ones(N)))
        wmin = tmin / nmin

        mumin = np.dot(wmin.T, returns_mean)
        muny = np.subtract(returns_mean, np.ones(N) * mumin)
        element1 = np.dot(np.transpose(muny), np.dot(np.linalg.inv(cov_matrix), muny))
        Phi = (N + 2) / ((N + 2) + M * element1)
        mubs = (1 - Phi) * returns_mean + Phi * mumin
        tbs = np.dot(np.linalg.inv(cov_matrix), mubs)
        nbs = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), mubs))
        weights = tbs / nbs

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_bs.append(returns)
        stock_weights_bs.append(weights)

    sharpe_bs = (pd.DataFrame(port_returns_bs).mean() / pd.DataFrame(port_returns_bs).std()).iloc[0]

    # ------------------------------------------- BS -----------------------------------------------------

    # ------------------------------------------ BS - C  -------------------------------------------------

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        tmin = np.dot(np.linalg.inv(cov_matrix), np.ones(N))
        nmin = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), np.ones(N)))
        wmin = tmin / nmin

        mumin = np.dot(wmin.T, returns_mean)
        muny = np.subtract(returns_mean, np.ones(N) * mumin)
        element1 = np.dot(np.transpose(muny), np.dot(np.linalg.inv(cov_matrix), muny))
        Phi = (N + 2) / ((N + 2) + M * element1)
        mubs = (1 - Phi) * returns_mean + Phi * mumin
        wguess = np.ones(N)
        solmin = minimize(objective_bs_c, wguess, method='SLSQP', bounds=bnds, constraints=con)
        weights = solmin.x

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_bs_c.append(returns)
        stock_weights_bs_c.append(weights)

    sharpe_bs_c = (pd.DataFrame(port_returns_bs_c).mean() / pd.DataFrame(port_returns_bs_c).std()).iloc[0]

    # ------------------------------------------ BS - C  -------------------------------------------------

    # -------------------------------------- P-VÆRDIER ---------------------------------------------------

    z_mv = zcal(port_returns_mv_oos, port_returns_naive)
    z_mv_c = zcal(port_returns_mv_c, port_returns_naive)
    z_min = zcal(port_returns_min, port_returns_naive)
    z_min_c = zcal(port_returns_min_c, port_returns_naive)
    z_g_min_c = zcal(port_returns_g_min_c, port_returns_naive)
    z_bs = zcal(port_returns_bs, port_returns_naive)
    z_bs_c = zcal(port_returns_bs_c, port_returns_naive)

    if (z_mv >= 0):
        p_mv = 1 - scipy.stats.norm(0, 1).cdf(z_mv)
    else:
        p_mv = scipy.stats.norm(0, 1).cdf(z_mv)

    if (z_mv_c >= 0):
        p_mv_c = 1 - scipy.stats.norm(0, 1).cdf(z_mv_c)
    else:
        p_mv_c = scipy.stats.norm(0, 1).cdf(z_mv_c)

    if (z_min >= 0):
        p_min = 1 - scipy.stats.norm(0, 1).cdf(z_min)
    else:
        p_min = scipy.stats.norm(0, 1).cdf(z_min)

    if (z_min_c >= 0):
        p_min_c = 1 - scipy.stats.norm(0, 1).cdf(z_min_c)
    else:
        p_min_c = scipy.stats.norm(0, 1).cdf(z_min_c)

    if (z_g_min_c >= 0):
        p_g_min_c = 1 - scipy.stats.norm(0, 1).cdf(z_g_min_c)
    else:
        p_g_min_c = scipy.stats.norm(0, 1).cdf(z_g_min_c)

    if (z_bs >= 0):
        p_bs = 1 - scipy.stats.norm(0, 1).cdf(z_bs)
    else:
        p_bs = scipy.stats.norm(0, 1).cdf(z_bs)

    if (z_bs_c >= 0):
        p_bs_c = 1 - scipy.stats.norm(0, 1).cdf(z_bs_c)
    else:
        p_bs_c = scipy.stats.norm(0, 1).cdf(z_bs_c)

    # -------------------------------------- P-VÆRDIER -------------------------------------------------

    print('')
    print('P-værdier:')

    print('p-værdi for mv      = ' + str(round(p_mv, 3)) + ' vs tabelværdi = ' + str(0.17))
    print('p-værdi for bs      = ' + str(round(p_bs, 3)) + ' vs tabelværdi = ' + str(0.19))
    print('p-værdi for min     = ' + str(round(p_min, 3)) + ' vs tabelværdi = ' + str(0.30))
    print('p-værdi for mv-c    = ' + str(round(p_mv_c, 3)) + ' vs tabelværdi = ' + str(0.03))
    print('p-værdi for bs-c    = ' + str(round(p_bs_c, 3)) + ' vs tabelværdi = ' + str(0.06))
    print('p-værdi for min-c   = ' + str(round(p_min_c, 3)) + ' vs tabelværdi = ' + str(0.41))
    print('p-værdi for g-min-c = ' + str(round(p_g_min_c, 3)) + ' vs tabelværdi = ' + str(0.31))

    # ---------------------------------------- SHARPES -------------------------------------------------------

    print('')
    print('Sharpe-værdier:')

    print("1/N            = " + str(sharpe_naive))
    print("mv (in sample) = " + str(sharpe_mv_is))
    print("mv             = " + str(sharpe_mv_oos))
    print("bs             = " + str(sharpe_bs))
    print("min            = " + str(sharpe_min))
    print("mv-c           = " + str(sharpe_mv_c))
    print("bs-c           = " + str(sharpe_bs_c))
    print("min-c          = " + str(sharpe_min_c))
    print("g-min-c        = " + str(sharpe_g_min_c))

    # ---------------------------------------- VÆGTNINGER -------------------------------------------------------

    print('')
    print('Min og max vægtninger:')

    print("1/N            = " + str(np.min(stock_weights_naive)) +
          "\n               = " + str(np.max(stock_weights_naive)))
    print("mv (in sample) = " + str(np.min(stock_weights_mv_is)) +
          "\n               = " + str(np.max(stock_weights_mv_is)))
    print("mv             = " + str(np.min(stock_weights_mv_oos)) +
          "\n               = " + str(np.max(stock_weights_mv_oos)))
    print("bs             = " + str(np.min(stock_weights_bs)) +
          "\n               = " + str(np.max(stock_weights_bs)))
    print("min            = " + str(np.min(stock_weights_min)) +
          "\n               = " + str(np.max(stock_weights_min)))
    print("mv-c           = " + str(np.min(stock_weights_mv_c)) +
          "\n               = " + str(np.max(stock_weights_mv_c)))
    print("bs-c           = " + str(np.min(stock_weights_bs_c)) +
          "\n               = " + str(np.max(stock_weights_bs_c)))
    print("min-c          = " + str(np.min(stock_weights_min_c)) +
          "\n               = " + str(np.max(stock_weights_min_c)))
    print("g-min-c        = " + str(np.min(stock_weights_g_min_c)) +
          "\n               = " + str(np.max(stock_weights_g_min_c)))

main()