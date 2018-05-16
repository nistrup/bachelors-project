from scipy.optimize import minimize
import quandl
import pandas as pd
import numpy as np
import scipy.stats


def main():
################################ SETUP AF DATA ###############################################


    source = 'csv' # quandl eller csv
    data_interval = 'monthly' # daily, monthy eller yearly - Hvis Quandl = 'daily'
    selected = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other',
                'Mkt']  # Angiv ønskede aktiver fra Quandl eller kolonner fra CSV-ark
    selected1 = ['RF3']  # Angiv ønskede aktiver fra Quandl eller kolonner fra CSV-ark

    # Hvis CVS:
    file_name = '10 Industry Portfolios - Average Value Weighted Returns.csv' # navn og file-exstention på data

    # Hvis Quandl
    # date_range = ['2014-1-1', '2016-12-31'] # dato fra og med - dato til og med, format YYYY-MM-DD

    # Træk af data fra Quandl
    if source == 'quandl':
        quandl.ApiConfig.api_key = "yTPaspmH6wqs9rAdSdmk"
        data = quandl.get_table('WIKI/PRICES', ticker=selected, qopts={'columns': ['date', 'ticker', 'adj_close']},
                                date={'gte': date_range[0], 'lte': date_range[1]}, paginate=True)
        clean = data.set_index('date')
        table = clean.pivot(columns='ticker')
        returns_data = table.pct_change()

    # Træk af data fra CSV-fil (allerede procent-vis ændring)
    elif source == 'csv':
        data = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected)
        data1 = pd.read_csv(file_name, delimiter=',', header=0, index_col='date', usecols=['date'] + selected1)
        returns_data = np.subtract(data, data1)

    actives = list(returns_data.columns.values)
    returns_data.head()

    if data_interval == 'monthly':  # "1/N opgaven" tager udgangspunkt i månedlig data og sharpe
        returns_monthly = returns_data
        returns_mean = returns_monthly.mean()

        cov_matrix = returns_monthly.cov()

    elif data_interval == 'daily':  # primært brugt til data fra Quandl giver årlig sharpe
        returns_daily = returns_data
        returns_mean = returns_daily.mean() * 252

        cov_matrix = returns_daily.cov() * 252

    global N
    global M
    global a
    global T
    T = 497
    N = 11
    M = 120
    a = (1/2)*(1/N)
################################## SETUP AF DATA ###############################################

############################# SETUP OBJECTIVE FUNCTIONS SHORTSALE CONSTRAINTS #########################################

    global cov_matrix_min_c
    global cov_matrix_bs_c
    global cov_matrix_g_min_c
    global cov_matrix_mv_c

    global returns_mean_min_c
    global returns_mean_bs_c
    global returns_mean_g_min_c
    global returns_mean_mv_c

    def objective_min_c(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        return np.dot(x, np.dot(cov_matrix_min_c, x))

    def objective_g_min_c(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        return np.dot(x, np.dot(cov_matrix_g_min_c, x))

    def objective_bs_c(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        gamma = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix_bs_c), returns_mean_bs_c))

        return -1 * (np.dot(x, returns_mean_bs_c) - (gamma / 2) * np.dot(x, np.dot(cov_matrix_bs_c, x)))


    def objective_mv_c(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        gamma = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix_mv_c), returns_mean_mv_c))

        return -1 * (np.dot(x, returns_mean_mv_c) - (gamma / 2) * np.dot(x, np.dot(cov_matrix_mv_c, x)))

    def constraint(x):
        return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] - 1

    b = (0, 1)
    b_g_min_c = (a, 1)
    bnds = (b, b, b, b, b, b, b, b, b, b, b)
    bnds_g_min_c =(b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c, b_g_min_c)
    con = {'type': 'eq', 'fun': constraint}
    cons = [con]


############################# SETUP OBJECTIVE FUNCTIONS SHORTSALE CONSTRAINTS #########################################

################################# SETUP AF RETURN VEKTORER OG SHARP ###################################################

    port_returns_naiv = []
    port_returns_mv_is = []
    port_returns_mv_oos = []
    port_returns_mv_c = []
    port_returns_min = []
    port_returns_min_c = []
    port_returns_bs = []
    port_returns_bs_c = []
    port_returns_g_min_c = []

    sharpe_naiv = []
    sharpe_mv_is = []
    sharpe_mv_oos = []
    sharpe_mv_c = []
    sharpe_min = []
    sharpe_min_c = []
    sharpe_bs = []
    sharpe_bs_c = []
    sharpe_g_min_c = []

################################# SETUP AF RETURN VEKTORER ########################################################

#######################################    N A I V   ########################################################

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        weights = np.ones(N)*(1/N)

        # udregner afkast for måned T + 1 med de udregnede optimale vægte
        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_naiv.append(returns)

    mu_naiv = sum(port_returns_naiv)/len(port_returns_naiv)

    var = 0

    for i in range(0, len(port_returns_naiv)):
        var = var + (mu_naiv - port_returns_naiv[i])**2
    var = var/len(port_returns_naiv)

    sd_naiv = np.sqrt(var)

    sharpe_naiv = mu_naiv/np.sqrt(var)


#######################################    N A I V   ########################################################

###################################### MV IN SAMPLE #########################################################

    # Danner nyt data interval, returns og cov-matrix for hvert interval
    new_returns_data = returns_data[M:T]
    returns_monthly = new_returns_data
    returns_mean = returns_monthly.mean()
    cov_matrix = returns_monthly.cov()

    t = np.dot(np.linalg.inv(cov_matrix), returns_mean)
    n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))

    weights = t/n
    returns = np.dot(weights, returns_mean)
    port_returns_mv_is.append(returns)

    var = np.dot(weights, np.dot(cov_matrix, weights))


    mu_mv_is = sum(port_returns_mv_is) / len(port_returns_mv_is)

    sd_mv_is = np.sqrt(var)

    sharpe_mv_is = mu_mv_is / np.sqrt(var)

###################################### MV IN SAMPLE #########################################################

###################################### MV OUT OF SAMPLE #########################################################

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        t = np.dot(np.linalg.inv(cov_matrix), returns_mean)
        n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), returns_mean))

        weights = t / n

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_mv_oos.append(returns)

    mu_mv_oos = sum(port_returns_mv_oos) / len(port_returns_mv_oos)

    var = 0

    for i in range(0, len(port_returns_mv_oos)):
        var = var + (mu_mv_oos - port_returns_mv_oos[i]) ** 2
    var = var / len(port_returns_mv_oos)

    sd_mv_oos = np.sqrt(var)

    sharpe_mv_oos = mu_mv_oos / np.sqrt(var)

###################################### MV OUT OF SAMPLE #########################################################

###################################### MV SHORTSALE CONSTRAINT ######################################################


    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean_mv_c = returns_monthly.mean()
        cov_matrix_mv_c = returns_monthly.cov() * (1 / (M - N - 2))

        wguess = np.ones(11)

        solmin = minimize(objective_mv_c, wguess, method='SLSQP', bounds=bnds, constraints=cons)

        weights = solmin.x

        # udregner afkast for måned T + 1 med de udregnede optimale vægte
        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_mv_c.append(returns)


    mu_mv_c = sum(port_returns_mv_c) / len(port_returns_mv_c)

    var = 0

    for i in range(0, len(port_returns_mv_c)):
        var = var + (mu_mv_c - port_returns_mv_c[i]) ** 2
    var = var / len(port_returns_mv_c)

    sd_mv_c = np.sqrt(var)

    sharpe_mv_c = mu_mv_c / np.sqrt(var)

###################################### MV SHORTSALE CONSTRAINT ######################################################

######################################### MINIMUM VARIANS ################################################


    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        t = np.dot(np.linalg.inv(cov_matrix), np.ones(N))
        n = np.dot(np.ones(N), np.dot(np.linalg.inv(cov_matrix), np.ones(11)))

        weights = t / n

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_min.append(returns)

    mu_min = sum(port_returns_min) / len(port_returns_min)

    var = 0

    for i in range(0, len(port_returns_min)):
        var = var + (mu_min - port_returns_min[i]) ** 2
    var = var / len(port_returns_min)

    sd_min = np.sqrt(var)

    sharpe_min = mu_min / np.sqrt(var)

######################################### MINIMUM VARIANS ################################################

############################ MINIMUM VARIANS SHORTSALE CONSTRAINT #########################################

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean_min_c = returns_monthly.mean()
        cov_matrix_min_c = returns_monthly.cov() * (1 / (M - N - 2))

        wguess = np.ones(11)

        solmin = minimize(objective_min_c, wguess, method='SLSQP', bounds=bnds, constraints=cons)

        weights = solmin.x

        # udregner afkast for måned T + 1 med de udregnede optimale vægte
        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_min_c.append(returns)

    mu_min_c = sum(port_returns_min_c) / len(port_returns_min_c)

    var = 0

    for i in range(0, len(port_returns_min_c)):
        var = var + (mu_min_c - port_returns_min_c[i]) ** 2
    var = var / len(port_returns_min_c)

    sd_min_c = np.sqrt(var)

    sharpe_min_c = mu_min_c / np.sqrt(var)

############################ MINIMUM VARIANS SHORTSALE CONSTRAINT #########################################

#################### GENERALISERET MINIMUM VARIANS SHORTSALE CONSTRAINT #########################################

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean_g_min_c = returns_monthly.mean()
        cov_matrix_g_min_c = returns_monthly.cov() * (1 / (M - N - 2))

        wguess = np.ones(11)

        solmin = minimize(objective_g_min_c, wguess, method='SLSQP', bounds=bnds_g_min_c, constraints=cons)

        weights = solmin.x

        # udregner afkast for måned T + 1 med de udregnede optimale vægte
        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_g_min_c.append(returns)

    mu_g_min_c = sum(port_returns_g_min_c) / len(port_returns_g_min_c)

    var = 0

    for i in range(0, len(port_returns_g_min_c)):
        var = var + (mu_g_min_c - port_returns_g_min_c[i]) ** 2
    var = var / len(port_returns_g_min_c)

    sd_g_min_c = np.sqrt(var)

    sharpe_g_min_c = mu_g_min_c / np.sqrt(var)

#################### GENERALISERET MINIMUM VARIANS SHORTSALE CONSTRAINT #########################################

###############################       B S       #########################################

    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean = returns_monthly.mean()
        cov_matrix = returns_monthly.cov()

        tmin = np.dot(np.linalg.inv(cov_matrix), np.ones(11))

        nmin = np.dot(np.ones(11), np.dot(np.linalg.inv(cov_matrix), np.ones(11)))

        wmin = tmin / nmin

        mumin = np.dot(wmin.T, returns_mean)

        muny = np.subtract(returns_mean, np.ones(11) * mumin)

        element1 = np.dot(np.transpose(muny), np.dot(np.linalg.inv(cov_matrix), muny))

        Phi = (N + 2) / ((N + 2) + M * element1)

        mubs = (1 - Phi) * returns_mean + Phi * mumin

        tbs = np.dot(np.linalg.inv(cov_matrix), mubs)

        nbs = np.dot(np.ones(11), np.dot(np.linalg.inv(cov_matrix), mubs))

        weights = tbs / nbs

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_bs.append(returns)


    mu_bs = sum(port_returns_bs) / len(port_returns_bs)

    var = 0

    for i in range(0, len(port_returns_bs)):
        var = var + (mu_bs - port_returns_bs[i]) ** 2
    var = var / len(port_returns_bs)

    sd_bs = np.sqrt(var)

    sharpe_bs = mu_bs / np.sqrt(var)

###############################       B S       #########################################

###############################   B S SHORTSALE CONSTRAINT   #########################################


    for month in range(0, len(returns_data) - M):
        # Danner nyt data interval, returns og cov-matrix for hvert interval
        new_returns_data = returns_data[month:month + M]
        returns_monthly = new_returns_data
        returns_mean_bs_c = returns_monthly.mean()
        cov_matrix_bs_c = returns_monthly.cov()

        tmin = np.dot(np.linalg.inv(cov_matrix_bs_c), np.ones(11))

        nmin = np.dot(np.ones(11), np.dot(np.linalg.inv(cov_matrix_bs_c), np.ones(11)))

        wmin = tmin / nmin

        mumin = np.dot(wmin.T, returns_mean_bs_c)

        muny = np.subtract(returns_mean_bs_c, np.ones(11) * mumin)

        element1 = np.dot(np.transpose(muny), np.dot(np.linalg.inv(cov_matrix_bs_c), muny))

        Phi = (N + 2) / ((N + 2) + M * element1)

        mubs = (1 - Phi) * returns_mean_bs_c + Phi * mumin

        wguess = np.ones(11)

        solmin = minimize(objective_bs_c, wguess, method='SLSQP', bounds=bnds, constraints=cons)

        weights = solmin.x

        returns = np.dot(weights, returns_data[month + M:month + M + 1].mean())
        port_returns_bs_c.append(returns)


    mu_bs_c = sum(port_returns_bs_c) / len(port_returns_bs_c)

    var = 0

    for i in range(0, len(port_returns_bs_c)):
        var = var + (mu_bs_c - port_returns_bs_c[i]) ** 2
    var = var / len(port_returns_bs_c)

    sd_bs_c = np.sqrt(var)

    sharpe_bs_c = mu_bs_c / np.sqrt(var)


###############################   B S SHORTSALE CONSTRAINT   #########################################


    print('sharpe naiv = ' + str(sharpe_naiv))
    print('sharpe_mv_is = ' + str(sharpe_mv_is))
    print('sharpe_mv_oos = ' + str(sharpe_mv_oos))
    print('sharpe_mv_c = ' + str(sharpe_mv_c))
    print('min = ' + str(sharpe_min))
    print('min_c = ' + str(sharpe_min_c))
    print('g_min_c = ' + str(sharpe_g_min_c))
    print('bs = ' + str(sharpe_bs))
    print('bs_c = ' + str(sharpe_bs_c))


    def zbro(a1, b1):

        cov_what = np.cov(a1, b1)

        mu_a1 = sum(a1)/len(a1)

        mu_b1 = sum(b1)/len(b1)

        sd_a1 = np.sqrt(cov_what[0, 0])
        sd_b1 = np.sqrt(cov_what[1, 1])

        element1 = 2*((sd_a1**2)*(sd_b1**2))
        element2 = -2*(sd_a1*sd_b1*cov_what[0, 1])
        element3 = (1/2)*((mu_a1**2)*(sd_b1**2))
        element4 = (1/2)*((mu_b1**2)*(sd_a1**2))
        element5 = -1*((mu_a1*mu_b1)/(sd_a1*sd_b1))*(cov_what[0, 1]**2)

        teta = (1/(T-M))*(element1 + element2 + element3 + element4 + element5)

        z = (sd_a1*mu_b1-sd_b1*mu_a1)/np.sqrt(teta)

        return z


############################### P-VÆRDIER BABY #####################################################

    z_mv = zbro(port_returns_mv_oos, port_returns_naiv)
    z_mv_c = zbro(port_returns_mv_c, port_returns_naiv)
    z_min = zbro(port_returns_min, port_returns_naiv)
    z_min_c = zbro(port_returns_min_c, port_returns_naiv)
    z_g_min_c = zbro(port_returns_g_min_c, port_returns_naiv)
    z_bs = zbro(port_returns_bs, port_returns_naiv)
    z_bs_c = zbro(port_returns_bs_c, port_returns_naiv)

    p_mv = 1 - scipy.stats.norm(0, 1).cdf(z_mv)
    p_mv_c = 1 - scipy.stats.norm(0, 1).cdf(z_mv_c)
    p_min = scipy.stats.norm(0, 1).cdf(z_min)
    p_min_c = scipy.stats.norm(0, 1).cdf(z_min_c)
    p_g_min_c = scipy.stats.norm(0, 1).cdf(z_g_min_c)
    p_bs = 1 - scipy.stats.norm(0, 1).cdf(z_bs)
    p_bs_c = 1 - scipy.stats.norm(0, 1).cdf(z_bs_c)

    print('p-værdi for p_mv = ' + str(p_mv) + ' vs tabelværdi = ' + str(0.17))
    print('p-værdi for p_mv_c = ' + str(p_mv_c) + ' vs tabelværdi = ' + str(0.03))
    print('p-værdi for p_min = ' + str(p_min) + ' vs tabelværdi = ' + str(0.30))
    print('p-værdi for p_min_c = ' + str(p_min_c) + ' vs tabelværdi = ' + str(0.41))
    print('p-værdi for p_g_min_c = ' + str(p_g_min_c) + ' vs tabelværdi = ' + str(0.31))
    print('p-værdi for p_bs = ' + str(p_bs) + ' vs tabelværdi = ' + str(0.19))
    print('p-værdi for p_bs_c = ' + str(p_bs_c) + ' vs tabelværdi = ' + str(0.06))

main()