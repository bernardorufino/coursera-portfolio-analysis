import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from scipy.optimize import minimize

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns and returns a pd.DataFrame that 
    contains:
      * 'Wealth': Wealth index
      * 'Peaks': Previous peaks
      * 'Drawdown': Percent drawdowns
    """
    wealth = 1000 * (1 + return_series).cumprod()
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks) / peaks
    return pd.DataFrame({
        'Wealth': wealth,
        'Peaks': peaks,
        'Drawdown': drawdowns
    })

def get_ffme_allreturns(columns=None):
    """
    Load the Fama-French market equity dataset for the monthly returns.
    """
    rets = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, parse_dates=True, 
                       na_values=[-99.99])
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    if columns:
        rets = rets[columns.keys()].rename(columns=columns)
    return rets

def get_ffme_returns():
    """
    Load the Fama-French market equity dataset for the monthly returns of the 
    top and bottom deciles by market cap.
    """
    return get_ffme_allreturns({
        'Lo 10': 'SmallCap',
        'Hi 10': 'LargeCap'
    })

def get_ffme_returns():
    """
    Load the Fama-French market equity dataset for the monthly returns of the 
    top and bottom deciles by market cap.
    """
    rets = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, parse_dates=True, 
                       na_values=[-99.99])
    columns = {
        'Lo 10': 'SmallCap',
        'Hi 10': 'LargeCap'
    }
    rets = rets[columns.keys()].rename(columns=columns)
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    """
    Load the EDHEC Hedge Fund Index returns.
    """
    rets = pd.read_csv('data/edhec-hedgefundindices.csv',
                       header=0, index_col=0, parse_dates=True)
    rets = rets / 100
    rets.index = rets.index.to_period('M')
    return rets

def get_ind_returns():
    """
    Load the Ken-French 30-Industry portfolios value weighted monthly returns
    """
    rets = pd.read_csv('data/ind30_m_vw_rets.csv',
                       header=0, index_col=0, parse_dates=True)
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    rets.columns = rets.columns.str.strip()
    return rets

def std_moment(data, k):
    """
    Computes the k-th moment of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    return ((data - data.mean())**k).mean() / data.std(ddof=0)**k

def skewness(data):
    """
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    return std_moment(data, 3)

def kurtosis(data):
    """
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    return std_moment(data, 4)

def is_normal(data, level=.01):
    """
    Applies a Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera (data)
    return p_value > level

def downside_deviation(rets):
    """
    "Correct" computaion of semi-deviation. But also pretty bad? This doesn't 
    consider the frequency with which these downside returns happen, if they 
    happen pretty rarily, you'll get a big number here. 
    """    
    mean = rets.mean()
    return ((rets[rets < mean] - mean)**2).mean()**.5

def semi_deviation(rets):
    """
    This is probably wrong, check downside_deviation() for the correct one.
    """
    return rets[rets < 0].std(ddof=0)

def var_historic(rets, level=5):
    """
    Computes historic VaR (value-at-risk) for level (0 <= x <= 100).
    Returns fall below this value level% of the time and above this value 
    (100-level)% of the time.
    """
    if isinstance(rets, pd.DataFrame):
        #return rets.agg(lambda column: var_historic(column, level))
        return rets.agg(var_historic, level=level)
    else:
        return -np.percentile(rets, level)

def var_gaussian(rets, level=5, modified=False):
    """
    Computes VaR (value-at-risk) for level (0 <= x <= 100).
    
    If modified is False: 
      Assumes rets is gaussian. 
      Returns -(mean + z_level * sd)
      
    If modified is True: 
      Use Cornish-Fisher approximation based on rets.
      Returns -(mean + z_level_approx * sd)
    
    """
    z_level = stats.norm.ppf(level/100)
    if modified:
        s = skewness(rets)
        k = kurtosis(rets)
        z_level = cornish_fisher(z_level, s, k)
        
    return -(rets.mean() +  z_level*rets.std(ddof=0))

def cornish_fisher(z_level, skewness, kurtosis):
    """
    Returns the Cornish-Fisher approximation of z_level (the level percentile) 
    of a distribution with skewness and kurtosis as provided.
    """
    return (z_level 
           + 1/6*(z_level**2-1)*skewness 
           + 1/24*(z_level**3-3*z_level)*(kurtosis-3) 
           - 1/36*(2*z_level**3-5*z_level)*skewness**2
   )

def cvar_historic(rets, level=5):
    if isinstance(rets, pd.DataFrame):
        return rets.agg(cvar_historic, level=level)
    else:
        var = var_historic(rets, level)
        return -rets[rets <= -var].mean()
    
def annualize_rets(rets, periods_per_year):
    """
    Annualize a set of returns
    TODO: Infer periods per year
    """
    return (1 + rets).agg(gmean) ** periods_per_year - 1
    # Or
    # return np.exp(np.log(1 + rets).mean()) ** periods_per_year - 1
    # return (1 + rets).prod() ** (periods_per_year / rets.shape[0]) - 1

def annualize_vol(rets, periods_per_year):
    """
    Annualize the volatility of a set of returns
    TODO: Infer periods per year
    """
    return rets.std() * periods_per_year**.5

def sharpe_ratio(rets, rf_rate_per_year, periods_per_year):
    """
    Computes the sharpe ratio given the risk-free rate (per year)
    TODO: Infer periods per year
    TODO: Why not annualize first than take the rf_rate?
    """
    rf_rate = (1 + rf_rate_per_year) ** (1/periods_per_year) - 1
    excess_ret = annualize_rets(rets - rf_rate, periods_per_year)
    vol = annualize_vol(rets, periods_per_year)
    return excess_ret / vol
    
def sharpe_ratio_wrong(rets, rf_rate_per_year, periods_per_year):
    """
    WRONG!!!
    Computes the sharpe ratio given the risk-free rate (per year)
    TODO: This is wrong, why?
    TODO: Why not annualize first than take the rf_rate?
    """
    ret = annualize_rets(rets, periods_per_year)
    vol = annualize_vol(rets, periods_per_year)
    return ret - rf_rate_per_year / vol
    
def portfolio_return(w, er):
    """
    Returns the weighted averate of returns for portfolio with weights w and 
    returns rets
    """
    # Didn't need the .T, according to np.matmul?, if its 1-D it prepends 1 to 
    # its shape when on the left, and when on the right it appends 1 to its
    # shape.
    return w.T @ er

def portfolio_vol(w, cov):
    """
    Returns the volatility of the portfolio with weights w and covariance matrix 
    cov
    """
    # Didn't need the .T, according to np.matmul?, if its 1-D it prepends 1 to 
    # its shape when on the left, and when on the right it appends 1 to its
    # shape.
    return (w.T @ cov @ w)**.5

def portfolio_retvols(weights, er, cov):
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    return rets, vols

def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier (not only the ef, but the curve)
    """
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError('Can only plot 2-asset frontiers')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    return ef.plot.line(x='Volatility', y='Returns', style=style)
    
def minimize_vol(target_return, er, cov):
    """
    Returns the weight vector responsible for the portfolio that generates 
    target_return with minimum volatility
    """
    n = er.shape[0]
    initial_weights = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'fun': lambda w: target_return - portfolio_return(w, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda w: w.sum() - 1        
    }
    results = minimize(portfolio_vol, initial_weights, args=(cov,), 
                       method='SLSQP', options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return results.x

def optimal_weights(n_points, er, cov):
    target_returns = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(r, er, cov) for r in target_returns]

def np_unit(n, i):
    array = np.zeros(n)
    array[i] = 1
    return array

def plot_ef_my(n_points, er, cov, style='-'):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets, vols = portfolio_retvols(weights, er, cov) 
    plt.plot(vols, rets, style)
    n = er.size
    weights = [np_unit(n, i) for i in range(n)]
    rets, vols = portfolio_retvols(weights, er, cov) 
    plt.scatter(vols, rets, c='red')
    
def plot_ef(n_points, er, cov, rf_rate=0.01, show_cml=True, style='-', 
            show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    # Draw curve
    weights = optimal_weights(n_points, er, cov)
    rets, vols = portfolio_retvols(weights, er, cov) 
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    ax = ef.plot.line(x='Volatility', y='Returns', style='-')
    
    # Draw assets in portfolio as markers
    n = er.size
    weights = [np_unit(n, i) for i in range(n)]
    rets, vols = portfolio_retvols(weights, er, cov) 
    ax.scatter(vols, rets, c='orange')
    
    # Show equally-weighted (EW) portfolio
    if show_ew:
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color='green', marker='o', 
                   markersize=9)
        ax.annotate('EW', (vol_ew, r_ew))
    
    # Show global minimum volatility (GMV) portfolio
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', 
                   markersize=9)
        ax.annotate('GMV', (vol_gmv, r_gmv))
    
    # Draw capital market line (CML)
    if show_cml:
        w_msr =msr(rf_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [rf_rate, r_msr]
        ax.set_xlim(left=0)
        ax.plot(cml_x, cml_y, color='red', marker='o', linestyle='dashed', 
                markersize=8, linewidth=2)
        ax.annotate('MSR', (vol_msr, r_msr))
    
    return ax
    
def plot_ef_series(n_points, rets, periods_per_year, rf_rate=0.01, 
                   show_cml=True, annotate=False):
    er = annualize_rets(rets, periods_per_year)
    cov = rets.cov()
    labels = rets.columns
    plot_ef(n_points, er, cov, rf_rate=rf_rate, show_cml=show_cml)
    
    n = er.size
    weights = [np_unit(n, i) for i in range(n)]
    rets, vols = portfolio_retvols(weights, er, cov) 
    plt.scatter(vols, rets, color='orange')
    if annotate:
        for i in range(n):
            plt.annotate(labels[i], (vols[i], rets[i]))

def msr(rf_rate, er, cov):
    """
    Returns the weight vector responsible for the portfolio that maximizes the 
    sharpe ratio (MSR) given the risk-free rate rf_rate
    """
    n = er.shape[0]
    initial_weights = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda w: w.sum() - 1        
    }
    def neg_sharpe_ratio(w):
        ret = portfolio_return(w, er)
        vol = portfolio_vol(w, cov)
        return -(ret - rf_rate) / vol
    results = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', 
                       options={'disp': False}, constraints=(weights_sum_to_1), 
                       bounds=bounds)
    return results.x
  
def gmv(cov):
    """
    Returns the weight vector responsible for the global minimum volatility 
    (GMV) portfolio given the covariance matrix.
    """
    n = cov.shape[0]
    # If all returns are the same the MSR is the one with lowest volatility, ie
    # the GMV.
    return msr(0, np.repeat(1, n), cov)
        
    
    
    