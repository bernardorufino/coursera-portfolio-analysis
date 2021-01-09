import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import riskkit as erk
import scipy.stats as stats
import multiprocessing as mp
import itertools
import os
import pickle
from scipy.stats.mstats import gmean

def fformat(f):
    return '{:.10f}'.format(f).rstrip('0').rstrip('.')

FFormatter = ticker.FuncFormatter(lambda x, pos: fformat(x))

def play_games(f, n_games, reward_multiplier=2.5):
    """
    Returns a series with all the values from n_games coin-flip games where in 
    each game a fraction f of the capital is paid and:
    If won (50%) => player gets back reward_multiplier * capital paid 
    If lost (50%) => player gets back 0
    """
    gain_multiplier = (1 - f) + f * reward_multiplier
    loss_multiplier = (1 - f)

    values = []
    sequences = []
    for i in range(2**n_games):
        sequence = []
        wins = 0
        for j in range(n_games):
            win = (i & (1 << j)) != 0
            sequence.append('w' if win else 'l')
            wins += 1 if win else 0
        losses = n_games - wins
        value = gain_multiplier**wins * loss_multiplier**losses
        sequences.append(''.join(reversed(sequence)))
        values.append(value)

    return pd.Series(values, sequences).sort_values()

def play_game(f, stop, reward_multiplier=2.5, series=False,
              max_iterations=1_000):
    """
    Play the game above until stop(iteration, value) => True
    Returns the series
    """
    gain_multiplier = (1 - f) + f * reward_multiplier
    loss_multiplier = (1 - f)

    i = 0
    v = 1
    if series:
        iterations = [i]
        values = [v]
    if f > 0:
        #millis = int(round(time.time() * 1000))
        np.random.seed()
        while i < max_iterations and v > 0 and not stop(i, v):
            won = np.random.uniform() < .5
            v *= gain_multiplier if won else loss_multiplier
            i += 1
            if series:
                iterations.append(i)
                values.append(v)
    
    ans = {'value': v, 'length': i + 1}
    if series:
        ans.update({'series': pd.Series(values, iterations)})
    return ans

def game_quantiles(games):
    """
    games is a dict with column names as key and a game pd.Series as value
    """
    ps = np.linspace(0, 1, 100)
    columns = {column: game.quantile(ps) for column, game in games.items()}
    return pd.DataFrame(columns, ps)

def plot_histogram(series, proportional=False, log=False):
    hist = series.value_counts().sort_index()
    if proportional:
        fig, ax = plt.subplots()
        ax.bar(hist.index, hist, width=len(series)/16)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        hist.index = ['{:.4f}'.format(x) for x in hist.index]
        ax = hist.plot.bar()
        if log:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

def plot_quantile(df):
    ax = df.plot(style='-', logy=True, cmap='cool')
    ax.minorticks_on()
    ax.yaxis.set_major_formatter(FFormatter)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(b=True, which='both', alpha=.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    return ax

def printif(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)

STOP_MODES = {
    'doubled': lambda i, v: v >= 2
}
        
def simulate_single(args):
    f, stop = args
    ans = play_game(f, STOP_MODES[stop], series=False)
    return ans['length'], ans['value']     
        
def simulate(simulations, stop_mode, fs, processes=1, display=True):
    pool = mp.Pool(processes)
    f_lengths = []
    f_values = []
    printif(display, '_' * len(fs))
    for f in fs:
        lengths = []
        values = []
        ans = pool.imap(simulate_single, itertools.repeat((f, stop_mode), simulations))
        lengths, values = map(list, zip(*ans))
        f_lengths.append(pd.Series(lengths))
        f_values.append(pd.Series(values))
        printif(display, '>', end='')
    printif(display)
    return f_lengths, f_values