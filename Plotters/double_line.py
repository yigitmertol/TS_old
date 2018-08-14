import numpy as np
import matplotlib.pyplot as plt
from Processors import  preprocessing_SNP


#Setup dummy data
N = 10
ind = np.arange(N)
bars = np.random.randn(N)
t = np.arange(0.01, 10.0, 0.01)

def make_2( ind, bars, t):
    # Plot graph with 2 y axes
    fig, ax1 = plt.subplots()

    # Plot bars
    ax1.bar(ind, bars, alpha=0.3)
    ax1.set_xlabel('$t$')

    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('bar', color='b')
    [tl.set_color('b') for tl in ax1.get_yticklabels()]

    # Set up ax2 to be the second y axis with x shared
    ax2 = ax1.twinx()
    # Plot a line
    ax2.plot(t, 2 * np.sin(0.25 * np.pi * t), 'r-')

    # Make the y-axis label and tick labels match the line color.
    ax2.set_ylabel('sin', color='r')
    [tl.set_color('r') for tl in ax2.get_yticklabels()]

    plt.show()

# make_2( ind, bars, t)