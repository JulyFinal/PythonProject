from matplotlib import pyplot
from pmdarima import arima
from pmdarima import datasets
from pmdarima import utils

figure_kwargs = {'figsize': (6, 6)}

head_index = 17*4+2
tail_index = 17*4-4
first_index = head_index - tail_index
last_index = head_index
ausbeer = datasets.load_ausbeer()
timeserie_beer = ausbeer[first_index:last_index]
decomposed = arima.decompose(timeserie_beer, 'additive', m=4)

axes = utils.decomposed_plot(decomposed, figure_kwargs=figure_kwargs,
                             show=False)
axes[0].set_title("Ausbeer Seasonal Decomposition")

decomposed = arima.decompose(datasets.load_airpassengers(),
                             'multiplicative', m=12)

axes = utils.decomposed_plot(decomposed, figure_kwargs=figure_kwargs,
                             show=False)
axes[0].set_title("Airpassengers Seasonal Decomposition")

pyplot.show()