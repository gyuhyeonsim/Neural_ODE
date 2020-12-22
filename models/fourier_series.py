import torch
import torch.nn as nn
from symfit import parameters, variables, sin, cos, Fit
import numpy as np

# https://stackoverflow.com/questions/52524919/fourier-series-fit-in-python
# reference: https://pypi.org/project/symfit/

class FourierSeries(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n= args.model['n']
        self.dilation = args.model['dilation']

    def forward(self, inter_t, extra_t, ydata):
        x, y = variables('x, y')

        if self.dilation == True:
            w, = parameters('w')
        else:
            w = 1

        model_dict = {y: self.fourier_series(x, f=w, n=self.n)}

        xdata = inter_t.numpy()

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=xdata, y=ydata)
        fit_result = fit.execute()

        pred = fit.model(x=xdata, **fit_result.params).y
        pred = torch.from_numpy(pred)

        extra_pred = fit.model(x=extra_t, **fit_result.params).y
        extra_pred = torch.from_numpy(extra_pred)

        return pred, extra_pred

    def fourier_series(self, x, f, n=0):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """
        # Make the parameter objects for all the terms
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))

        # Construct the series
        series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                         for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))

        return series