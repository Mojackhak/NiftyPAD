__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

from . import models
from . import kt
import os
import numpy as np
import matplotlib.pyplot as plt


class TAC:

    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.mft = kt.dt2mft(dt)

        self.km_results = []

    def run_model(self, model_name, model_inputs):
        km_results = dict()
        km_results.update({'model_name': model_name})
        model = getattr(models, model_name)
        kps = model(self.tac, **model_inputs)
        km_results.update(kps)
        self.km_results = km_results

    def run_model_para2tac(self, model_name, model_inputs):
        km_results = dict()
        km_results.update({'model_name': model_name})
        model = getattr(models, model_name)
        kps = model(**model_inputs)
        km_results.update(kps)
        self.km_results = km_results


class Ref:

    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.inputf1_feng_srtm = []
        self.inputf1 = []
        self.inputf1cubic = []
        self.inputf1_exp1 = []
        self.inputf1_exp2 = []
        self.inputf1_exp_am = []
        self.input_interp_method = []
        self.input_interp_1 = []
        self.idx_to_fit = None
        self.fill_in_seconds = None
        self.w = None

    def run_feng_srtm(self):
        self.inputf1_feng_srtm, _ = models.feng_srtm(self.tac, self.dt, w=self.w, fig=True)
        self.input_interp_1 = self.inputf1_feng_srtm
        self.input_interp_method = 'feng_srtm'

    def interp_1(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1 = kt.interpt1(mft, self.tac, self.dt)
        self.input_interp_1 = self.inputf1
        self.input_interp_method = 'linear'

    def interp_1cubic(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1cubic = kt.interpt1cubic(mft, self.tac, self.dt)
        self.input_interp_1 = self.inputf1cubic
        self.input_interp_method = 'cubic'

    def run_exp1(self):
        self.interp_1()
        self.inputf1_exp1 = self.inputf1
        inputf1_exp1 = 0
        if self.idx_to_fit is not None:
            inputf1_exp1, _ = models.exp_1(self.tac, self.dt, idx=self.idx_to_fit, w=self.w, fig=True)
        if self.fill_in_seconds is not None:
            self.inputf1_exp1[self.fill_in_seconds] = inputf1_exp1[self.fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp1
        self.input_interp_method = 'exp_1'

    def run_exp2(self):
        self.interp_1()
        self.inputf1_exp2 = self.inputf1
        inputf1_exp2 = 0
        if self.idx_to_fit is not None:
            inputf1_exp2, _ = models.exp_2(self.tac, self.dt, idx=self.idx_to_fit, w=self.w, fig=True)
        if self.fill_in_seconds is not None:
            self.inputf1_exp2[self.fill_in_seconds] = inputf1_exp2[self.fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp2
        self.input_interp_method = 'exp_2'

    def run_exp_am(self):
        self.interp_1()
        self.inputf1_exp_am = self.inputf1
        inputf1_exp_am = 0
        if self.idx_to_fit is not None:
            inputf1_exp_am, _ = models.exp_am(self.tac, self.dt, idx=self.idx_to_fit, fig=True)
        if self.fill_in_seconds is not None:
            self.inputf1_exp_am[self.fill_in_seconds] = inputf1_exp_am[self.fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp_am
        self.input_interp_method = 'exp_am'

    def auto_find_idx_to_fit(self):
        self.idx_to_fit = list(range(np.argmax(self.tac), self.tac.size))

    def auto_find_fill_in_seconds(self):
        self.fill_in_seconds = []
        dt_gaps = kt.dt_find_gaps(self.dt)
        for i in dt_gaps:
            self.fill_in_seconds = self.fill_in_seconds + list(range(i[0], i[1]))

    def run_interp(self, input_interp_method='linear'):
        input_interp_methods = {'linear': 'interp_1', 'cubic': 'interp_1cubic', 'exp_1': 'run_exp1',
                                'exp_2': 'run_exp2', 'exp_am': 'run_exp_am', 'feng_srtm': 'run_feng_srtm'}
        self.input_interp_method = input_interp_method
        interp_method = getattr(self, input_interp_methods[input_interp_method])
        if 'exp' in input_interp_method:
            if self.idx_to_fit is None:
                self.auto_find_idx_to_fit()
            if self.fill_in_seconds is None:
                self.auto_find_fill_in_seconds()
        interp_method()

    def plot_tac(self, method_interp, save_fig=None):
        # Switch-like behavior using a dictionary
        interp_data = {
            'interp_1': self.inputf1,
            'interp_1cubic': self.inputf1cubic,
            'exp1': self.inputf1_exp1,
            'exp2': self.inputf1_exp2,
            'exp_am': self.inputf1_exp_am,
            'feng_srtm': self.inputf1_feng_srtm
        }.get(method_interp)

        # Raise an error if method_interp is not valid
        if interp_data is None:
            raise ValueError(f"Invalid method_interp '{method_interp}'. Choose from: {list(interp_data.keys())}.")

        # Raise an error if the selected data is empty
        if len(interp_data) == 0:
            raise ValueError(f"The data for method '{method_interp}' is empty. Cannot plot.Calculate with '{method_interp}' first.")

        # Compute the derivative of the TAC (approximated by the finite difference method)
        tac_derivative = np.gradient(interp_data)

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Plot the interpolated TAC and original TAC on the left y-axis
        ax1.plot(interp_data, label='Interpolated TAC (line)', linestyle='-', marker='')
        ax1.scatter(self.dt.mean(axis=0), self.tac, color='red', label='Original TAC (points)')

        # Set labels and legend for the left y-axis
        ax1.set_xlabel("Time (arbitrary units)")
        ax1.set_ylabel("TAC", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.3))  
        
        # Create a twin y-axis for the derivative of TAC
        ax2 = ax1.twinx()
        ax2.plot(tac_derivative, label="Derivative of TAC", linestyle='--', color='green')

        # Set labels and legend for the right y-axis
        ax2.set_ylabel("Derivative of TAC", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.2))  


        # Display the plot
        plt.title("TAC and its Derivative")
        
        # Save the figure if save_fig is specified
        if save_fig:
            if not os.path.isdir(save_fig):
                raise ValueError(f"The path '{save_fig}' is not a valid directory.")
            save_path = os.path.join(save_fig, f'TAC_{method_interp}.svg')
            plt.savefig(save_path, format='svg')
            print(f"Figure saved to: {save_path}")

        # Display the plot
        plt.show()


