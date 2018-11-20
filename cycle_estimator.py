print("Importing Matplotlib")
import matplotlib.pyplot as plt

print("Importing Numpy")
import numpy as np

print("Importing Scikit Learn")
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm

print("Importing math, random")
from math import exp
import random
random.seed(10)

print("Importing finished")


class Cycle:
    def __init__(
        self, cycle_idx, period_length, initial_rate, decline, noise_distribution
    ):
        self.periods = [p for p in range(period_length)]
        self.input = [[p, cycle_idx] for p in self.periods]
        self.pure_rates = [initial_rate * exp(-decline * p) for p in self.periods]
        self.noisy_rates = [
            r + random.uniform(*noise_distribution) for r in self.pure_rates
        ]
        for i in range(len(self.noisy_rates)):
            if self.noisy_rates[i] < 0:
                self.noisy_rates[i] = 0


class Well:
    def __init__(self):
        self.cycles = []
        self.all_x = []
        self.all_periods = []
        self.all_pure_rates = []
        self.all_noisy_rates = []

    def add_cycle(self, cycle):
        self.all_x += cycle.input
        self.cycles.append(cycle)
        if self.all_periods:
            tmp_periods = [p + max(self.all_periods) for p in cycle.periods]
        else:
            tmp_periods = cycle.periods
        self.all_periods += tmp_periods
        self.all_pure_rates += cycle.pure_rates
        self.all_noisy_rates += cycle.noisy_rates


class DataGenerator:
    def __init__(self, n_wells, n_cycles):
        self.well_data = [Well() for i in range(n_wells)]
        self.n_cycles = n_cycles

    def generate_cycle_data(
        self,
        rate_distribution=[3, 10],
        length_distribution=[10, 20],
        noise_distribution=[-2, 2],
        intra_cycle_decline=0.2,
        extra_cycle_decline=0.3,
    ):
        for w in self.well_data:
            initial_rate = random.uniform(*rate_distribution)
            for cycle in range(self.n_cycles):
                period_length = random.randrange(*length_distribution)
                new_cycle = Cycle(
                    cycle,
                    period_length,
                    initial_rate * exp(-extra_cycle_decline * cycle),
                    intra_cycle_decline,
                    noise_distribution,
                )
                w.add_cycle(new_cycle)

    def get_all_data(self):
        all_inputs = []
        all_pure_rates = []
        all_noisy_rates = []
        for w in self.well_data:
            all_inputs += w.all_x
            all_pure_rates += w.all_pure_rates
            all_noisy_rates += w.all_noisy_rates
        return all_inputs, all_pure_rates, all_noisy_rates

    def get_prediction_x(self):
        all_x = []
        for i in range(self.n_cycles):
            cycle_x = []
            for w in self.well_data:
                if len(w.cycles[i].input) > len(cycle_x):
                    cycle_x = w.cycles[i].input
            all_x += cycle_x
        return all_x

    def calc_average_data(self, cycle, pure=True):
        periods_length = max([len(w.cycles[cycle].periods) for w in self.well_data])
        avg_periods = [i for i in range(periods_length)]
        avg_rates = []
        for p in avg_periods:
            sum_rate = 0.0
            count = 0
            for w in self.well_data:
                if len(w.cycles[cycle].periods) > p:
                    count += 1
                    if pure:
                        sum_rate += w.cycles[cycle].pure_rates[p]
                    else:
                        sum_rate += w.cycles[cycle].noisy_rates[p]
            avg_rates.append(sum_rate/count)
        return avg_periods, avg_rates

    def plot_well_data(self, lines, columns):
        fig, axes = plt.subplots(lines, columns)
        for i in range(lines):
            for j in range(columns):
                axes[i][j].set_axisbelow(True)
                axes[i][j].grid(linestyle='dashed')
                idx = i * columns + j
                axes[i][j].scatter(
                    self.well_data[idx].all_periods, 
                    self.well_data[idx].all_pure_rates, 
                    label='Vazão'
                )
                axes[i][j].scatter(
                    self.well_data[idx].all_periods, 
                    self.well_data[idx].all_noisy_rates, 
                    label='Vazão com ruído'
                )
                axes[i][j].set_ylim([0,10])
                axes[i][j].set_xlabel('Tempo (meses)')
                axes[i][j].set_ylabel('Vazão (m3/d)')
                axes[i][j].legend()
        plt.show()

    def plot_cycle_data(self):
        fig, axes = plt.subplots(2, self.n_cycles)
        for i in range(self.n_cycles):
            for w in self.well_data:
                axes[0][i].scatter(w.cycles[i].periods, w.cycles[i].pure_rates)
                axes[1][i].scatter(w.cycles[i].periods, w.cycles[i].noisy_rates)
        plt.show()

    def plot_averages(self, nn=None):
        fig = plt.figure(figsize=(12,8))
        if not nn:
            columns = 2
        else:
            columns = 3
        axes = fig.subplots(2, columns)
        fig.tight_layout(pad=3)

        last_length = 0
        for w in self.well_data:
            axes[0][0].scatter(w.all_periods, w.all_pure_rates)
            axes[1][0].scatter(w.all_periods, w.all_pure_rates, alpha=0.2)
            axes[0][1].scatter(w.all_periods, w.all_noisy_rates)
            axes[1][1].scatter(w.all_periods, w.all_noisy_rates, alpha=0.2)
            if nn:
                axes[0][2].scatter(w.all_periods, w.all_noisy_rates)
                axes[1][2].scatter(w.all_periods, w.all_noisy_rates, alpha=0.2)

        for i in range(self.n_cycles):
            average_periods, average_rates_pure = self.calc_average_data(i, True)
            average_periods, average_rates_noisy = self.calc_average_data(i, False)
            for i,p in enumerate(average_periods):
                average_periods[i] += last_length
            last_length += len(average_periods)
            axes[1][0].scatter(average_periods, average_rates_pure)
            axes[1][1].scatter(average_periods, average_rates_noisy)

        predicted_rates = nn.predict(self.get_prediction_x())
        prediction_periods = [i for i in range(len(self.get_prediction_x()))]
        axes[1][2].scatter(prediction_periods, predicted_rates)

        titles = [['Vazões sem ruído', 'Vazões com ruído', 'Vazões com ruído'],
                  ['Valores médios', 'Valores médios', 'Previsão RNA']]
        for i in range(2):
            for j in range(columns):
                axes[i][j].set_ylim([0,10])
                axes[i][j].set_xlabel('Tempo (meses)')
                axes[i][j].set_ylabel('Vazão (m3/d)')
                axes[i][j].set_title(titles[i][j])

        plt.show()

    def plot_prediction(self, nn):
        fig, axes = plt.subplots(3, 4)
        
        for i in range(self.n_cycles):
            longest_periods = []
            longest_input = []
            for w in self.well_data:
                axes[0][i].scatter(w.cycles[i].periods, w.cycles[i].pure_rates)
                axes[1][i].scatter(w.cycles[i].periods, w.cycles[i].pure_rates, alpha=0.2)
                axes[2][i].scatter(w.cycles[i].periods, w.cycles[i].pure_rates, alpha=0.2)
                if len(w.cycles[i].periods) > len(longest_periods):
                    longest_periods = w.cycles[i].periods
                    longest_input = w.cycles[i].input
            average_periods, average_rates = self.calc_average_data(i, True)
            axes[1][i].scatter(average_periods, average_rates)
            predicted_rates = nn.predict(longest_input)
            axes[2][i].scatter(longest_periods, predicted_rates)
        
        for w in self.well_data:
            axes[0][3].scatter(w.all_periods, w.all_pure_rates)
            axes[1][3].scatter(w.all_periods, w.all_pure_rates, alpha=0.2)
            axes[2][3].scatter(w.all_periods, w.all_pure_rates, alpha=0.2)
        
        predicted_rates = nn.predict(self.get_prediction_x())
        prediction_periods = [i for i in range(len(self.get_prediction_x()))]
        axes[2][3].scatter(prediction_periods, predicted_rates)
        plt.show()

    def plot_prediction_noisy(self, nn):
        fig = plt.figure(figsize=(12,8))
        axes = fig.subplots(3, 4)

        axes[0][0].set_title('Ciclo 1')
        axes[0][1].set_title('Ciclo 2')
        axes[0][2].set_title('Ciclo 3')
        axes[0][3].set_title('Todos os ciclos')
        fig.tight_layout(pad=3)
        for i in range(3):
            for j in range(4):
                if i == 1:
                    axes[i][j].set_title('Valores médios') 
                elif i == 2:
                    axes[i][j].set_title('Previsão RNA') 
                axes[i][j].set_ylim([0,10])
                axes[i][j].set_xlabel('Tempo (meses)')
                axes[i][j].set_ylabel('Vazão (m3/d)')
        
        for w in self.well_data:
            axes[0][3].scatter(w.all_periods, w.all_noisy_rates)
            axes[1][3].scatter(w.all_periods, w.all_noisy_rates, alpha=0.2)
            axes[2][3].scatter(w.all_periods, w.all_noisy_rates, alpha=0.2)

        all_averages = []
        for i in range(self.n_cycles):
            longest_periods = []
            longest_input = []
            for w in self.well_data:
                axes[0][i].scatter(w.cycles[i].periods, w.cycles[i].noisy_rates)
                axes[1][i].scatter(w.cycles[i].periods, w.cycles[i].noisy_rates, alpha=0.2)
                axes[2][i].scatter(w.cycles[i].periods, w.cycles[i].noisy_rates, alpha=0.2)
                if len(w.cycles[i].periods) > len(longest_periods):
                    longest_periods = w.cycles[i].periods
                    longest_input = w.cycles[i].input
            average_periods, average_rates = self.calc_average_data(i, False)
            axes[1][i].scatter(average_periods, average_rates)
            all_averages += average_rates

            predicted_rates = nn.predict(longest_input)
            axes[2][i].scatter(longest_periods, predicted_rates)
        
        predicted_rates = nn.predict(self.get_prediction_x())
        prediction_periods = [i for i in range(len(self.get_prediction_x()))]
        axes[1][3].scatter(prediction_periods, all_averages)
        axes[2][3].scatter(prediction_periods, predicted_rates)
        
        plt.show()

    def compare_to_pure_rate(self, nn):
        all_averages = []
        for i in range(self.n_cycles):
            average_periods, average_rates = self.calc_average_data(i, True)
            all_averages += average_rates

        predicted_rates = nn.predict(self.get_prediction_x())
        mse = mean_squared_error(all_averages, predicted_rates)
        r2s = r2_score(all_averages, predicted_rates)
        print(f"mse:{mse} r2_score:{100*r2s:.2f}")


class NeuralNetwork:
    def __init__(self, X, y, hidden_layer_sizes=(10,), debug=False):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=27
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=1e-3,
            activation="logistic",
            solver="lbfgs",
            verbose=debug,
            random_state=3,  # 1 da r2 100% :O
        )

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        return [mean_squared_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)]

    def predict(self, X):
        return self.clf.predict(X)


def main():
    print("Começou")
    debug = True
    # input_files = ["input_t.txt", "input_t_t-1.txt", "input_t_t-1_t-2.txt"]
    # input_mngrs = []
    # for inp in input_files:
    #     input_mngrs.append(InputManager(inp))
    # architectures = [(100,), (100,), (100,)]
    # labels = ["2x50x1", "3x50x1", "4x50x1"]
    # neural_networks = []

    # for i, mngr in enumerate(input_mngrs):
    #     nn = NeuralNetwork(mngr.X, mngr.y, architectures[i])
    #     nn.fit()
    #     mse, r2s = nn.accuracy()
    #     print(f"{labels[i]}: mse:{mse} r2_score:{100*r2s:.2f}")
    #     neural_networks.append(nn)
    #     mngr.plot_data(nn, i)
    #     # break

    dg = DataGenerator(n_wells=10, n_cycles=3)
    dg.generate_cycle_data(
        rate_distribution=[3, 10],
        length_distribution=[10, 20],
        noise_distribution=[-3, 3],
        intra_cycle_decline=0.2,
        extra_cycle_decline=0.3
    )
    # dg.plot_well_data(2,2)
    # dg.plot_averages()
    # dg.plot_cycle_data()
    all_inputs, all_rates_pure, all_rates_noisy = dg.get_all_data()
    # nn = NeuralNetwork(all_inputs, all_rates_pure)
    # nn.fit()
    # mse, r2s = nn.accuracy()
    # dg.plot_prediction(nn)

    # architectures = [(10,), (10,10,), (10,10,10,), (30,30,30,), (30,60,60,30,), (50,), (100,)]
    architectures = [(30,60,60,30,)]

    for arch in architectures:
        nn_2 = NeuralNetwork(all_inputs, all_rates_noisy, arch)
        nn_2.fit()
        mse, r2s = nn_2.accuracy()
        # dg.plot_prediction_noisy(nn_2)
        dg.compare_to_pure_rate(nn_2)
        dg.plot_averages(nn_2)
    # print(f"mse:{mse} r2_score:{100*r2s:.2f}")


if __name__ == "__main__":
    main()


# References
# https://stackoverflow.com/questions/42713276/valueerror-unknown-label-type-while-implementing-mlpclassifier
# http://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/
# https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
# Prediction error http://www.scikit-yb.org/en/latest/api/regressor/peplot.html
# https://stackoverflow.com/questions/41069905/trouble-fitting-simple-data-with-mlpregressor
