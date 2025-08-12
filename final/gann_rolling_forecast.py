
# gann_rolling_forecast.py
# 基于 GANN 的滑动窗口预测版本：每次训练预测未来 N 天，测试集固定为最后 150 天

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from deap import base, creator, tools
import random
import warnings
warnings.filterwarnings("ignore")

# ========== 参数配置 ==========
TIME_STEP = 30
POP_SIZE = 20
N_GEN = 10
EARLY_STOP_PATIENCE = 5
LOSS_MODE = 'MAE'

ACTIVATION_FUNCS = ['sigmoid', 'tanh', 'linear']
LEARNING_RATES = [round(0.1 + i*0.05, 2) for i in range(16)]
MOMENTUMS = [round(0.1 + i*0.05, 2) for i in range(13)]
HIDDEN_UNITS = [4, 8, 16]

# ========== 数据读取 ==========
def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    rates = df['Rate'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    rates_scaled = scaler.fit_transform(rates)
    return rates_scaled, scaler, rates.flatten()

# ========== 网络结构构建 ==========
def build_ff_model(input_dim, num_layers, num_units, activation, lr, momentum):
    model = Sequential()
    model.add(Dense(num_units, input_shape=(input_dim,), activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=optimizer,
                  loss='mean_absolute_error' if LOSS_MODE == 'MAE' else 'mean_squared_error')
    return model

# ========== 编码解码辅助函数 ==========
def binary_to_int(bits):
    return int("".join(str(b) for b in bits), 2)

def decode_individual(ind):
    layer_bits = ind[0]
    unit_bits = ind[1:3]
    act_bits = ind[3:5]
    lr_bits = ind[5:9]
    mom_bits = ind[9:13]

    num_layers = 1 if layer_bits == 0 else 2
    num_units = HIDDEN_UNITS[binary_to_int(unit_bits) % len(HIDDEN_UNITS)]
    activation = ACTIVATION_FUNCS[binary_to_int(act_bits) % len(ACTIVATION_FUNCS)]
    lr = LEARNING_RATES[binary_to_int(lr_bits) % len(LEARNING_RATES)]
    momentum = MOMENTUMS[binary_to_int(mom_bits) % len(MOMENTUMS)]

    return num_layers, num_units, activation, lr, momentum

# ========== 遗传算法设定 ==========
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def random_individual():
    return creator.Individual([random.randint(0, 1) for _ in range(13)])

toolbox = base.Toolbox()
toolbox.register("individual", random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    try:
        num_layers, num_units, activation, lr, momentum = decode_individual(ind)
        model = build_ff_model(X_train.shape[1], num_layers, num_units, activation, lr, momentum)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        pred = model.predict(X_val, verbose=0).reshape(-1)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return (np.inf,)
        loss = mean_absolute_error(y_val, pred) if LOSS_MODE == 'MAE' else mean_squared_error(y_val, pred)
        return (loss,)
    except:
        return (np.inf,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# ========== 滑动预测主函数 ==========
def run_gann_rolling_forecast(data_path, predict_horizon=5, test_days=150):
    data_scaled, scaler, data_original = load_data(data_path)
    preds = []
    trues = []

    last_idx = len(data_scaled) - test_days
    for i in range(last_idx - TIME_STEP - predict_horizon + 1, last_idx):
        X_window = []
        y_window = []

        for j in range(i - TIME_STEP + 1, i + 1):
            X_window.append(data_scaled[j:j+TIME_STEP, 0])
            y_window.append(data_scaled[j+TIME_STEP : j+TIME_STEP + predict_horizon, 0])

        X_window = np.array(X_window)
        y_window = np.array(y_window).reshape(-1)

        global X_train, y_train, X_val, y_val
        split = int(len(X_window) * 0.8)
        X_train, y_train = X_window[:split], y_window[:split]
        X_val, y_val = X_window[split:], y_window[split:]

        pop = toolbox.population(n=POP_SIZE)
        best = None
        best_score = np.inf
        no_improve = 0

        for gen in range(N_GEN):
            print(f"Generation {gen+1}/{N_GEN}")
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            current_best = tools.selBest(pop, 1)[0]
            current_score = current_best.fitness.values[0]

            print(f"  → Best fitness this generation: {current_score:.6f}")

        
            if current_score + 1e-6 < best_score:
                best_score = current_score
                best = current_best
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalids = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalids:
                ind.fitness.values = toolbox.evaluate(ind)

            pop[:] = offspring

        num_layers, num_units, activation, lr, momentum = decode_individual(best)
        model = build_ff_model(TIME_STEP, num_layers, num_units, activation, lr, momentum)
        model.fit(X_window, y_window, epochs=20, batch_size=16, verbose=0)
        pred_scaled = model.predict(X_window[-1].reshape(1, -1), verbose=0).reshape(-1)

        pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        true_unscaled = data_original[i + TIME_STEP : i + TIME_STEP + predict_horizon]
        preds.extend(pred_unscaled)
        trues.extend(true_unscaled)

    preds = np.array(preds)
    trues = np.array(trues)

    mae = mean_absolute_error(trues, preds)
    mape = np.mean(np.abs((trues - preds) / trues)) * 100
    mse = mean_squared_error(trues, preds)
    maxae = np.max(np.abs(trues - preds))
    r2 = r2_score(trues, preds)

    print(f" Rolling Forecast (Last {test_days} days, predict {predict_horizon} per step)")
    print(f"AAE       : {mae:.6f}")
    print(f"MAPE (%)  : {mape:.4f}")
    print(f"MSE       : {mse:.6e}")
    print(f"Max AE    : {maxae:.5f}")
    print(f"R-Squared : {r2:.5f}")


# ========== 调用 ==========
if __name__ == "__main__":
    import os
    # 使用相对路径，相对于项目根目录
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final", "ukcndata.csv")
    run_gann_rolling_forecast(data_path, predict_horizon=5, test_days=150)
