
# gann_forecast_normalized_out150.py
# 修改点：
# 1. 增加输出归一化指标
# 2. 固定最后150个观测点为out-of-sample测试集，不进入训练/验证

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools
import random
import warnings
warnings.filterwarnings("ignore")

# 参数配置
TIME_STEP = 30
POP_SIZE = 100
N_GEN = 25
TEST_SIZE = 150
TRAIN_RATIO, VAL_RATIO = 0.7, 0.30
EARLY_STOP_PATIENCE = 10
LOSS_MODE = 'MAE'

ACTIVATION_FUNCS = ['sigmoid', 'tanh', 'linear']
LEARNING_RATES = [round(0.1 + i*0.05, 2) for i in range(16)]
MOMENTUMS = [round(0.1 + i*0.05, 2) for i in range(13)]
HIDDEN_UNITS = [4, 8, 16]

# 数据处理函数
def load_and_split_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    rates = df['Rate'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    rates_scaled = scaler.fit_transform(rates)

    def create_dataset(series):
        X, y = [], []
        for i in range(len(series) - TIME_STEP):
            X.append(series[i:i+TIME_STEP, 0])
            y.append(series[i+TIME_STEP, 0])
        return np.array(X), np.array(y)

    X_all, y_all = create_dataset(rates_scaled)

    X_train_all = X_all[:-TEST_SIZE]
    y_train_all = y_all[:-TEST_SIZE]
    X_test = X_all[-TEST_SIZE:]
    y_test = y_all[-TEST_SIZE:]

    total = len(X_train_all)
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

    return X_train_all[:train_end], y_train_all[:train_end],            X_train_all[train_end:val_end], y_train_all[train_end:val_end],            X_test, y_test, scaler

# 模型构建函数
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

# 二进制解码
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

# 遗传算法设置
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
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
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

# 主函数：运行 GANN
def run_gann_structure(loss_mode='MAE'):
    global X_train, y_train, X_val, y_val, X_test, y_test, scaler, LOSS_MODE
    LOSS_MODE = loss_mode

    pop = toolbox.population(n=POP_SIZE)
    best_score = np.inf
    best = None
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
    print(f"\nBest network architecture found:")
    print(f"  Hidden layers   : {num_layers}")
    print(f"  Hidden units    : {num_units}")
    print(f"  Activation func : {activation}")
    print(f"  Learning rate   : {lr}")
    print(f"  Momentum        : {momentum}")

    model = build_ff_model(X_train.shape[1], num_layers, num_units, activation, lr, momentum)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_scaled = y_test.reshape(-1)

    # 反归一化
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

    # 原始空间指标
    print(f"\n Results (Original scale):")
    print(f"AAE       : {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"MAPE (%)  : {np.mean(np.abs((y_true - y_pred)/y_true))*100:.4f}")
    print(f"MSE       : {mean_squared_error(y_true, y_pred):.6e}")
    print(f"Max AE    : {np.max(np.abs(y_true - y_pred)):.5f}")
    print(f"R-Squared : {r2_score(y_true, y_pred):.5f}")

    # 归一化空间指标
    print(f"\n Results (Normalized scale):")
    print(f"AAE (norm)  : {mean_absolute_error(y_scaled, pred_scaled):.6f}")
    print(f"MAPE (norm) : {np.mean(np.abs((y_scaled - pred_scaled)/y_scaled))*100:.4f}")
    print(f"MSE (norm)  : {mean_squared_error(y_scaled, pred_scaled):.6e}")
    print(f"Max AE (n)  : {np.max(np.abs(y_scaled - pred_scaled)):.5f}")
    print(f"R-Squared   : {r2_score(y_scaled, pred_scaled):.5f}")

# 主入口
if __name__ == '__main__':
    import os
    # 使用相对路径，相对于项目根目录
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "sorted_output_file.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_split_data(path)
    run_gann_structure(loss_mode='MAE')
