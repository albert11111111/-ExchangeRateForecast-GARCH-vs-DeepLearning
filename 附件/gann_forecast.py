# gann_forecast.py
# 基于 Nag & Mitra (2002) 设计的 GANN 二进制编码版本

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# 确保 tensorflow 导入在脚本的早期，以便进行 GPU 检测
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from deap import base, creator, tools
import random
import warnings
warnings.filterwarnings("ignore")

# ========== 参数配置 ==========
TIME_STEP = 30
POP_SIZE = 30
N_GEN = 25
TRAIN_RATIO, VAL_RATIO = 0.7, 0.15
EARLY_STOP_PATIENCE = 10
LOSS_MODE = 'MAE'
FEEDBACK = False

# 枚举值定义
ACTIVATION_FUNCS = ['sigmoid', 'tanh', 'linear']  # 2位 => 3种
LEARNING_RATES = [round(0.1 + i*0.05, 2) for i in range(16)]  # 0.10~0.85 共16种 => 4位
MOMENTUMS = [round(0.1 + i*0.05, 2) for i in range(13)]       # 0.10~0.70 共13种 => 4位
HIDDEN_UNITS = [4, 8, 16]  # 2位

# ========== 数据读取与划分 ==========
def load_and_split_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    rates = df['rate'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    rates_scaled = scaler.fit_transform(rates)

    def create_dataset(series):
        X, y = [], []
        for i in range(len(series) - TIME_STEP):
            X.append(series[i:i+TIME_STEP, 0])
            y.append(series[i+TIME_STEP, 0])
        return np.array(X), np.array(y)

    X_all, y_all = create_dataset(rates_scaled)
    total = len(X_all)
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)
    return (X_all[:train_end], y_all[:train_end],
            X_all[train_end:val_end], y_all[train_end:val_end],
            X_all[val_end:], y_all[val_end:], scaler, rates)

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
    layer_bits = ind[0]                   # 1 bit
    unit_bits = ind[1:3]                 # 2 bits
    act_bits = ind[3:5]                  # 2 bits
    lr_bits = ind[5:9]                   # 4 bits
    mom_bits = ind[9:13]                 # 4 bits

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
        model.fit(X_train, y_train, epochs=20, batch_size=32,
                  validation_data=(X_val, y_val), verbose=0)
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

# ========== 主流程 ==========
def run_gann_structure(feedback=False, loss_mode='MAE'):
    global X_train, y_train, X_val, y_val, X_test, y_test, LOSS_MODE, FEEDBACK
    LOSS_MODE = loss_mode
    FEEDBACK = feedback

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
    model = build_ff_model(X_train.shape[1], num_layers, num_units, activation, lr, momentum)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    pred = model.predict(X_test, verbose=0).reshape(-1)

    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    maxae = np.max(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    label = f"{'GFB' if feedback else 'GFF'}-{loss_mode}"
    print(f"\nResults for {label}")
    print(f"AAE       : {mae:.6f}")
    print(f"MAPE (%)  : {mape:.4f}")
    print(f"MSE       : {mse:.6e}")
    print(f"Max AE    : {maxae:.5f}")
    print(f"R-Squared : {r2:.5f}")


#==================对照组
from tensorflow.keras.callbacks import EarlyStopping

def run_baseline_nn(hidden_layers=1, hidden_units=8, activation='sigmoid', lr=0.1, momentum=0.1):
    print("\n[Baseline NN] Feedforward network (No Genetic Algorithm)")

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(X_train.shape[1],), activation=activation))
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
              validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    pred = model.predict(X_test, verbose=0).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    maxae = np.max(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"AAE       : {mae:.6f}")
    print(f"MAPE (%)  : {mape:.4f}")
    print(f"MSE       : {mse:.6e}")
    print(f"Max AE    : {maxae:.5f}")
    print(f"R-Squared : {r2:.5f}")

# ========== 主入口 ==========
if __name__ == '__main__':
    # 检查 TensorFlow GPU 可用性
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 允许内存增长，避免一开始就占用所有 GPU 内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个 GPU:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        except RuntimeError as e:
            # 发生错误时打印信息
            print(f"设置 GPU 时发生运行时错误: {e}")
    else:
        print("未找到 GPU，将在 CPU 上运行。")

    path = r"F:\\系统建模与仿真\\new-Timeserires\\Time-Series-Library-main\\Time-Series-Library-main\\英镑兑人民币_20250324_102930.csv"
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, _ = load_and_split_data(path)


    run_baseline_nn(hidden_layers=1, hidden_units=8, activation='sigmoid', lr=0.1, momentum=0.1)


    run_gann_structure(feedback=False, loss_mode='MAE')   # GFF-AEL
    # run_gann_structure(feedback=True,  loss_mode='MAE')   # GFB-AEL
    # run_gann_structure(feedback=False, loss_mode='MSE')   # GFF-SEL
    # run_gann_structure(feedback=True,  loss_mode='MSE')   # GFB-SEL
