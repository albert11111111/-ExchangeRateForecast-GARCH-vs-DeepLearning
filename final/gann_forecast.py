# gann_forecast.py
# 基于 Nag & Mitra (2002) 设计的 GANN 二进制编码版本

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from deap import base, creator, tools
import random
import warnings
warnings.filterwarnings("ignore")

# ========== 参数配置 ==========
TIME_STEP = 30
POP_SIZE = 100
N_GEN = 25
TRAIN_RATIO, VAL_RATIO = 0.7, 0.3
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
    df = pd.read_csv(path, parse_dates=['Date'])
    
    rates = df['Rate'].values

    # 计算 log-return：r[t] = log(R[t] / R[t-1])
    log_returns = np.log(rates[1:] / rates[:-1])  # 长度为 n-1
    log_rates = log_returns.reshape(-1, 1)

    # 标准化 log-return
    scaler = StandardScaler()
    log_scaled = scaler.fit_transform(log_rates)


    # 构建数据集：使用 TIME_STEP 天的收益率预测第 TIME_STEP+1 天
    def create_dataset(series):
        X, y = [], []
        for i in range(len(series) - TIME_STEP):
            X.append(series[i:i+TIME_STEP, 0])
            y.append(series[i+TIME_STEP, 0])
        return np.array(X), np.array(y)

    X_all, y_all = create_dataset(log_scaled)


    # ✅ 保留最后150个点为测试集
    TEST_SIZE = 150
    X_test = X_all[-TEST_SIZE:]
    y_test = y_all[-TEST_SIZE:]

    X_trainval = X_all[:-TEST_SIZE]
    y_trainval = y_all[:-TEST_SIZE]

    total = len(X_trainval)
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

    X_train = X_trainval[:train_end]
    y_train = y_trainval[:train_end]
    X_val = X_trainval[train_end:val_end]
    y_val = y_trainval[train_end:val_end]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, rates

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
def run_gann_structure(feedback=False, loss_mode='MAE',rates=None,log_returns=True):
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

    model = Sequential()
    model.add(Dense(num_units, input_shape=(X_train.shape[1],), activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation=activation))
    model.add(Dense(1, activation='tanh'))  # 输出范围 (-1, 1)
    # 输出视为对数收益率，乘以最大可能波动率（例如 0.05）

    optimizer = SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # ===== 模型训练过程 =====
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # ===== 模型预测与异常修正 =====
    pred = model.predict(X_test, verbose=0).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    # ===== 汇率还原与评估指标 =====
    if log_returns:
        y_true_log = y_test
        y_pred_log = pred
        y_pred_log = np.clip(y_pred_log, -0.001, 0.001)  # 控制在 ±5%

        r_base = rates[-len(y_true_log)-1:-1].reshape(-1)
        y_pred = np.exp(y_pred_log) * r_base
        y_true = rates[-len(y_pred):].reshape(-1)  # ✅ 真实汇率直接取值
        
    else:
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
    plot_prediction_vs_true(y_true, y_pred)
    # 指标计算
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
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

    # 📉 绘图：预测 vs 真实
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True Rate", linewidth=2)
    plt.plot(y_pred, label="Predicted Rate", linewidth=2)
    plt.title(f"Predicted vs True Exchange Rates ({label})")
    plt.xlabel("Days")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ✅ 保存图像，使用 label 命名文件
    plt.savefig(f"prediction_curve_{label}.png")
    plt.show()




#==================对照组
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_baseline_nn(
    hidden_layers=2,
    hidden_units=8,
    activation='sigmoid',
    lr=0.1,
    momentum=0.4,
    rates=None,
    log_returns=True
):
    print("\n[Baseline NN] Feedforward network (No Genetic Algorithm)")
    
    # ===== 构建模型结构 =====
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(X_train.shape[1],), activation=activation))
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1, activation='tanh'))  # 输出范围 (-1, 1)
    # 输出视为对数收益率，乘以最大可能波动率（例如 0.05）

    optimizer = SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # ===== 模型训练过程 =====
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # ===== 模型预测与异常修正 =====
    pred = model.predict(X_test, verbose=0).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    # ===== 汇率还原与评估指标 =====
    if log_returns:
        y_true_log = y_test
        y_pred_log = pred
        y_pred_log = np.clip(y_pred_log, -0.001, 0.001)  # 控制在 ±5%

        r_base = rates[-len(y_true_log)-1:-1].reshape(-1)
        y_pred = np.exp(y_pred_log) * r_base
        y_true = rates[-len(y_pred):].reshape(-1)  # ✅ 真实汇率直接取值
        
    else:
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
    plot_prediction_vs_true(y_true, y_pred)
    # ===== 指标计算（含保护机制）=====
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    maxae = np.max(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"AAE       : {mae:.6f}")
    print(f"MAPE (%)  : {mape:.4f}")
    print(f"MSE       : {mse:.6e}")
    print(f"Max AE    : {maxae:.5f}")
    print(f"R-Squared : {r2:.5f}")

    # ===== 可视化训练过程曲线 =====
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")

    plt.show()

import matplotlib.pyplot as plt
def plot_prediction_vs_true(y_true, y_pred, title="Prediction vs. True (Test Set)", start_index=0):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="True", color='black')
        plt.plot(y_pred, label="Predicted", color='red', linestyle='--')
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Exchange Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("history.png")
        plt.show()




# ========== 主入口 ==========
if __name__ == '__main__':
    import os
    # 使用相对路径，相对于项目根目录
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final", "ukcndata.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, rates = load_and_split_data(path)
#     run_baseline_nn(
#     hidden_layers=1,
#     hidden_units=8,
#     activation='tanh',
#     lr=0.1,
#     momentum=0.3,
#     rates=rates,
#     log_returns=True
# )
    
    run_gann_structure(feedback=False, loss_mode='MAE',rates=rates,log_returns=True)   # GFF-AEL
    run_gann_structure(feedback=True,  loss_mode='MAE',rates=rates,log_returns=True)   # GFB-AEL
    run_gann_structure(feedback=False, loss_mode='MSE',rates=rates,log_returns=True)   # GFF-SEL
    run_gann_structure(feedback=True,  loss_mode='MSE',rates=rates,log_returns=True)   # GFB-SEL
