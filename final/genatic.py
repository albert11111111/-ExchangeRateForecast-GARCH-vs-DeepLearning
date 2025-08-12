import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools
import random

# ========== 数据准备 ==========
import os
# 使用相对路径，相对于项目根目录
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final", "ukcndata.csv")
data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
rates = data['Rate'].values

scaler = MinMaxScaler(feature_range=(0, 1))
rates_scaled = scaler.fit_transform(rates.reshape(-1, 1))

def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(rates_scaled)
# ⚠️ 前馈网络不需要 reshape 为 3D！
# X: (samples, time_steps), y: (samples,)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ========== 构建前馈网络模型 ==========
def create_ffn_model(input_dim, num_layers, num_neurons, learning_rate):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=input_dim, activation='relu'))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# ========== 遗传算法设置 ==========
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# 个体基因定义：层数（1~3），节点数（4~64），学习率（0.001~0.1）
toolbox.register("attr_num_layers", random.randint, 1, 3)
toolbox.register("attr_num_neurons", random.randint, 4, 64)
toolbox.register("attr_learning_rate", random.uniform, 0.001, 0.1)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_num_layers, toolbox.attr_num_neurons, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 适应度函数（使用 AEL）
def evaluate(individual):
    num_layers = int(round(individual[0]))
    num_neurons = int(round(individual[1]))
    learning_rate = float(individual[2])

    # 合理范围控制
    num_layers = max(1, min(3, num_layers))
    num_neurons = max(4, min(128, num_neurons))
    learning_rate = max(0.0001, min(0.1, learning_rate))

    try:
        model = create_ffn_model(X.shape[1], num_layers, num_neurons, learning_rate)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        predictions = model.predict(X).reshape(-1)

        # 关键：检查是否包含 NaN
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return (np.inf,)  # 给一个极大惩罚，避免程序报错

        ael = mean_absolute_error(y, predictions)
        return (ael,)

    except Exception as e:
        print(f"Evaluation error: {e}")
        return (np.inf,)  # 出错的个体适应度设为最差



toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ========== 运行遗传算法 ==========
population = toolbox.population(n=10)
NGEN = 30
CXPB, MUTPB = 0.7, 0.2

best_fitness_per_gen = []

# Early stopping 参数
early_stop_patience = 3      # 连续多少代无明显提升就停止
early_stop_threshold = 1e-6    # 差异小于这个值认为无提升
no_improve_counter = 0         # 连续无提升代数
best_so_far = np.inf           # 当前最优适应度


for gen in range(NGEN):
    print(f"Generation {gen+1}")
    
    # 评估适应度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 记录当前代的最优适应度
    best_fitness = min([ind.fitness.values[0] for ind in population])
    best_fitness_per_gen.append(best_fitness)
    print(f"  Best fitness: {best_fitness:.5f}")

    # 选择、复制
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 变异
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新个体
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring
    # Early stopping 判断逻辑
    if best_fitness < best_so_far - early_stop_threshold:
        best_so_far = best_fitness
        no_improve_counter = 0
    else:
        no_improve_counter += 1
        print(f"  No significant improvement for {no_improve_counter} generations")

    if no_improve_counter >= early_stop_patience:
        print(f"\n🛑 Early stopping: No improvement in {early_stop_patience} generations.")
        break


# # ========== 可视化收敛曲线 ==========
# plt.plot(best_fitness_per_gen, marker='o')
# plt.xlabel("Generation")
# plt.ylabel("Best Fitness (AEL)")
# plt.title("Convergence Curve of GANN (Feedforward Network)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 从最终种群中选择适应度最好的个体
best_ind = tools.selBest(population, 1)[0]
num_layers = int(round(best_ind[0]))
num_neurons = int(round(best_ind[1]))
learning_rate = float(best_ind[2])

# 重新训练模型
model = create_ffn_model(X.shape[1], num_layers, num_neurons, learning_rate)
model.fit(X, y, epochs=50, batch_size=32, verbose=0)
pred = model.predict(X).reshape(-1)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 安全检查
if np.any(np.isnan(pred)) or np.any(np.isnan(y)):
    print("❌ 错误：输入中包含 NaN，无法计算指标")
else:
    # MAPE 要求不能除以 0
    mask = y != 0
    mape = np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100 if np.any(mask) else np.nan

    aae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    max_ae = np.max(np.abs(y - pred))
    r_sq = r2_score(y, pred)

    print("\n📊 Final Model Evaluation Metrics:")
    print(f"AAE      : {aae:.6f}")
    print(f"MAPE (%) : {mape:.5f}")
    print(f"MSE      : {mse:.6e}")
    print(f"Max AE   : {max_ae:.5f}")
    print(f"R-SQ     : {r_sq:.5f}")
