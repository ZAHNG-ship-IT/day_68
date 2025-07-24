import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class MediterraneanSharkAlgorithm:
    """
    地中海鲨鱼算法(Mediterranean Shark Algorithm, MSA)

    模拟鲨鱼在地中海中的觅食行为来求解优化问题
    包含三种主要行为：游泳、觅食、攻击
    """

    def __init__(self,
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[float, float],
                 pop_size: int = 30,
                 max_iter: int = 100):
        """
        初始化地中海鲨鱼算法参数

        参数:
        objective_func: 目标函数
        dim: 问题维度
        bounds: 变量边界 (下界, 上界)
        pop_size: 鲨鱼群体大小
        max_iter: 最大迭代次数
        """
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter

        # 算法控制参数
        self.alpha = 2.0      # 游泳行为强度系数
        self.beta = 0.5       # 觅食行为强度系数
        self.gamma = 1.0      # 攻击行为强度系数

        # 初始化种群
        self.sharks = None
        self.fitness = None
        self.best_shark = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

    def initialize_population(self):
        """初始化鲨鱼种群位置"""
        lower, upper = self.bounds
        self.sharks = np.random.uniform(
            lower, upper, (self.pop_size, self.dim)
        )

        # 计算初始适应度
        self.fitness = np.array([
            self.objective_func(shark) for shark in self.sharks
        ])

        # 找到初始最优鲨鱼
        best_idx = np.argmin(self.fitness)
        self.best_shark = self.sharks[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def swimming_behavior(self, shark_idx: int, iteration: int) -> np.ndarray:
        """
        鲨鱼游泳行为：在搜索空间中随机游动

        参数:
        shark_idx: 鲨鱼索引
        iteration: 当前迭代次数

        返回:
        new_position: 新位置
        """
        current_shark = self.sharks[shark_idx]

        # 自适应游泳强度：随迭代次数减小
        swim_intensity = self.alpha * (1 - iteration / self.max_iter)

        # 随机游泳方向
        swim_direction = np.random.uniform(-1, 1, self.dim)

        # 计算新位置
        new_position = current_shark + swim_intensity * swim_direction

        return self._boundary_check(new_position)

    def foraging_behavior(self, shark_idx: int) -> np.ndarray:
        """
        鲨鱼觅食行为：向食物丰富区域移动

        参数:
        shark_idx: 鲨鱼索引

        返回:
        new_position: 新位置
        """
        current_shark = self.sharks[shark_idx]

        # 选择一个随机的更优鲨鱼作为觅食目标
        better_sharks = self.fitness < self.fitness[shark_idx]
        if np.any(better_sharks):
            better_indices = np.where(better_sharks)[0]
            target_idx = np.random.choice(better_indices)
            target_shark = self.sharks[target_idx]
        else:
            # 如果没有更优的鲨鱼，向最优鲨鱼移动
            target_shark = self.best_shark

        # 觅食移动
        forage_vector = target_shark - current_shark
        new_position = current_shark + self.beta * forage_vector * np.random.random()

        return self._boundary_check(new_position)

    def attacking_behavior(self, shark_idx: int) -> np.ndarray:
        """
        鲨鱼攻击行为：快速向最优位置冲刺

        参数:
        shark_idx: 鲨鱼索引

        返回:
        new_position: 新位置
        """
        current_shark = self.sharks[shark_idx]

        # 向全局最优位置攻击
        attack_vector = self.best_shark - current_shark

        # 攻击强度随机化
        attack_intensity = self.gamma * np.random.random()

        new_position = current_shark + attack_intensity * attack_vector

        return self._boundary_check(new_position)

    def _boundary_check(self, position: np.ndarray) -> np.ndarray:
        """边界检查：确保位置在可行域内"""
        lower, upper = self.bounds
        return np.clip(position, lower, upper)

    def update_shark_position(self, shark_idx: int, iteration: int):
        """
        更新单个鲨鱼位置
        根据概率选择不同的行为模式

        参数:
        shark_idx: 鲨鱼索引
        iteration: 当前迭代次数
        """
        # 行为选择概率
        behavior_prob = np.random.random()

        if behavior_prob < 0.4:
            # 40%概率执行游泳行为
            new_position = self.swimming_behavior(shark_idx, iteration)
        elif behavior_prob < 0.7:
            # 30%概率执行觅食行为
            new_position = self.foraging_behavior(shark_idx)
        else:
            # 30%概率执行攻击行为
            new_position = self.attacking_behavior(shark_idx)

        # 计算新位置的适应度
        new_fitness = self.objective_func(new_position)

        # 贪婪选择：如果新位置更好则更新
        if new_fitness < self.fitness[shark_idx]:
            self.sharks[shark_idx] = new_position
            self.fitness[shark_idx] = new_fitness

            # 更新全局最优解
            if new_fitness < self.best_fitness:
                self.best_shark = new_position.copy()
                self.best_fitness = new_fitness

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行优化过程

        返回:
        best_solution: 最优解
        best_value: 最优值
        """
        # 初始化种群
        self.initialize_population()
        self.convergence_curve.append(self.best_fitness)

        print(f"初始最优值: {self.best_fitness:.6f}")

        # 主循环
        for iteration in range(self.max_iter):
            # 更新每个鲨鱼的位置
            for shark_idx in range(self.pop_size):
                self.update_shark_position(shark_idx, iteration)

            # 记录收敛曲线
            self.convergence_curve.append(self.best_fitness)

            # 打印进度
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration + 1}: 最优值 = {self.best_fitness:.6f}")

        print(f"优化完成！最终最优值: {self.best_fitness:.6f}")
        return self.best_shark, self.best_fitness

    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('最优值')
        plt.title('地中海鲨鱼算法收敛曲线')
        plt.grid(True, alpha=0.3)
        plt.show()

# 测试函数
def sphere_function(x):
    """球面函数：f(x) = sum(x^2)"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin函数：多峰函数"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock函数：经典测试函数"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# 使用示例
if __name__ == "__main__":
    print("=== 地中海鲨鱼算法测试 ===\n")

    # 测试球面函数
    print("1. 测试球面函数 (全局最优值: 0)")
    msa1 = MediterraneanSharkAlgorithm(
        objective_func=sphere_function,
        dim=10,
        bounds=(-100, 100),
        pop_size=30,
        max_iter=100
    )
    best_solution1, best_value1 = msa1.optimize()
    print(f"最优解: {best_solution1}")
    print(f"最优值: {best_value1:.6f}\n")

    # 测试Rastrigin函数
    print("2. 测试Rastrigin函数 (全局最优值: 0)")
    msa2 = MediterraneanSharkAlgorithm(
        objective_func=rastrigin_function,
        dim=10,
        bounds=(-5.12, 5.12),
        pop_size=50,
        max_iter=200
    )
    best_solution2, best_value2 = msa2.optimize()
    print(f"最优解: {best_solution2}")
    print(f"最优值: {best_value2:.6f}\n")

    # 绘制收敛曲线
    msa2.plot_convergence()

    # 测试Rosenbrock函数
    print("3. 测试Rosenbrock函数 (全局最优值: 0)")
    msa3 = MediterraneanSharkAlgorithm(
        objective_func=rosenbrock_function,
        dim=5,
        bounds=(-2, 2),
        pop_size=40,
        max_iter=300
    )
    best_solution3, best_value3 = msa3.optimize()
    print(f"最优解: {best_solution3}")
    print(f"最优值: {best_value3:.6f}")