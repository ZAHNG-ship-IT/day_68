import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class VolterraModel:
    """
    沃尔泰拉捕食者-猎物模型 (Volterra Predator-Prey Model)

    描述两个物种间的相互作用：
    - 猎物(x): 在没有捕食者时指数增长
    - 捕食者(y): 依赖猎物生存

    微分方程组:
    dx/dt = ax - bxy  (猎物增长率 - 被捕食率)
    dy/dt = -cy + dxy (捕食者死亡率 + 捕食收益)
    """

    def __init__(self, a=1.0, b=0.5, c=1.0, d=0.5):
        """
        初始化模型参数

        参数:
        a: 猎物自然增长率 (>0)
        b: 捕食效率系数 (>0)
        c: 捕食者自然死亡率 (>0)
        d: 捕食转化效率 (>0)
        """
        self.a = a  # 猎物增长率
        self.b = b  # 捕食效率
        self.c = c  # 捕食者死亡率
        self.d = d  # 转化效率

        # 平衡点计算
        self.equilibrium_x = c / d  # 猎物平衡密度
        self.equilibrium_y = a / b  # 捕食者平衡密度

        print(f"模型参数: a={a}, b={b}, c={c}, d={d}")
        print(f"平衡点: 猎物={self.equilibrium_x:.3f}, 捕食者={self.equilibrium_y:.3f}")

    def volterra_equations(self, t, y):
        """
        沃尔泰拉方程组

        参数:
        t: 时间
        y: [x, y] 猎物和捕食者密度

        返回:
        dydt: 密度变化率 [dx/dt, dy/dt]
        """
        x, y_pred = y

        # 确保密度非负
        x = max(x, 0)
        y_pred = max(y_pred, 0)

        # 沃尔泰拉方程
        dxdt = self.a * x - self.b * x * y_pred  # 猎物变化率
        dydt = -self.c * y_pred + self.d * x * y_pred  # 捕食者变化率

        return [dxdt, dydt]

    def simulate(self, x0, y0, t_span=(0, 20), t_eval=None):
        """
        模拟种群动态

        参数:
        x0: 猎物初始密度
        y0: 捕食者初始密度
        t_span: 时间范围 (开始时间, 结束时间)
        t_eval: 评估时间点

        返回:
        t: 时间数组
        x: 猎物密度时间序列
        y: 捕食者密度时间序列
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        # 求解微分方程
        sol = solve_ivp(
            self.volterra_equations,
            t_span,
            [x0, y0],
            t_eval=t_eval,
            method='RK45',  # 4阶龙格-库塔方法
            rtol=1e-8
        )

        return sol.t, sol.y[0], sol.y[1]

    def phase_portrait(self, x_range=(0, 6), y_range=(0, 6), num_trajectories=8):
        """
        绘制相平面图

        参数:
        x_range: 猎物密度范围
        y_range: 捕食者密度范围
        num_trajectories: 轨迹数量
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建网格用于绘制方向场
        x = np.linspace(x_range[0], x_range[1], 20)
        y = np.linspace(y_range[0], y_range[1], 20)
        X, Y = np.meshgrid(x, y)

        # 计算方向场
        DX = self.a * X - self.b * X * Y
        DY = -self.c * Y + self.d * X * Y

        # 归一化箭头长度
        M = np.sqrt(DX**2 + DY**2)
        M[M == 0] = 1  # 避免除零
        DX_norm = DX / M
        DY_norm = DY / M

        # 绘制方向场
        ax.quiver(X, Y, DX_norm, DY_norm, M, alpha=0.6, cmap='viridis', scale=30)

        # 绘制多条轨迹
        colors = plt.cm.Set1(np.linspace(0, 1, num_trajectories))

        for i in range(num_trajectories):
            # 随机选择初始条件
            x0 = np.random.uniform(x_range[0] + 0.5, x_range[1] - 0.5)
            y0 = np.random.uniform(y_range[0] + 0.5, y_range[1] - 0.5)

            # 模拟轨迹
            t, x_traj, y_traj = self.simulate(x0, y0, t_span=(0, 15))

            # 绘制轨迹
            ax.plot(x_traj, y_traj, color=colors[i], linewidth=2, alpha=0.8)
            ax.plot(x0, y0, 'o', color=colors[i], markersize=6)  # 起点

        # 标记平衡点
        ax.plot(self.equilibrium_x, self.equilibrium_y, 'r*',
                markersize=15, label=f'平衡点({self.equilibrium_x:.2f}, {self.equilibrium_y:.2f})')

        ax.set_xlabel('猎物密度 (x)', fontsize=12)
        ax.set_ylabel('捕食者密度 (y)', fontsize=12)
        ax.set_title('沃尔泰拉模型相平面图', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        plt.tight_layout()
        plt.show()

    def time_series_plot(self, x0, y0, t_span=(0, 20)):
        """
        绘制时间序列图

        参数:
        x0: 猎物初始密度
        y0: 捕食者初始密度
        t_span: 时间范围
        """
        t, x, y = self.simulate(x0, y0, t_span)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 时间序列图
        ax1.plot(t, x, 'b-', linewidth=2, label='猎物')
        ax1.plot(t, y, 'r-', linewidth=2, label='捕食者')
        ax1.axhline(y=self.equilibrium_x, color='b', linestyle='--', alpha=0.7, label='猎物平衡值')
        ax1.axhline(y=self.equilibrium_y, color='r', linestyle='--', alpha=0.7, label='捕食者平衡值')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('种群密度')
        ax1.set_title(f'沃尔泰拉模型时间演化 (初值: x₀={x0}, y₀={y0})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 相轨迹图
        ax2.plot(x, y, 'g-', linewidth=2)
        ax2.plot(x0, y0, 'go', markersize=8, label='起点')
        ax2.plot(self.equilibrium_x, self.equilibrium_y, 'r*', markersize=12, label='平衡点')
        ax2.set_xlabel('猎物密度')
        ax2.set_ylabel('捕食者密度')
        ax2.set_title('相空间轨迹')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def stability_analysis(self):
        """
        稳定性分析：计算雅可比矩阵和特征值
        """
        print("\n=== 稳定性分析 ===")
        print(f"平衡点: ({self.equilibrium_x:.3f}, {self.equilibrium_y:.3f})")

        # 在平衡点处的雅可比矩阵
        # J = [[∂f₁/∂x, ∂f₁/∂y],
        #      [∂f₂/∂x, ∂f₂/∂y]]
        J = np.array([
            [0, -self.b * self.equilibrium_x],
            [self.d * self.equilibrium_y, 0]
        ])

        print(f"雅可比矩阵:\n{J}")

        # 计算特征值
        eigenvalues = np.linalg.eigvals(J)
        print(f"特征值: {eigenvalues}")

        # 判断稳定性
        real_parts = np.real(eigenvalues)
        if np.all(real_parts == 0):
            print("系统类型: 中心点 (周期轨道)")
        elif np.all(real_parts < 0):
            print("系统类型: 稳定焦点")
        elif np.any(real_parts > 0):
            print("系统类型: 不稳定")

        return eigenvalues

    def conservation_law(self, x, y):
        """
        守恒量计算：沃尔泰拉模型的首次积分
        H(x,y) = d*x - c*ln(x) + b*y - a*ln(y)
        """
        # 避免对零或负数取对数
        x = np.maximum(x, 1e-10)
        y = np.maximum(y, 1e-10)

        H = self.d * x - self.c * np.log(x) + self.b * y - self.a * np.log(y)
        return H

# 扩展模型：考虑环境阻力的修正沃尔泰拉模型
class ModifiedVolterraModel(VolterraModel):
    """
    修正沃尔泰拉模型：考虑环境阻力和竞争

    dx/dt = ax(1 - x/K) - bxy  (logistic增长 + 捕食)
    dy/dt = -cy + dxy - ey²   (捕食收益 + 种内竞争)
    """

    def __init__(self, a=1.0, b=0.5, c=1.0, d=0.5, K=10.0, e=0.1):
        super().__init__(a, b, c, d)
        self.K = K  # 环境容量
        self.e = e  # 捕食者种内竞争系数

    def volterra_equations(self, t, y):
        """修正的沃尔泰拉方程组"""
        x, y_pred = y
        x = max(x, 0)
        y_pred = max(y_pred, 0)

        # 修正方程
        dxdt = self.a * x * (1 - x/self.K) - self.b * x * y_pred
        dydt = -self.c * y_pred + self.d * x * y_pred - self.e * y_pred**2

        return [dxdt, dydt]

# 使用示例和测试
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False

    print("=== 沃尔泰拉捕食者-猎物模型 ===\n")

    # 1. 基础模型测试
    print("1. 经典沃尔泰拉模型")
    model = VolterraModel(a=1.0, b=0.5, c=0.8, d=0.3)

    # 稳定性分析
    eigenvals = model.stability_analysis()

    # 时间序列分析
    print("\n2. 时间序列模拟")
    model.time_series_plot(x0=4, y0=3, t_span=(0, 20))

    # 相平面分析
    print("\n3. 相平面分析")
    model.phase_portrait(x_range=(0, 8), y_range=(0, 6), num_trajectories=6)

    # 守恒量验证
    print("\n4. 守恒量验证")
    t, x, y = model.simulate(x0=4, y0=3, t_span=(0, 10))
    H = model.conservation_law(x, y)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, H)
    plt.title('守恒量随时间变化')
    plt.xlabel('时间')
    plt.ylabel('H(x,y)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, x, label='猎物')
    plt.plot(t, y, label='捕食者')
    plt.title('种群动态')
    plt.xlabel('时间')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"守恒量变化范围: {np.min(H):.6f} ~ {np.max(H):.6f}")
    print(f"守恒量标准差: {np.std(H):.8f} (理论值应为0)")

    # 5. 修正模型测试
    print("\n5. 修正沃尔泰拉模型 (考虑环境阻力)")
    modified_model = ModifiedVolterraModel(a=1.5, b=0.8, c=1.0, d=0.6, K=8, e=0.2)
    modified_model.time_series_plot(x0=3, y0=2, t_span=(0, 25))
    modified_model.phase_portrait(x_range=(0, 10), y_range=(0, 8))