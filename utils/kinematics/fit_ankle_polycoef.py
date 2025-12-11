"""
拟合踝关节并联机构的多项式系数
用于MuJoCo equality约束
"""
import numpy as np
from tdpm import ThreeDOFParallelMechanism
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def generate_training_data(tdpm: ThreeDOFParallelMechanism,
                          n_samples: int = 1000,
                          angle_range: float = 0.2):
    """
    生成训练数据：mu -> theta映射

    参数:
        tdpm: 并联机构实例
        n_samples: 采样点数
        angle_range: 姿态角范围（弧度）
    """
    # 在工作空间内随机采样姿态角
    mu_samples = np.random.uniform(-angle_range, angle_range, (n_samples, 3))

    theta_samples = np.zeros((n_samples, 3))

    for i, mu in enumerate(mu_samples):
        result = tdpm.forward_kinematics(mu)
        theta_samples[i] = [
            result['theta_a'],
            result['theta_b'],
            result['theta_c']
        ]

    return theta_samples, mu_samples


def fit_polynomial_inverse(tdpm: ThreeDOFParallelMechanism,
                          degree: int = 3,
                          n_samples: int = 5000):
    """
    拟合逆运动学的多项式近似
    theta -> mu

    返回:
        poly_features: 多项式特征转换器
        models: [model_roll, model_pitch, model_yaw]
    """
    print(f"生成 {n_samples} 个训练样本...")
    theta_data, mu_data = generate_training_data(tdpm, n_samples)

    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    theta_poly = poly.fit_transform(theta_data)

    print(f"多项式特征数量: {theta_poly.shape[1]}")

    # 为每个输出分量拟合模型
    models = []
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        model = LinearRegression()
        model.fit(theta_poly, mu_data[:, i])

        # 评估拟合质量
        score = model.score(theta_poly, mu_data[:, i])
        print(f"{axis}: R² = {score:.6f}")

        models.append(model)

    return poly, models


def evaluate_approximation(tdpm: ThreeDOFParallelMechanism,
                          poly, models,
                          n_test: int = 1000):
    """
    评估多项式近似的精度
    """
    theta_test, mu_true = generate_training_data(tdpm, n_test)
    theta_poly = poly.transform(theta_test)

    mu_pred = np.column_stack([
        model.predict(theta_poly) for model in models
    ])

    errors = mu_pred - mu_true

    print("\n=== 近似误差统计 ===")
    print(f"Roll  - MAE: {np.mean(np.abs(errors[:, 0])):.6f}, "
          f"Max: {np.max(np.abs(errors[:, 0])):.6f}")
    print(f"Pitch - MAE: {np.mean(np.abs(errors[:, 1])):.6f}, "
          f"Max: {np.max(np.abs(errors[:, 1])):.6f}")
    print(f"Yaw   - MAE: {np.mean(np.abs(errors[:, 2])):.6f}, "
          f"Max: {np.max(np.abs(errors[:, 2])):.6f}")

    return errors


def extract_mujoco_coefficients(poly, models, max_degree: int = 4):
    """
    提取MuJoCo可用的多项式系数

    MuJoCo的polycoef格式: [c0, c1, c2, c3, c4]
    表示: c0 + c1*q + c2*q^2 + c3*q^3 + c4*q^4

    但这只能表示单变量多项式！
    所以我们需要对每个输入分别建立关系
    """
    print("\n=== 警告 ===")
    print("MuJoCo的equality-joint只支持单变量多项式映射")
    print("3-DOF并联机构需要多变量耦合，无法直接用polycoef表示")
    print("\n推荐方案:")
    print("1. 使用MuJoCo的equality-connect约束（位置约束）")
    print("2. 使用Python回调函数实现实时逆运动学")
    print("3. 如果必须用joint equality，只能做解耦近似")

    # 尝试解耦近似（假设每个mu主要由对应的theta控制）
    print("\n=== 解耦近似系数（仅供参考，精度有限）===")

    # 提取线性项（最简单的近似）
    for i, axis in enumerate(['roll (cube)', 'pitch (axle)', 'yaw (foot)']):
        coef = models[i].coef_
        intercept = models[i].intercept_

        # 多项式特征的顺序通常是：
        # [1, theta_a, theta_b, theta_c, theta_a^2, theta_a*theta_b, ...]
        print(f"\n{axis}:")
        print(f"  截距: {intercept:.6f}")
        print(f"  theta_a系数: {coef[1]:.6f}")
        print(f"  theta_b系数: {coef[2]:.6f}")
        print(f"  theta_c系数: {coef[3]:.6f}")


if __name__ == "__main__":
    # 从MJCF测量的实际参数（需要你填入）
    l0 = 0.02  # 平台半径，从MJCF geometry测量
    l1 = 0.02  # 动平台到O点距离
    l2 = 0.02  # 静平台到O点距离

    print("初始化3-DOF并联机构...")
    tdpm = ThreeDOFParallelMechanism(l0, l1, l2)

    # 拟合多项式
    poly, models = fit_polynomial_inverse(tdpm, degree=3, n_samples=5000)

    # 评估精度
    evaluate_approximation(tdpm, poly, models)

    # 提取系数
    extract_mujoco_coefficients(poly, models)
