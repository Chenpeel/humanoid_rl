import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, Dict


class ThreeDOFParallelMechanism:
    """
    三自由度并联机构运动学求解器
    3-DOF parallel mechanism
    """

    def __init__(self, l0: float, l1: float, l2: float):
        """
        参数:
            l0: 平台半径
            l1: 动平台到O点的距离
            l2: 静平台到O点的距离
        """
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2

        # 动平台点（在α平面）
        self.Mu = self._construct_platform_points(l0, -l1)
        # 静平台点初始位置（在β平面，δ=-π时）
        self.Nu = self._construct_platform_points(l0, l2)

    @staticmethod
    def _construct_platform_points(l0: float, z: float) -> np.ndarray:
        """构造正三角形平台的三个顶点"""
        sqrt3 = np.sqrt(3)
        return np.array([
            [0, -l0, z],                    # A 或 A'
            [-sqrt3/2*l0, l0/2, z],         # B 或 B'
            [sqrt3/2*l0, l0/2, z]           # C 或 C'
        ]).T

    @staticmethod
    def rotation_matrix(mu: np.ndarray) -> np.ndarray:
        """
        构造ZYX欧拉角旋转矩阵
        mu = [roll, pitch, yaw]
        """
        r, p, y = mu

        # 预计算三角函数
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        # R = Rz(y) * Ry(p) * Rx(r)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])

        return R

    @staticmethod
    def rotation_matrix_derivatives(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray]:
        """计算旋转矩阵对各轴的偏导数"""
        r, p, y = mu

        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        # ∂R/∂r
        dR_dr = np.array([
            [0, cy*sp*cr + sy*sr, -cy*sp*sr + sy*cr],
            [0, sy*sp*cr - cy*sr, -sy*sp*sr - cy*cr],
            [0, cp*cr,            -cp*sr]
        ])

        # ∂R/∂p
        dR_dp = np.array([
            [-cy*sp, cy*cp*sr, cy*cp*cr],
            [-sy*sp, sy*cp*sr, sy*cp*cr],
            [-cp,    -sp*sr,   -sp*cr]
        ])

        # ∂R/∂y
        dR_dy = np.array([
            [-sy*cp, -sy*sp*sr - cy*cr, -sy*sp*cr + cy*sr],
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [0,      0,                  0]
        ])

        return dR_dr, dR_dp, dR_dy

    def forward_kinematics(self, mu: np.ndarray, mu_z: float = 0.0) -> Dict:
        """
        正向运动学：从姿态角求解点位置和杆角度

        参数:
            mu: [roll, pitch, yaw] 姿态角
            mu_z: β平面绕自身法向量的旋转角

        返回:
            包含A', B', C'坐标和各杆角度的字典
        """
        # 1. 计算旋转矩阵
        R = self.rotation_matrix(mu)

        # 2. 计算L2位置和β平面法向量
        n_beta = R @ np.array([0, 0, 1])
        OL2 = self.l2 * n_beta

        # 3. 构造β平面内的旋转矩阵（绕n_beta旋转mu_z）
        # 使用Rodrigues旋转公式
        K = self._skew_symmetric(n_beta)
        R_beta = np.eye(3) + np.sin(mu_z) * K + (1 - np.cos(mu_z)) * K @ K

        # 4. 计算旋转后的静平台点
        Nu_rotated = R_beta @ self.Nu

        # 5. 静平台点的全局坐标
        Nu_prime = OL2[:, np.newaxis] + Nu_rotated

        # 6. 计算杆的向量和长度
        rods = Nu_prime - self.Mu
        rod_lengths = np.linalg.norm(rods, axis=0)

        # 7. 计算角度
        z_axis = np.array([0, 0, 1])

        angles = {}
        labels = ['a', 'b', 'c']

        for i, label in enumerate(labels):
            rod = rods[:, i]
            rod_unit = rod / rod_lengths[i]

            # φ: 杆与z轴(OL1方向)的夹角
            phi = np.arccos(np.clip(rod_unit @ z_axis, -1, 1))

            # θ: 杆与β平面的夹角
            # 杆与法向量的夹角的余角
            cos_angle = np.clip(rod_unit @ n_beta, -1, 1)
            theta = np.pi/2 - np.arccos(cos_angle)

            angles[f'phi_{label}'] = phi
            angles[f'theta_{label}'] = theta

        return {
            'A_prime': Nu_prime[:, 0],
            'B_prime': Nu_prime[:, 1],
            'C_prime': Nu_prime[:, 2],
            'rod_lengths': rod_lengths,
            'n_beta': n_beta,
            'OL2': OL2,
            **angles
        }

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """构造向量的斜对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def compute_jacobian(self, mu: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵 J = ∂F/∂μ

        参数:
            mu: 当前姿态角 [r, p, y]

        返回:
            3x3雅可比矩阵
        """
        R = self.rotation_matrix(mu)
        dR_dr, dR_dp, dR_dy = self.rotation_matrix_derivatives(mu)

        J = np.zeros((3, 3))

        for i in range(3):
            eta_i = self.Nu[:, i]
            gamma_i = self.Mu[:, i]

            # ∂f_i/∂r
            J[i, 0] = eta_i.T @ dR_dr.T @ gamma_i

            # ∂f_i/∂p
            J[i, 1] = eta_i.T @ dR_dp.T @ gamma_i

            # ∂f_i/∂y
            J[i, 2] = eta_i.T @ dR_dy.T @ gamma_i

        return J

    def constraint_equations(self, mu: np.ndarray) -> np.ndarray:
        """
        约束方程 F(μ) = 0
        F_i = (η'_i - γ_i) · γ_i
        """
        R = self.rotation_matrix(mu)
        Nu_prime = R @ self.Nu

        F = np.zeros(3)
        for i in range(3):
            rod = Nu_prime[:, i] - self.Mu[:, i]
            F[i] = rod @ self.Mu[:, i]

        return F

    def inverse_kinematics_newton(self,
                                  target_angles: Dict[str, float] = None,
                                  target_positions: np.ndarray = None,
                                  mu_init: np.ndarray = None,
                                  max_iter: int = 50,
                                  tol: float = 1e-6) -> Dict:
        """
        逆向运动学：Newton-Raphson迭代法

        参数:
            target_angles: 目标角度字典 {'theta_a': ..., 'theta_b': ..., 'theta_c': ...}
            target_positions: 目标位置 3x3数组（优先级高于角度）
            mu_init: 初始猜测
            max_iter: 最大迭代次数
            tol: 收敛容差

        返回:
            求解结果字典
        """
        if mu_init is None:
            mu_init = np.zeros(3)

        mu = mu_init.copy()

        for iteration in range(max_iter):
            # 计算约束方程值
            F = self.constraint_equations(mu)

            # 检查收敛
            if np.linalg.norm(F) < tol:
                result = self.forward_kinematics(mu)
                result['mu'] = mu
                result['iterations'] = iteration
                result['converged'] = True
                return result

            # 计算雅可比矩阵
            J = self.compute_jacobian(mu)

            # 检查奇异性
            if np.linalg.cond(J) > 1e10:
                print(f"警告：雅可比矩阵接近奇异，条件数 = {np.linalg.cond(J)}")

            # Newton-Raphson更新
            try:
                delta_mu = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                print("雅可比矩阵奇异，使用伪逆")
                delta_mu = np.linalg.pinv(J) @ (-F)

            # 更新姿态角
            mu = mu + delta_mu

            # 限制在工作空间内
            mu = np.clip(mu, -np.pi/6, np.pi/6)

        print(f"警告：未收敛，最终误差 = {np.linalg.norm(F)}")
        result = self.forward_kinematics(mu)
        result['mu'] = mu
        result['iterations'] = max_iter
        result['converged'] = False
        return result

    def inverse_kinematics_scipy(self,
                                 target_positions: np.ndarray = None,
                                 mu_init: np.ndarray = None) -> Dict:
        """
        使用scipy的fsolve求解逆运动学
        """
        if mu_init is None:
            mu_init = np.zeros(3)

        # 定义目标函数（如果提供了目标位置）
        if target_positions is not None:
            def objective(mu):
                R = self.rotation_matrix(mu)
                Nu_prime = R @ self.Nu
                return (Nu_prime - target_positions).flatten()
        else:
            objective = self.constraint_equations

        # 求解
        mu_solution, info, ier, msg = fsolve(
            objective, mu_init, full_output=True)

        result = self.forward_kinematics(mu_solution)
        result['mu'] = mu_solution
        result['converged'] = (ier == 1)
        result['message'] = msg

        return result

    def velocity_kinematics(self, mu: np.ndarray, mu_dot: np.ndarray) -> Dict:
        """
        速度运动学：计算杆端点的速度

        v = J * μ_dot

        参数:
            mu: 当前姿态角
            mu_dot: 姿态角速度

        返回:
            端点速度字典
        """
        J = self.compute_jacobian(mu)

        # 计算约束空间的速度
        F_dot = J @ mu_dot

        # 计算实际端点速度（需要更复杂的推导）
        R = self.rotation_matrix(mu)
        dR_dr, dR_dp, dR_dy = self.rotation_matrix_derivatives(mu)

        # ω = [mu_dot[0], mu_dot[1], mu_dot[2]]对应的旋转矩阵导数
        R_dot = (dR_dr * mu_dot[0] + dR_dp * mu_dot[1] + dR_dy * mu_dot[2])

        Nu_prime_dot = R_dot @ self.Nu

        return {
            'F_dot': F_dot,
            'A_prime_dot': Nu_prime_dot[:, 0],
            'B_prime_dot': Nu_prime_dot[:, 1],
            'C_prime_dot': Nu_prime_dot[:, 2],
            'jacobian': J
        }
