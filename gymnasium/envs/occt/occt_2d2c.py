import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import yaml  
from model import Model2D2C
import datetime

class TwoCarrierEnv(gym.Env):
    """两辆车运载超大件系统的自定义强化学习环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, config_path=None):
        super().__init__()
        
        # 加载2d2c.yaml配置文件（优先使用传入路径，否则加载同一目录下的2d2c.yaml）
        self.config_name = '2d2c'
        self.config = self._load_config(config_path)
        
        # 初始化动力学模型（使用加载的config）
        self.model = Model2D2C(self.config)
        
        # 状态空间定义（观测：两车状态+铰接力）
        # 状态向量包含：超大件位置/姿态、两车姿态、各部分速度（共10维）+ 铰接力（2维）
        obs_low = np.array([
            -100, -100, -np.pi,  # 超大件 X, Y, Psi_o
            -np.pi, -np.pi,      # 两车姿态 Psi_1, Psi_2
            -10, -10, -5,        # 超大件速度 X_dot, Y_dot, Psi_dot_o
            -5, -5,              # 两车角速度 Psi_dot_1, Psi_dot_2
            -1e4, -1e4           # 铰接力 Fh_x, Fh_y（第二辆车）
        ])
        obs_high = np.array([
            100, 100, np.pi,
            np.pi, np.pi,
            10, 10, 5,
            5, 5,
            1e4, 1e4
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float64
        )
        
        # 动作空间定义（第二辆车的控制量：4个参数）
        # [前轮转向角, 后轮转向角, 前轮推力, 后轮推力]
        self.action_space = spaces.Box(
            low=np.array([-np.pi/6, -np.pi/6, 0, 0]),  # 转向角限制±30°，推力非负
            high=np.array([np.pi/6, np.pi/6, 1e3, 1e3]),  # 推力上限示例值
            dtype=np.float64
        )
        
        # 第一辆车的固定控制量（可移到2d2c.yaml中配置）
        self.u1_fixed = np.array([0, 0, 1e3, 1e3])  # 示例：直驶，固定推力
        
        # 渲染相关
        self.render_mode = render_mode
        self.render_frames = []

    def _load_config(self, config_path):
        """加载同一目录下的2d2c.yaml配置文件"""
        # 若未传入config路径，默认加载同一目录下的2d2c.yaml
        if config_path is None:
            # 获取当前脚本所在绝对目录，避免相对路径错乱
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "2d2c.yaml")
        
        # 检查配置文件是否存在，不存在则使用默认配置兜底
        if not os.path.exists(config_path):
            print(f"警告：未找到配置文件 {config_path}，将使用默认配置")
            return self._get_default_config()
        
        # 读取并解析YAML配置文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print(f"成功加载YAML配置文件：{config_path}")
            return config
        except Exception as e:
            print(f"错误：解析2d2c.yaml失败，原因：{e}，将使用默认配置")
            return self._get_default_config()

    def _get_default_config(self):
        """默认仿真配置（兜底使用，与YAML配置格式一致）"""
        return {
            'N_c': 2, 'N_q': 5, 'N_x': 10, 'N_u': 8,  # 双载体配置
            'M_o': 1000, 'I_o': 1000,                 # 超大件参数
            'M_1': 500, 'M_2': 500,                   # 车辆质量
            'I_1': 100, 'I_2': 100,                   # 转动惯量
            'x__o_1': 5, 'x__o_2': -5,                # 铰链相对超大件坐标
            'y__o_1': 0, 'y__o_2': 0,
            'x__1_1': 1, 'x__2_2': 1,                 # 质心相对自身铰链坐标
            'y__1_1': 0, 'y__2_2': 0,
            'C_f': 10000, 'C_r': 10000,               # 轮胎刚度
            'l_f': 2, 'l_r': 2,                       # 前后轮距质心距离
            'X_o_0': 0, 'Y_o_0': 0, 'Psi_o_0': 0,     # 初始状态
            'Psi_1_0': 0, 'Psi_2_0': 0,
            'X_dot_o_0': 0, 'Y_dot_o_0': 0, 'Psi_dot_o_0': 0,
            'Psi_dot_1_0': 0, 'Psi_dot_2_0': 0,
            'T': 10, 'dt': 0.1, 'integrator': 'RK4',  # 仿真参数
            'framerate': 10, 'range': 20,             # 可视化参数
            'oversized_cargo_bias': 2, 'oversized_cargo_width': 3
        }

    def _get_observation(self):
        """从模型提取观测值"""
        x = self.model.x  # 当前状态向量
        # 提取第二辆车的铰接力（Fh_arch存储格式：[Fh1_x, Fh1_y, Fh2_x, Fh2_y]）
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        return np.concatenate([
            x[:5],   # 位置与姿态：[Xo, Yo, Psio, Psi1, Psi2]
            x[5:10], # 速度：[Xo_dot, Yo_dot, Psio_dot, Psi1_dot, Psi2_dot]
            [Fh2_x, Fh2_y]  # 第二辆车铰接力
        ])

    def _calculate_reward(self):
        """计算奖励（核心：最小化铰接力+跟随约束）"""
        # 1. 铰接力惩罚（核心目标）
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        hinge_force_penalty = 0.001 * (Fh2_x**2 + Fh2_y**2)  # 缩放因子避免数值过大
        
        # 2. 跟随误差惩罚（位置+姿态）
        X1, Y1 = self.model.getXYi(self.model.x, 0)  # 第一辆车位置
        X2, Y2 = self.model.getXYi(self.model.x, 1)  # 第二辆车位置
        Psi1 = self.model.x[3]
        Psi2 = self.model.x[4]
        pos_error = np.hypot(X2 - X1, Y2 - Y1)
        psi_error = np.abs(Psi2 - Psi1)
        tracking_penalty = 1.0 * pos_error + 0.5 * psi_error  # 位置误差权重更高
        
        # 3. 控制量平滑性惩罚（避免动作突变）
        if self.model.count > 0:
            u2_prev = self.model.u_arch[self.model.count - 1, 4:8]  # 上一步第二辆车控制量
            u2_current = self.model.u_arch[self.model.count, 4:8]
            control_smooth_penalty = 0.1 * np.sum((u2_current - u2_prev)**2)
        else:
            control_smooth_penalty = 0
        
        # 总奖励 = 负惩罚（最小化目标）
        return - (hinge_force_penalty + 0 * tracking_penalty + 0 * control_smooth_penalty)

    def step(self, action):
        """环境一步交互"""
        # 组合控制量：第一辆车固定控制量 + 第二辆车动作
        # u = np.concatenate([self.u1_fixed, action])
        u = self.u1_fixed.tolist() + action.tolist()
        
        # 执行仿真步
        self.model.step(u)
        
        # 获取观测、奖励
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # 判断终止条件（仿真结束）
        terminated = self.model.is_finish
        truncated = False  # 可扩展：如位置误差超过阈值则截断
        info = {
            'Fh2': (self.model.Fh_arch[self.model.count, 2], 
                    self.model.Fh_arch[self.model.count, 3]),
            'pos_error': np.hypot(
                self.model.getXYi(self.model.x, 1)[0] - self.model.getXYi(self.model.x, 0)[0],
                self.model.getXYi(self.model.x, 1)[1] - self.model.getXYi(self.model.x, 0)[1]
            )
        }
        
        # 渲染处理
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        # 重新初始化模型（复用加载的配置）
        self.model = Model2D2C(self.config)
        self.render_frames = []
        observation = self._get_observation()
        info = {}
        return observation, info

    def render(self):
        """返回渲染帧（用于rgb_array模式）"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """生成单帧渲染图像（基于原模型可视化方法）"""
        if self.render_mode == "human":
            # 实时显示（可调用原模型的plot方法）
            pass
        elif self.render_mode == "rgb_array":
            # 返回RGB数组（示例：黑色背景，可扩展为实际可视化）
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """关闭环境（沿用你之前验证过的可视化逻辑，生成带时间戳的视频）"""
        # 1. 先判断模型是否有generateVideo方法
        if not hasattr(self.model, 'generateVideo'):
            print("提示：模型无generateVideo方法，跳过视频生成")
            return
        
        # 2. 沿用你的路径逻辑：当前文件目录 + output（绝对路径，避免错乱）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "output")
        
        # 3. 提前创建目录（递归创建，确保所有父目录都存在）
        os.makedirs(output_dir, exist_ok=True)
        print(f"确认输出目录存在：{output_dir}")
        
        # 4. 沿用你的文件名逻辑：配置名 + 时间戳 + .mp4（避免文件覆盖）
        time_str = datetime.datetime.now().strftime(r'%y%m%d%H%S')
        file_name = f"{self.config_name}_{time_str}.mp4"
        
        # 5. 尝试生成视频，捕获异常避免程序崩溃
        try:
            self.model.generateVideo(output_dir, file_name)  # 与你之前的调用格式一致
            video_path = os.path.join(output_dir, file_name)
            print(f"视频已成功保存至：{video_path}")
        except Exception as e:
            print(f"错误：生成视频失败，原因：{e}")


# 注册环境（便于通过gym.make()调用）
gym.register(
    id="TwoCarrierEnv-v0",
    entry_point=TwoCarrierEnv,
    max_episode_steps=1000
)

# 测试代码（验证环境是否正常运行）
if __name__ == "__main__":
    # 安装依赖提醒（若未安装pyyaml）
    try:
        import yaml
    except ImportError:
        print("请先安装pyyaml：pip install pyyaml")
        exit(1)

    # 创建环境
    env = gym.make("TwoCarrierEnv-v0")
    obs, info = env.reset(seed=42)
    print("=" * 50)
    print("初始观测形状：", obs.shape)
    print("初始观测值：", obs)
    print("初始辅助信息：", info)
    print("=" * 50)

    # 运行1000步仿真
    for step in range(1000):
        # 随机采样动作（实际训练时替换为RL算法输出）
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 打印步序信息
        print(f"第 {step+1} 步")
        print(f"  动作：{[round(a, 4) for a in action]}")
        print(f"  观测：{[round(o, 4) for o in obs]}")
        print(f"  奖励：{reward:.4f}")
        print(f"  第二辆车铰接力：X={info['Fh2'][0]:.2f}, Y={info['Fh2'][1]:.2f}")
        print(f"  两车位置误差：{info['pos_error']:.2f}")
        print(f"  任务是否终止：{terminated}，是否截断：{truncated}")
        print("-" * 30)

        # 若episode结束，重置环境
        if terminated or truncated:
            print("Episode结束，重置环境...")
            obs, info = env.reset()
            print("=" * 50)

    # 关闭环境，生成仿真视频
    env.close()
    print("仿真结束，视频已保存至 ./sim_results/two_carrier.mp4")