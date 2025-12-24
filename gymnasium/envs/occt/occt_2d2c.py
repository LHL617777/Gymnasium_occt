import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import yaml  
from model import Model2D2C
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image
from io import BytesIO

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
        
        # 可视化相关初始化
        self.render_mode = render_mode
        self.render_frames = []  # 存储rgb_array帧
        self.trajectories = {   # 存储轨迹数据
            'cargo': [],
            'car1': [],
            'car2': []
        }
        self.fig = None
        self.ax = None
        self.animation = None

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
        # 记录轨迹数据
        self._record_trajectories()
        
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
        self.trajectories = {   # 重置轨迹
            'cargo': [],
            'car1': [],
            'car2': []
        }
        self._reset_visualization()  # 重置可视化
        observation = self._get_observation()
        return observation, {}

    def _record_trajectories(self):
        """记录轨迹数据用于可视化"""
        # 超大件位置
        cargo_pos = (self.model.x[0], self.model.x[1])
        self.trajectories['cargo'].append(cargo_pos)
        
        # 车辆位置
        x1, y1 = self.model.getXYi(self.model.x, 0)
        x2, y2 = self.model.getXYi(self.model.x, 1)
        self.trajectories['car1'].append((x1, y1))
        self.trajectories['car2'].append((x2, y2))
    
    def _reset_visualization(self):
        """重置可视化画布"""
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    def _render_frame(self):
        """生成单帧渲染图像（增强版）"""
        # 清除当前轴
        self.ax.clear()
        
        # 获取当前状态数据
        x = self.model.x
        Xo, Yo, Psio = x[0], x[1], x[2]
        Psi1, Psi2 = x[3], x[4]
        x1, y1 = self.model.getXYi(x, 0)
        x2, y2 = self.model.getXYi(x, 1)
        
        # 获取铰接力
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        
        # 设置坐标范围
        range_val = self.config.get('range', 20)
        self.ax.set_xlim(Xo - range_val, Xo + range_val)
        self.ax.set_ylim(Yo - range_val, Yo + range_val)
        self.ax.set_aspect('equal')

        # 绘制轨迹
        if len(self.trajectories['cargo']) > 1:
            cargo_traj = np.array(self.trajectories['cargo'])
            self.ax.plot(cargo_traj[:,0], cargo_traj[:,1], 'k--', alpha=0.5, label='货物轨迹')
            car1_traj = np.array(self.trajectories['car1'])
            self.ax.plot(car1_traj[:,0], car1_traj[:,1], 'b--', alpha=0.5, label='车辆1轨迹')
            car2_traj = np.array(self.trajectories['car2'])
            self.ax.plot(car2_traj[:,0], car2_traj[:,1], 'r--', alpha=0.5, label='车辆2轨迹')

        # 绘制超大件（矩形）
        cargo_bias = self.config.get('oversized_cargo_bias', 1)
        cargo_width = self.config.get('oversized_cargo_width', 3)
        cargo = patches.Rectangle(
            (Xo - cargo_bias, Yo - cargo_width/2),
            2*cargo_bias, cargo_width,
            angle=np.degrees(Psio),
            rotation_point='center',
            facecolor='gray', alpha=0.7, label='超大件'
        )
        self.ax.add_patch(cargo)

        # 绘制车辆1（蓝色箭头）
        car1 = patches.Arrow(
            x1, y1,
            3*np.cos(Psi1), 3*np.sin(Psi1),
            width=2, color='blue', label='车辆1'
        )
        self.ax.add_patch(car1)

        # 绘制车辆2（红色箭头）
        car2 = patches.Arrow(
            x2, y2,
            3*np.cos(Psi2), 3*np.sin(Psi2),
            width=2, color='red', label='车辆2'
        )
        self.ax.add_patch(car2)

        # 绘制铰接力（绿色矢量）
        self.ax.quiver(
            x2, y2, Fh2_x*0.001, Fh2_y*0.001,
            color='green', width=0.005, label='铰接力'
        )

        # 添加文本信息
        self.ax.text(0.05, 0.95, f"步数: {self.model.count}", 
                     transform=self.ax.transAxes, backgroundcolor='w')
        self.ax.text(0.05, 0.90, f"铰接力: ({Fh2_x:.1f}, {Fh2_y:.1f})", 
                     transform=self.ax.transAxes, backgroundcolor='w')
        
        self.ax.legend(loc='upper right')

        # 渲染为图像
        self.fig.canvas.draw()
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frame = np.array(img)

        if self.render_mode == "human":
            plt.pause(0.01)  # 实时显示
        elif self.render_mode == "rgb_array":
            self.render_frames.append(frame)
        
        return frame


    def close(self):
        """关闭环境并生成视频"""
        if self.fig is not None:
            plt.close(self.fig)

        if self.render_mode == "rgb_array" and len(self.render_frames) > 0:
            try:
                # 创建输出目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                # 生成视频文件名
                time_str = datetime.datetime.now().strftime(r'%y%m%d%H%S')
                file_name = f"{self.config_name}_vis_{time_str}.mp4"
                video_path = os.path.join(output_dir, file_name)

                # 使用OpenCV合成视频
                fps = self.metadata['render_fps']
                height, width, _ = self.render_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for frame in self.render_frames:
                    # 转换为BGR格式（OpenCV要求）
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                out.release()
                print(f"可视化视频已保存至: {video_path}")

            except Exception as e:
                print(f"生成视频失败: {e}")
                # 保存单帧图像作为备选
                for i, frame in enumerate(self.render_frames[::10]):  # 每10帧保存一张
                    img_path = os.path.join(output_dir, f"frame_{i}.png")
                    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"已保存关键帧至: {output_dir}")


# 注册环境（便于通过gym.make()调用）
gym.register(
    id="TwoCarrierEnv-v0",
    entry_point=TwoCarrierEnv,
    max_episode_steps=1000
)

# 测试代码（验证环境是否正常运行）
if __name__ == "__main__":
    # 创建环境
    RENDER_MODE = "human"  # 可切换为 "rgb_array" 生成视频
    env = gym.make(
        "TwoCarrierEnv-v0",
        render_mode=RENDER_MODE,
        config_path=None  # 使用默认2d2c.yaml配置
    )
    obs, info = env.reset(seed=42)
    print("=" * 50)
    print("初始观测形状：", obs.shape)
    print(f"初始观测值（关键部分）：Xo={obs[0]:.2f}, Yo={obs[1]:.2f}, 铰接力X={obs[-2]:.2f}, 铰接力Y={obs[-1]:.2f}")
    print(f"初始辅助信息：{info}")
    print("=" * 50)

    # 5. 运行仿真（精简打印，每隔100步打印一次关键信息）
    total_steps = 0
    max_episodes = 1  # 可调整仿真轮数
    for episode in range(max_episodes):
        print(f"\n===== 第 {episode+1}/{max_episodes} 轮仿真开始 =====")
        episode_reward = 0  # 累计每轮奖励
        for step in range(env.spec.max_episode_steps):
            total_steps += 1
            # 随机采样动作（实际训练时替换为RL算法输出；若要稳定运动，可改用固定动作）
            # 固定动作示例（第二辆车直驶，推力适中）：action = np.array([0, 0, 500, 500])
            action = env.action_space.sample()
            
            # 环境交互
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # 精简打印：每隔100步打印一次，避免冗余
            if (step + 1) % 100 == 0:
                print(f"  第 {step+1} 步 | 累计奖励：{episode_reward:.2f}")
                print(f"  铰接力（X,Y）：({info['Fh2'][0]:.2f}, {info['Fh2'][1]:.2f}) | 位置误差：{info['pos_error']:.2f}")
                print(f"  任务状态：终止={terminated} | 截断={truncated}")
                print("  " + "-" * 20)

            # 若episode结束，重置环境
            if terminated or truncated:
                finish_reason = "仿真步数耗尽（truncated）" if truncated else "满足终止条件（terminated）"
                print(f"  第 {step+1} 步：{finish_reason}，本轮结束")
                print(f"  本轮累计奖励：{episode_reward:.2f}")
                obs, info = env.reset()
                break

    # 6. 关闭环境，生成视频（仅rgb_array模式生效）
    env.close()
    
    # 7. 规范输出提示
    if RENDER_MODE == "rgb_array":
        print("\n仿真结束！视频已保存至当前目录的【output】文件夹")
    elif RENDER_MODE == "human":
        print("\n仿真结束！实时可视化窗口已关闭")
    else:
        print("\n仿真结束！未启用可视化功能")
    print("=" * 50)