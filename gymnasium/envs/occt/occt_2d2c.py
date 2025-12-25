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

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

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
            'car2': [],
            'hinge1': [],
            'hinge2': []
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
            'car2': [],
            'hinge1': [],
            'hinge2': []
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

        # 新增：铰接点轨迹（清晰展示车辆与超大件的连接关系）
        Xo, Yo, Psio = self.model.x[0], self.model.x[1], self.model.x[2]
        # 铰接点1（车辆1与超大件连接点）
        hinge1_x = Xo + self.config['x__o_1'] * np.cos(Psio) - self.config['y__o_1'] * np.sin(Psio)
        hinge1_y = Yo + self.config['x__o_1'] * np.sin(Psio) + self.config['y__o_1'] * np.cos(Psio)
        # 铰接点2（车辆2与超大件连接点）
        hinge2_x = Xo + self.config['x__o_2'] * np.cos(Psio) - self.config['y__o_2'] * np.sin(Psio)
        hinge2_y = Yo + self.config['x__o_2'] * np.sin(Psio) + self.config['y__o_2'] * np.cos(Psio)
        self.trajectories['hinge1'].append((hinge1_x, hinge1_y))
        self.trajectories['hinge2'].append((hinge2_x, hinge2_y))

    def _reset_visualization(self):
        """重置可视化画布（优化画布布局，避免元素遮挡）"""
        if self.fig is not None:
            plt.close(self.fig)
        # 调整画布大小，预留信息显示区域
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        # 设置画布背景色，提升对比度
        self.ax.set_facecolor('#f8f8f8')
    
    def _render_frame(self):
        """生成单帧渲染图像（核心优化：超大件+运载车辆可视化增强）"""
        # 清除当前轴
        self.ax.clear()
        
        # 获取当前状态数据（精准提取，确保可视化与仿真状态一致）
        x = self.model.x
        Xo, Yo, Psio = x[0], x[1], x[2]
        Psi1, Psi2 = x[3], x[4]
        # 车辆质心位置
        car1_cx, car1_cy = self.model.getXYi(x, 0)
        car2_cx, car2_cy = self.model.getXYi(x, 1)
        # 铰接力
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        # 铰接点位置（超大件与车辆的连接点，基于配置计算）
        hinge1_x = Xo + self.config['x__o_1'] * np.cos(Psio) - self.config['y__o_1'] * np.sin(Psio)
        hinge1_y = Yo + self.config['x__o_1'] * np.sin(Psio) + self.config['y__o_1'] * np.cos(Psio)
        hinge2_x = Xo + self.config['x__o_2'] * np.cos(Psio) - self.config['y__o_2'] * np.sin(Psio)
        hinge2_y = Yo + self.config['x__o_2'] * np.sin(Psio) + self.config['y__o_2'] * np.cos(Psio)

        # 1. 优化坐标范围：自适应所有元素，避免车辆/超大件移出视野
        all_x = [Xo, car1_cx, car2_cx, hinge1_x, hinge2_x]
        all_y = [Yo, car1_cy, car2_cy, hinge1_y, hinge2_y]
        x_min, x_max = min(all_x) - 10, max(all_x) + 10
        y_min, y_max = min(all_y) - 10, max(all_y) + 10
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X 坐标 (m)', fontsize=10)
        self.ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        self.ax.set_title('双车运载超大件仿真可视化', fontsize=12, fontweight='bold')

        # 2. 优化超大件可视化：更逼真，标记铰接点和质心
        cargo_bias = self.config.get('oversized_cargo_bias', 3)
        cargo_width = self.config.get('oversized_cargo_width', 2.5)
        # 超大件主体（灰色填充，黑色边框，提升辨识度）
        cargo = patches.Rectangle(
            (Xo - cargo_bias, Yo - cargo_width/2),
            2*cargo_bias, cargo_width,
            angle=np.degrees(Psio),
            rotation_point='center',  # 确保绕质心旋转，与仿真一致
            facecolor='#7f8c8d',      # 深灰色，更贴近真实货物
            edgecolor='black',        # 黑色边框，区分轮廓
            linewidth=1.5,
            alpha=0.8,
            label='超大件（质心）'
        )
        self.ax.add_patch(cargo)
        # 标记超大件质心（红色圆点，清晰定位）
        self.ax.scatter(Xo, Yo, color='red', s=50, zorder=5, label='超大件质心')
        # 标记铰接点（蓝色/橙色方块，区分两车连接点）
        self.ax.scatter(hinge1_x, hinge1_y, color='blue', s=60, marker='s', zorder=5, label='铰接点1（车1）')
        self.ax.scatter(hinge2_x, hinge2_y, color='orange', s=60, marker='s', zorder=5, label='铰接点2（车2）')
        # 绘制超大件与铰接点的连接线（展示连接关系）
        self.ax.plot([Xo, hinge1_x], [Yo, hinge1_y], color='black', linewidth=1, linestyle=':')
        self.ax.plot([Xo, hinge2_x], [Yo, hinge2_y], color='black', linewidth=1, linestyle=':')

        # 3. 优化运载车辆可视化：替换箭头为真实车辆形状，区分车头车尾
        car_length = 3  # 车辆长度
        car_width = 1.5 # 车辆宽度
        # 车辆1（蓝色，带车头标记，绕质心旋转）
        # 车辆矩形（基于质心和姿态角定位，确保与仿真姿态一致）
        car1 = patches.Rectangle(
            (car1_cx - car_length/2, car1_cy - car_width/2),
            car_length, car_width,
            angle=np.degrees(Psi1),
            rotation_point='center',
            facecolor='#3498db',      # 亮蓝色，区分车辆1
            edgecolor='black',
            linewidth=1,
            alpha=0.8,
            label='运载车辆1'
        )
        self.ax.add_patch(car1)
        # 标记车辆1车头（三角形，指示行驶方向）
        car1_nose_x = car1_cx + (car_length/2) * np.cos(Psi1)
        car1_nose_y = car1_cy + (car_length/2) * np.sin(Psi1)
        self.ax.scatter(car1_nose_x, car1_nose_y, color='darkblue', s=30, marker='^', zorder=6)

        # 车辆2（红色，带车头标记，绕质心旋转）
        car2 = patches.Rectangle(
            (car2_cx - car_length/2, car2_cy - car_width/2),
            car_length, car_width,
            angle=np.degrees(Psi2),
            rotation_point='center',
            facecolor='#e74c3c',      # 亮红色，区分车辆2
            edgecolor='black',
            linewidth=1,
            alpha=0.8,
            label='运载车辆2'
        )
        self.ax.add_patch(car2)
        # 标记车辆2车头（三角形，指示行驶方向）
        car2_nose_x = car2_cx + (car_length/2) * np.cos(Psi2)
        car2_nose_y = car2_cy + (car_length/2) * np.sin(Psi2)
        self.ax.scatter(car2_nose_x, car2_nose_y, color='darkred', s=30, marker='^', zorder=6)

        # 4. 优化轨迹绘制：分层显示，区分不同元素轨迹
        if len(self.trajectories['cargo']) > 1:
            # 超大件轨迹（黑色虚线，透明度更低，避免遮挡）
            cargo_traj = np.array(self.trajectories['cargo'])
            self.ax.plot(cargo_traj[:,0], cargo_traj[:,1], 'k--', alpha=0.3, linewidth=1, label='超大件轨迹')
            # 车辆1轨迹（蓝色虚线）
            car1_traj = np.array(self.trajectories['car1'])
            self.ax.plot(car1_traj[:,0], car1_traj[:,1], '#3498db', linestyle='--', alpha=0.4, linewidth=1, label='车辆1轨迹')
            # 车辆2轨迹（红色虚线）
            car2_traj = np.array(self.trajectories['car2'])
            self.ax.plot(car2_traj[:,0], car2_traj[:,1], '#e74c3c', linestyle='--', alpha=0.4, linewidth=1, label='车辆2轨迹')
            # 铰接点轨迹（浅灰虚线，补充信息）
            hinge1_traj = np.array(self.trajectories['hinge1'])
            hinge2_traj = np.array(self.trajectories['hinge2'])
            # 铰接点1轨迹：线型为':'，颜色通过color参数指定（蓝色）
            self.ax.plot(
                hinge1_traj[:,0], hinge1_traj[:,1], 
                ':',  # 仅指定线型，颜色单独通过color参数传递
                color='blue', 
                alpha=0.2, 
                linewidth=0.8
            )
            # 铰接点2轨迹：线型为':'，颜色通过color参数指定（橙色）
            self.ax.plot(
                hinge2_traj[:,0], hinge2_traj[:,1], 
                ':',  # 仅指定线型，避免与完整颜色名称冲突
                color='orange', 
                alpha=0.2, 
                linewidth=0.8
            )
        # 5. 铰接力可视化（优化矢量箭头，避免遮挡）
        self.ax.quiver(
            hinge2_x, hinge2_y,  # 从铰接点2出发，更贴合实际受力位置
            Fh2_x*0.001, Fh2_y*0.001,
            color='green', 
            width=0.005, 
            headwidth=5, 
            headlength=8,
            alpha=0.7,
            label='铰接力（车辆2→超大件）'
        )

        # 6. 优化文本信息：避免遮挡，排版更整洁
        text_props = {'transform': self.ax.transAxes, 'backgroundcolor': 'white', 'fontsize': 9}
        self.ax.text(0.02, 0.98, f"步数: {self.model.count}", va='top', **text_props)
        self.ax.text(0.02, 0.94, f"超大件质心：({Xo:.1f}, {Yo:.1f})", va='top', **text_props)
        self.ax.text(0.02, 0.90, f"铰接力：({Fh2_x:.1f}, {Fh2_y:.1f})", va='top', **text_props)
        
        # 优化图例：避免重叠，自动调整位置
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        # 添加网格，便于读取坐标
        self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # 渲染为图像（原有逻辑不变，确保兼容性）
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