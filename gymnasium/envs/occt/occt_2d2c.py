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
        self.is_sim_finished = False  # 新增：标识仿真是否已结束
        print(f"初始化渲染模式：{self.render_mode}，是否为rgb_array：{self.render_mode == 'rgb_array'}")

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

        # 新增日志：确认每步是否触发帧渲染
        # print(f"第 {self.model.count} 步：准备调用_render_frame()，当前渲染模式：{self.render_mode}")
        
        # 渲染处理
        self._render_frame()  # 移除原有的if判断，统一触发帧渲染
        if self.render_mode == "human":
            plt.pause(0.001)  # 仅human模式需要窗口暂停刷新
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=True):
        """重置环境"""
        super().reset(seed=seed)
        # 重新初始化模型（复用加载的配置）
        self.model = Model2D2C(self.config)
        # 仅当clear_frames为True且仿真未结束时，才清空帧列表
        if clear_frames and not self.is_sim_finished:
            self.render_frames = []
        self.trajectories = {   # 重置轨迹
            'cargo': [],
            'car1': [],
            'car2': [],
            'hinge1': [],
            'hinge2': []
        }
        # 强制初始化可视化，无论何种模式
        self._reset_visualization()  # 确保fig和ax被创建
        observation = self._get_observation()
        # 新增日志：确认可视化初始化
        # print(f"环境重置完成，可视化fig是否为空：{self.fig is None}")
        return observation, {}

    # 新增：标记仿真结束的方法
    def mark_sim_finished(self):
        self.is_sim_finished = True
        print("仿真已标记为结束，后续reset()不会清空帧列表")

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
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax.set_facecolor('#f8f8f8')
        
        # 初始化所有图形对象（只创建一次，后续仅更新属性）
        # 1. 超大件
        self.cargo_patch = patches.Rectangle(
            (0, 0), 2*self.config.get('oversized_cargo_bias', 2), 
            self.config.get('oversized_cargo_width', 3),
            facecolor='#7f8c8d', edgecolor='black', linewidth=1.5, alpha=0.8
        )
        self.ax.add_patch(self.cargo_patch)
        # 超大件质心
        self.cargo_centroid = self.ax.scatter([], [], color='red', s=50, zorder=5)
        # 铰接点
        self.hinge1_scatter = self.ax.scatter([], [], color='blue', s=60, marker='s', zorder=5)
        self.hinge2_scatter = self.ax.scatter([], [], color='orange', s=60, marker='s', zorder=5)
        # 铰接点连接线（初始为空）
        self.hinge1_line, = self.ax.plot([], [], color='black', linewidth=1, linestyle=':')
        self.hinge2_line, = self.ax.plot([], [], color='black', linewidth=1, linestyle=':')
        
        # 2. 车辆
        car_length = 3
        car_width = 1.5
        self.car1_patch = patches.Rectangle(
            (0, 0), car_length, car_width,
            facecolor='#3498db', edgecolor='black', linewidth=1, alpha=0.8
        )
        self.car2_patch = patches.Rectangle(
            (0, 0), car_length, car_width,
            facecolor='#e74c3c', edgecolor='black', linewidth=1, alpha=0.8
        )
        self.ax.add_patch(self.car1_patch)
        self.ax.add_patch(self.car2_patch)
        # 车辆车头
        self.car1_nose = self.ax.scatter([], [], color='darkblue', s=30, marker='>', zorder=6)
        self.car2_nose = self.ax.scatter([], [], color='darkred', s=30, marker='>', zorder=6)
        
        # 3. 轨迹（初始为空，后续追加新点）
        self.cargo_traj_line, = self.ax.plot([], [], 'k--', alpha=0.3, linewidth=1)
        self.car1_traj_line, = self.ax.plot([], [], '#3498db', linestyle='--', alpha=0.4, linewidth=1)
        self.car2_traj_line, = self.ax.plot([], [], '#e74c3c', linestyle='--', alpha=0.4, linewidth=1)
        self.hinge1_traj_line, = self.ax.plot([], [], ':', color='blue', alpha=0.2, linewidth=0.8)
        self.hinge2_traj_line, = self.ax.plot([], [], ':', color='orange', alpha=0.2, linewidth=0.8)
        
        # 4. 铰接力矢量（复用对象，更新方向和大小）
        self.force_quiver = self.ax.quiver([], [], [], [], color='green', width=0.005, headwidth=5, headlength=8, alpha=0.7)
        
    def _render_frame(self):
        """生成单帧渲染图像（核心优化：超大件+运载车辆可视化增强）"""
        # 先判断可视化是否初始化，未初始化则先初始化
        if self.fig is None or self.ax is None:
            self._reset_visualization()
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

        # 1. 更新超大件
        cargo_bias = self.config.get('oversized_cargo_bias', 2)
        cargo_width = self.config.get('oversized_cargo_width', 3)
        # 更新位置和旋转角度
        self.cargo_patch.set_xy((Xo - cargo_bias, Yo - cargo_width/2))
        self.cargo_patch.set_angle(np.degrees(Psio))
        # 更新质心和铰接点
        self.cargo_centroid.set_offsets((Xo, Yo))
        self.hinge1_scatter.set_offsets((hinge1_x, hinge1_y))
        self.hinge2_scatter.set_offsets((hinge2_x, hinge2_y))
        # 更新铰接点连接线
        self.hinge1_line.set_data([Xo, hinge1_x], [Yo, hinge1_y])
        self.hinge2_line.set_data([Xo, hinge2_x], [Yo, hinge2_y])

        # 2. 更新车辆
        car_length = 3
        car_width = 1.5
        # 更新车辆1位置和角度
        self.car1_patch.set_xy((car1_cx - car_length/2, car1_cy - car_width/2))
        self.car1_patch.set_angle(np.degrees(Psi1))
        # 更新车辆1车头
        car1_nose_x = car1_cx + (car_length/2) * np.cos(Psi1)
        car1_nose_y = car1_cy + (car_length/2) * np.sin(Psi1)
        self.car1_nose.set_offsets((car1_nose_x, car1_nose_y))
        # 更新车辆2
        self.car2_patch.set_xy((car2_cx - car_length/2, car2_cy - car_width/2))
        self.car2_patch.set_angle(np.degrees(Psi2))
        car2_nose_x = car2_cx + (car_length/2) * np.cos(Psi2)
        car2_nose_y = car2_cy + (car_length/2) * np.sin(Psi2)
        self.car2_nose.set_offsets((car2_nose_x, car2_nose_y))

        # 3. 更新轨迹（仅追加新点，不重新绘制全部）
        if len(self.trajectories['cargo']) > 1:
            cargo_traj = np.array(self.trajectories['cargo'])
            self.cargo_traj_line.set_data(cargo_traj[:,0], cargo_traj[:,1])
            car1_traj = np.array(self.trajectories['car1'])
            self.car1_traj_line.set_data(car1_traj[:,0], car1_traj[:,1])
            car2_traj = np.array(self.trajectories['car2'])
            self.car2_traj_line.set_data(car2_traj[:,0], car2_traj[:,1])
            hinge1_traj = np.array(self.trajectories['hinge1'])
            self.hinge1_traj_line.set_data(hinge1_traj[:,0], hinge1_traj[:,1])
            hinge2_traj = np.array(self.trajectories['hinge2'])
            self.hinge2_traj_line.set_data(hinge2_traj[:,0], hinge2_traj[:,1])

        # 4. 更新铰接力
        self.force_quiver.set_offsets((hinge2_x, hinge2_y))
        self.force_quiver.set_UVC(Fh2_x*0.001, Fh2_y*0.001)

        # 5. 更新坐标范围、文本、图例（按需更新，无需每次重新设置）
        self.ax.set_xlim(min([Xo, car1_cx, car2_cx])-10, max([Xo, car1_cx, car2_cx])+10)
        self.ax.set_ylim(min([Yo, car1_cy, car2_cy])-10, max([Yo, car1_cy, car2_cy])+10)
        self.ax.set_title("两车运载超大件系统仿真可视化", fontsize=16)

        # 仅刷新画布，不执行冗余的图像转换
        self.fig.canvas.draw_idle()  # 高效刷新，比canvas.draw()更快

        # 仅在rgb_array模式下执行图像转换
        if self.render_mode == "rgb_array":
            # 新增日志：确认进入帧保存分支
            # print(f"进入rgb_array帧保存逻辑，当前帧索引：{len(self.render_frames)}")
            try:
                buf = BytesIO()
                self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=self.fig.get_facecolor())
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                frame = np.array(img)
                # 校验帧尺寸（确保所有帧尺寸一致）
                if len(self.render_frames) > 0:
                    ref_shape = self.render_frames[0].shape
                    if frame.shape != ref_shape:
                        frame = cv2.resize(frame, (ref_shape[1], ref_shape[0]))
                        # print(f"帧尺寸不一致，已调整为：{ref_shape}")
                self.render_frames.append(frame)
                # 释放缓存，避免内存泄漏
                buf.close()
                # print(f"第 {len(self.render_frames)} 帧保存成功")
                return frame
            except Exception as e:
                # 新增：捕获帧保存异常
                print(f"帧保存失败，错误：{type(e).__name__}: {e}")


    def close(self):
        """关闭环境并生成视频"""
        # 新增：打印关键状态，确认帧列表和渲染模式
        print(f"===== 进入close()方法 ======")
        print(f"当前render_mode：{self.render_mode}")
        print(f"当前render_frames列表长度：{len(self.render_frames)}")
        print(f"render_frames是否为列表：{isinstance(self.render_frames, list)}")
        
        if self.fig is not None:
            plt.close(self.fig)

        if self.render_mode == "rgb_array" and len(self.render_frames) > 0:
            try:
                # 创建输出目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                print(f"输出目录已准备：{output_dir}，共待写入帧数量：{len(self.render_frames)}")

                # 生成视频文件名
                time_str = datetime.datetime.now().strftime(r'%y%m%d%H%M%S')  # 修正时间格式，避免重复
                file_name = f"{self.config_name}_vis_{time_str}.mp4"
                video_path = os.path.join(output_dir, file_name)

                # 使用OpenCV合成视频（兼容多系统编码）
                fps = self.metadata['render_fps']
                height, width, _ = self.render_frames[0].shape
                # 兼容Windows/Mac/Linux的编码格式
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4格式（优先）
                # 备选编码：若mp4v失败，切换为XVID（avi格式）
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_path = video_path.replace(".mp4", ".avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    print(f"mp4格式不支持，切换为avi格式，保存路径：{video_path}")

                # 写入帧
                for idx, frame in enumerate(self.render_frames):
                    # 转换为BGR格式（OpenCV要求）
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                    if idx % 100 == 0:
                        print(f"已写入 {idx+1}/{len(self.render_frames)} 帧")

                out.release()
                print(f"可视化视频已成功保存至: {video_path}")

            except Exception as e:
                # 暴露具体异常信息，便于排查
                print(f"生成视频失败，详细错误信息：{type(e).__name__}: {e}")
                # 保存单帧图像作为备选
                try:
                    for i, frame in enumerate(self.render_frames[::10]):  # 每10帧保存一张
                        img_path = os.path.join(output_dir, f"frame_{i:03d}.png")
                        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"已保存关键帧至: {output_dir}，共保存 {len(self.render_frames[::10])} 张")
                except Exception as e2:
                    print(f"保存关键帧也失败，错误：{type(e2).__name__}: {e2}")
        elif self.render_mode == "rgb_array" and len(self.render_frames) == 0:
            print("警告：未生成任何视频帧，无法创建视频！请检查帧渲染逻辑是否正常执行")
        else:
            print("非rgb_array模式，不生成视频")


# 注册环境（便于通过gym.make()调用）
gym.register(
    id="TwoCarrierEnv-v0",
    entry_point=TwoCarrierEnv,
    max_episode_steps=1000
)

# 测试代码（验证环境是否正常运行）
if __name__ == "__main__":
    # 创建环境
    RENDER_MODE = "rgb_array"
    env = gym.make(
        "TwoCarrierEnv-v0",
        render_mode=RENDER_MODE,
        config_path=None  # 使用默认2d2c.yaml配置
    )
    # 新增：获取原始自定义环境实例（解除TimeLimit包装）
    raw_env = env.unwrapped
    # 新增：确认环境实例的渲染模式
    print(f"环境实例渲染模式：{raw_env.render_mode}，是否与预期一致：{raw_env.render_mode == RENDER_MODE}")
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

            # 若episode结束，不调用reset()（避免清空帧列表），直接跳出循环
            if terminated or truncated:
                finish_reason = "仿真步数耗尽（truncated）" if truncated else "满足终止条件（terminated）"
                print(f"  第 {step+1} 步：{finish_reason}，本轮结束")
                print(f"  本轮累计奖励：{episode_reward:.2f}")
                raw_env.mark_sim_finished()
                break  # 直接跳出，不调用env.reset()

    # 6. 关闭环境，生成视频（仅rgb_array模式生效）
    env.close()
    
    # 7. 规范输出提示（修复虚假提示）
    if RENDER_MODE == "rgb_array":
        # 检查output目录是否有视频文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "output")
        video_files = []
        if os.path.exists(output_dir):
            video_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4', '.avi'))]
        if video_files:
            print(f"\n仿真结束！视频已保存至当前目录的【output】文件夹，文件名：{video_files[-1]}")
        else:
            print("\n仿真结束！未生成视频，请查看控制台错误提示排查问题")
    elif RENDER_MODE == "human":
        print("\n仿真结束！实时可视化窗口已关闭")
    else:
        print("\n仿真结束！未启用可视化功能")
    print("=" * 50)