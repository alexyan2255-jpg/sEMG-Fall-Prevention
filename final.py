import serial
import time
import numpy as np
import onnxruntime as ort
from collections import deque
import threading
from datetime import datetime
import os
import can
from enum import Enum

# 解决OpenMP重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("使用ONNX Runtime进行CPU推理")

class ActuatorState(Enum):
    """执行器状态枚举"""
    IDLE = 0
    LEFT_SWING = 2
    RIGHT_SWING = 1

class RaspberryPiActuatorController:
    """树莓派执行器控制器 - 使用SocketCAN和双舵机串口"""
    
    def __init__(self, can_interface='can0', servo1_port='/dev/ttyUSB0', servo2_port='/dev/ttyUSB1', servo_baudrate=1000000):
        """
        初始化树莓派执行器控制器
        Args:
            can_interface: CAN接口名称 (通常是 can0)
            servo1_port: 舵机1串口 (通常是 /dev/ttyUSB0)
            servo2_port: 舵机2串口 (通常是 /dev/ttyUSB1)
            servo_baudrate: 舵机波特率 (1000000 = 1000K)
        """
        # SocketCAN初始化
        try:
            self.can_bus = can.interface.Bus(
                channel=can_interface,
                bustype='socketcan'
            )
            print(f"SocketCAN接口 {can_interface} 连接成功")
        except Exception as e:
            print(f"SocketCAN连接失败: {e}")
            print("请确保已执行: sudo ip link set can0 up type can bitrate 1000000")
            raise
        
        # 舵机1串口初始化
        try:
            self.servo1_serial = serial.Serial(servo1_port, servo_baudrate, timeout=1)
            print(f"舵机1串口 {servo1_port} 连接成功 (波特率: {servo_baudrate})")
        except Exception as e:
            print(f"舵机1串口连接失败: {e}")
            print("请检查串口设备路径和权限")
            raise
        
        # 舵机2串口初始化
        try:
            self.servo2_serial = serial.Serial(servo2_port, servo_baudrate, timeout=1)
            print(f"舵机2串口 {servo2_port} 连接成功 (波特率: {servo_baudrate})")
        except Exception as e:
            print(f"舵机2串口连接失败: {e}")
            print("请检查串口设备路径和权限")
            raise
        
        self.current_state = ActuatorState.IDLE
        self.action_in_progress = False
        
        # 预定义舵机命令
        self.servo_commands = {
            "ID1_servo_lock": bytes.fromhex("FF FF 01 0B 03 28 01 D0 00 00 B8 0B D0 07 5D"),
            "ID1_servo_unlock": bytes.fromhex("FF FF 01 0B 03 28 01 D0 FC 03 B8 0B D0 07 5E"),
            "ID2_servo_lock": bytes.fromhex("FF FF 02 0B 03 28 01 D0 F8 07 B8 0B D0 07 5D"),
            "ID2_servo_unlock": bytes.fromhex("FF FF 02 0B 03 28 01 C8 E8 03 F4 01 96 00 88")
        }
        
        # 预定义电机命令
        self.motor_commands = {
            "ID1_motor_run": [0xA2, 0x00, 0x00, 0x00, 0xE0, 0x40, 0xFD, 0xFF],  # -300 rpm
            "ID2_motor_run": [0xA2, 0x00, 0x00, 0x00, 0x20, 0xBF, 0x02, 0x00],  # -300 rpm
            "motor_stop": [0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]     # 停止
        }
        
        print("树莓派执行器控制器初始化完成")
    
    def send_servo_command(self, command_key):
        """发送舵机命令到指定舵机"""
        try:
            command = self.servo_commands[command_key]
            
            # 根据命令类型确定使用哪个舵机串口
            if "ID1" in command_key:
                serial_port = self.servo1_serial
                port_name = "舵机1"
            else:  # ID2
                serial_port = self.servo2_serial
                port_name = "舵机2"
            
            bytes_written = serial_port.write(command)
            serial_port.flush()
            print(f"{port_name}命令发送成功: {command_key} ({bytes_written} bytes)")
            return True
        except Exception as e:
            print(f"舵机命令发送失败 {command_key}: {e}")
            return False
    
    def send_motor_command(self, motor_id, command_key):
        """发送电机CAN命令"""
        try:
            data = self.motor_commands[command_key]
            can_id = 0x141 if motor_id == 1 else 0x142
            
            msg = can.Message(
                arbitration_id=can_id,
                is_extended_id=False,
                data=data
            )
            
            self.can_bus.send(msg)
            print(f"电机{motor_id}命令发送成功: {command_key} (ID: 0x{can_id:X})")
            return True
        except Exception as e:
            print(f"电机{motor_id}命令发送失败 {command_key}: {e}")
            return False
    
    def execute_left_swing(self):
        """执行左摆动作 - 非阻塞"""
        if self.action_in_progress:
            print("动作正在执行中，忽略左摆指令")
            return
        
        def _execute():
            try:
                self.action_in_progress = True
                self.current_state = ActuatorState.LEFT_SWING
                print("🔥 开始执行左摆动作...")
                
                # 同时发送舵机锁定和电机运行命令
                servo_success = self.send_servo_command("ID2_servo_lock")
                motor_success = self.send_motor_command(1, "ID1_motor_run")
                
                # 持续1.5秒
                time.sleep(1.5)
                
                # 发送舵机解锁和电机停止命令
                self.send_servo_command("ID2_servo_unlock")
                self.send_motor_command(1, "motor_stop")
                
                print("✅ 左摆动作完成")
                
            except Exception as e:
                print(f"左摆动作执行异常: {e}")
            finally:
                self.current_state = ActuatorState.IDLE
                self.action_in_progress = False
        
        thread = threading.Thread(target=_execute, name="LeftSwingThread")
        thread.daemon = True
        thread.start()
    
    def execute_right_swing(self):
        """执行右摆动作 - 非阻塞"""
        if self.action_in_progress:
            print("动作正在执行中，忽略右摆指令")
            return
        
        def _execute():
            try:
                self.action_in_progress = True
                self.current_state = ActuatorState.RIGHT_SWING
                print("🔥 开始执行右摆动作...")
                
                # 同时发送舵机锁定和电机运行命令
                servo_success = self.send_servo_command("ID1_servo_lock")
                motor_success = self.send_motor_command(2, "ID2_motor_run")
                
                # 持续1.5秒
                time.sleep(1.5)
                
                # 发送舵机解锁和电机停止命令
                self.send_servo_command("ID1_servo_unlock")
                self.send_motor_command(2, "motor_stop")
                
                print("✅ 右摆动作完成")
                
            except Exception as e:
                print(f"右摆动作执行异常: {e}")
            finally:
                self.current_state = ActuatorState.IDLE
                self.action_in_progress = False
        
        thread = threading.Thread(target=_execute, name="RightSwingThread")
        thread.daemon = True
        thread.start()
    
    def cleanup(self):
        """清理资源"""
        print("正在清理执行器控制器资源...")
        
        # 停止所有电机
        try:
            self.send_motor_command(1, "motor_stop")
            self.send_motor_command(2, "motor_stop")
        except:
            pass
        
        # 关闭串口
        if hasattr(self, 'servo1_serial') and self.servo1_serial.is_open:
            self.servo1_serial.close()
        if hasattr(self, 'servo2_serial') and self.servo2_serial.is_open:
            self.servo2_serial.close()
        
        # 关闭CAN总线
        if hasattr(self, 'can_bus'):
            self.can_bus.shutdown()
        
        print("执行器控制器资源清理完成")

# 卡尔曼滤波器
class KalmanFilter:
    def __init__(self, num_classes=3, process_noise=1e-3, measurement_noise=1e-1):
        self.num_classes = num_classes
        self.x = np.ones(num_classes) / num_classes
        self.P = np.eye(num_classes) * 0.1
        self.Q = np.eye(num_classes) * process_noise
        self.R = np.eye(num_classes) * measurement_noise
        self.F = np.eye(num_classes) * 0.9 + np.ones((num_classes, num_classes)) * 0.1 / num_classes
        self.H = np.eye(num_classes)
        self.is_initialized = False

    def update(self, measurement):
        measurement = np.array(measurement)

        if not self.is_initialized:
            self.x = measurement.copy()
            self.is_initialized = True
            return self.x

        # 预测步骤
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新步骤
        y = measurement - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(self.num_classes) - K @ self.H) @ P_pred

        # 确保概率和为1且非负
        self.x = np.maximum(self.x, 0)
        self.x = self.x / np.sum(self.x)

        return self.x.copy()

# 双腿EMG实时预测控制系统（整合执行器控制）
class DualLegEMGActuatorSystem:
    def __init__(self, left_model_path, right_model_path, port='/dev/ttyACM0', baudrate=115200,
                 can_interface='can0', servo1_port='/dev/ttyUSB0', servo2_port='/dev/ttyUSB1',
                 left_threshold=0.34, right_threshold=0.45, seq_len=200, buffer_size=500):
        """
        初始化双腿EMG预测控制系统
        """
        self.seq_len = seq_len
        self.port = port
        self.baudrate = baudrate
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold

        # 初始化EMG数据串口
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"EMG串口 {port} 连接成功")
        except Exception as e:
            print(f"EMG串口连接失败: {e}")
            raise

        # 初始化执行器控制器
        try:
            self.actuator = RaspberryPiActuatorController(
                can_interface=can_interface,
                servo1_port=servo1_port, 
                servo2_port=servo2_port
            )
        except Exception as e:
            print(f"执行器控制器初始化失败: {e}")
            print("如果不使用执行器，可以注释掉这部分代码")
            self.actuator = None

        # 加载ONNX模型
        self.left_session = self.load_onnx_model(left_model_path, "左腿模型")
        self.right_session = self.load_onnx_model(right_model_path, "右腿模型")

        # 获取模型输入输出信息
        self.left_input_name = self.left_session.get_inputs()[0].name
        self.left_output_names = [output.name for output in self.left_session.get_outputs()]
        self.right_input_name = self.right_session.get_inputs()[0].name
        self.right_output_names = [output.name for output in self.right_session.get_outputs()]

        # 初始化卡尔曼滤波器
        self.left_kalman = KalmanFilter()
        self.right_kalman = KalmanFilter()

        # 初始化数据缓冲区
        self.left_buffer = deque(maxlen=buffer_size)
        self.right_buffer = deque(maxlen=buffer_size)

        # 类别映射
        self.class_names = {0: "正常", 1: "左摆", 2: "右摆"}

        # 帧头和分隔符
        self.frame_header = b'\x0d\x0a'
        self.delimiter = b'\x2c'

        # 统计信息
        self.packet_count = 0
        self.prediction_count = 0
        self.control_count = 0
        self.start_time = time.time()

        print("双腿EMG预测控制系统初始化完成")
        print(f"控制阈值: 左摆={left_threshold}, 右摆={right_threshold}")

    def load_onnx_model(self, model_path, model_name):
        """加载ONNX模型"""
        try:
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)

            input_info = session.get_inputs()[0]
            print(f"{model_name}加载成功")
            print(f"  输入形状: {input_info.shape}")
            print(f"  输入名称: {input_info.name}")
            print(f"  输出数量: {len(session.get_outputs())}")

            return session
        except Exception as e:
            print(f"{model_name}加载失败: {e}")
            raise

    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def read_emg_packet(self):
        """读取一个完整的肌电数据包"""
        buffer = b''

        # 寻找帧头
        while True:
            byte = self.ser.read(1)
            if not byte:
                return None
            buffer += byte
            if buffer.endswith(self.frame_header):
                break

        # 读取数据
        data_buffer = b''
        delimiter_count = 0

        while delimiter_count < 12:  # 期望12个数据
            byte = self.ser.read(1)
            if not byte:
                break

            if byte == self.delimiter:
                delimiter_count += 1
                if delimiter_count < 12:
                    data_buffer += byte
            elif byte == b'\x0d':
                next_byte = self.ser.read(1)
                if next_byte == b'\x0a':
                    break
                else:
                    data_buffer += byte + next_byte
            else:
                data_buffer += byte

        return self.parse_packet(data_buffer)

    def parse_packet(self, data_buffer):
        """解析数据包"""
        try:
            data_str = data_buffer.decode('ascii')
            values = data_str.split(',')

            if len(values) >= 12:
                emg_data = [float(val) for val in values[:12]]
                return emg_data
            else:
                return None
        except Exception:
            return None

    def predict_and_control(self):
        """预测并控制执行器"""
        if len(self.left_buffer) < self.seq_len or len(self.right_buffer) < self.seq_len:
            return None, None, None

        # 准备输入数据
        left_input_data = np.array(list(self.left_buffer)[-self.seq_len:])  # (200, 4)
        right_input_data = np.array(list(self.right_buffer)[-self.seq_len:])  # (200, 4)

        if left_input_data.shape[1] != 4 or right_input_data.shape[1] != 4:
            print(f"警告: 输入通道数不匹配，左腿: {left_input_data.shape[1]}, 右腿: {right_input_data.shape[1]}")
            return None, None, None

        # 转换为ONNX模型期望的输入格式 (1, 200, 4)
        left_input_tensor = left_input_data.astype(np.float32).reshape(1, self.seq_len, 4)
        right_input_tensor = right_input_data.astype(np.float32).reshape(1, self.seq_len, 4)

        try:
            # ONNX推理
            left_outputs = self.left_session.run(self.left_output_names, {self.left_input_name: left_input_tensor})
            right_outputs = self.right_session.run(self.right_output_names, {self.right_input_name: right_input_tensor})

            # 获取logits
            left_logits = left_outputs[0][0]  
            right_logits = right_outputs[0][0] 

            # 计算概率
            left_raw_probs = self.softmax(left_logits)
            right_raw_probs = self.softmax(right_logits)

            # 卡尔曼滤波
            left_filtered_probs = self.left_kalman.update(left_raw_probs)
            right_filtered_probs = self.right_kalman.update(right_raw_probs)

            # 融合概率
            fused_probs = (left_filtered_probs + right_filtered_probs) / 2

            # 阈值判断并执行控制
            left_swing_prob = fused_probs[1]  # 左摆概率
            right_swing_prob = fused_probs[2]  # 右摆概率

            control_action = None
            if self.actuator:  # 只有在执行器可用时才执行控制
                if left_swing_prob >= self.left_threshold:
                    control_action = "LEFT_SWING"
                    self.actuator.execute_left_swing()
                    self.control_count += 1
                    
                elif right_swing_prob >= self.right_threshold:
                    control_action = "RIGHT_SWING"
                    self.actuator.execute_right_swing()
                    self.control_count += 1

            return fused_probs, control_action, (left_filtered_probs, right_filtered_probs)

        except Exception as e:
            print(f"预测控制出错: {e}")
            return None, None, None

    def format_prediction_output(self, fused_probs, control_action, individual_probs, packet_num, timestamp):
        """格式化预测输出"""
        output_lines = []
        output_lines.append(f"\n{'=' * 80}")
        output_lines.append(f"包序号: {packet_num:06d} | 时间: {timestamp}")
        
        if fused_probs is not None:
            fused_pred = np.argmax(fused_probs)
            output_lines.append(f"融合预测: {self.class_names[fused_pred]} ({fused_probs[fused_pred]:.3f})")
            output_lines.append(f"融合概率: [正常:{fused_probs[0]:.3f}, 左摆:{fused_probs[1]:.3f}, 右摆:{fused_probs[2]:.3f}]")
            
            if individual_probs:
                left_probs, right_probs = individual_probs
                left_pred = np.argmax(left_probs)
                right_pred = np.argmax(right_probs)
                output_lines.append(f"左腿: {self.class_names[left_pred]} ({left_probs[left_pred]:.3f}) | 右腿: {self.class_names[right_pred]} ({right_probs[right_pred]:.3f})")
            
            if control_action:
                output_lines.append(f"🎯 执行动作: {control_action} (#{self.control_count})")
            
            if self.actuator:
                output_lines.append(f"🤖 执行器状态: {self.actuator.current_state.name}")
        else:
            left_status = f"数据不足({len(self.left_buffer)}/{self.seq_len})"
            right_status = f"数据不足({len(self.right_buffer)}/{self.seq_len})"
            output_lines.append(f"状态: 左腿-{left_status} | 右腿-{right_status}")

        # 统计信息
        elapsed_time = time.time() - self.start_time
        fps = self.prediction_count / elapsed_time if elapsed_time > 0 else 0
        output_lines.append(f"统计: 包数:{self.packet_count} | 预测:{self.prediction_count} | 控制:{self.control_count} | 速度:{fps:.1f}FPS")
        output_lines.append(f"{'=' * 80}")

        return '\n'.join(output_lines)

    def run_realtime_prediction(self, max_duration=None, prediction_interval=5):
        """运行实时预测"""
        print(f"\n🚀 开始实时预测控制... (每{prediction_interval}个包预测一次)")
        print("按 Ctrl+C 停止")

        try:
            start_time = time.time()

            while True:
                # 检查运行时间
                if max_duration and (time.time() - start_time) > max_duration:
                    break

                # 读取数据包
                packet = self.read_emg_packet()
                if packet is None:
                    continue

                self.packet_count += 1

                # 分离左右腿数据
                left_data = packet[0:4]  # 通道1-4 (左腿4通道)
                right_data = packet[4:8]  # 通道5-8 (右腿4通道)

                # 添加到缓冲区
                self.left_buffer.append(left_data)
                self.right_buffer.append(right_data)

                # 定期进行预测和控制
                if self.packet_count % prediction_interval == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    fused_probs, control_action, individual_probs = self.predict_and_control()

                    # 输出结果
                    output_str = self.format_prediction_output(
                        fused_probs, control_action, individual_probs, self.packet_count, timestamp)
                    print(output_str)

                    self.prediction_count += 1

                # 数据接收指示
                if self.packet_count % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = self.packet_count / elapsed if elapsed > 0 else 0
                    print(f"📡 已接收 {self.packet_count} 个数据包 (速率: {rate:.1f} 包/秒)")

        except KeyboardInterrupt:
            print("\n用户中断，正在停止...")
        except Exception as e:
            print(f"\n预测过程出错: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n🔧 正在清理系统资源...")
        
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("✓ EMG串口已关闭")

        if self.actuator:
            self.actuator.cleanup()

        elapsed_time = time.time() - self.start_time
        print(f"\n📊 运行总结:")
        print(f"  总运行时间: {elapsed_time:.1f} 秒")
        print(f"  总接收包数: {self.packet_count}")
        print(f"  总预测次数: {self.prediction_count}")
        print(f"  总控制次数: {self.control_count}")
        print(f"  平均接收速度: {self.packet_count / elapsed_time:.1f} 包/秒")
        if self.prediction_count > 0:
            print(f"  平均预测速度: {self.prediction_count / elapsed_time:.1f} FPS")
        print("✅ 系统已安全关闭")

# 使用示例
def main():
    # 配置路径 - 基于你的工作代码
    LEFT_MODEL_PATH = r"/home/a123/alex/emg_left.onnx"     # 左腿ONNX模型路径
    RIGHT_MODEL_PATH = r"/home/a123/alex/emg_right.onnx"   # 右腿ONNX模型路径
    EMG_PORT = '/dev/ttyACM0'                              # EMG数据串口
    
    # 执行器配置
    CAN_INTERFACE = 'can0'         # CAN接口
    SERVO1_PORT = '/dev/ttyUSB0'   # 舵机1控制串口 (1000K波特率)
    SERVO2_PORT = '/dev/ttyUSB1'   # 舵机2控制串口 (1000K波特率)
    
    # 控制阈值
    LEFT_THRESHOLD = 0.4
    RIGHT_THRESHOLD = 0.5

    try:
        # 创建EMG预测控制系统
        system = DualLegEMGActuatorSystem(
            left_model_path=LEFT_MODEL_PATH,
            right_model_path=RIGHT_MODEL_PATH,
            port=EMG_PORT,
            baudrate=115200,
            can_interface=CAN_INTERFACE,
            servo1_port=SERVO1_PORT,
            servo2_port=SERVO2_PORT,
            left_threshold=LEFT_THRESHOLD,
            right_threshold=RIGHT_THRESHOLD,
            seq_len=200
        )

        # 开始实时预测控制
        system.run_realtime_prediction(
            max_duration=None,  # 无限运行
            prediction_interval=5  # 每5个包预测一次
        )

    except Exception as e:
        print(f"系统启动失败: {e}")
        print("\n故障排查:")
        print("1. 检查ONNX模型文件路径")
        print("2. 检查EMG串口连接")
        print("3. 检查CAN接口: sudo ip link set can0 up type can bitrate 1000000")
        print("4. 检查舵机串口权限: sudo usermod -a -G dialout $USER")

if __name__ == "__main__":
    main()