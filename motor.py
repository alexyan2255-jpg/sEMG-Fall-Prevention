import serial
import time
import can
from enum import Enum
from datetime import datetime

class ActuatorState(Enum):
    """执行器状态枚举"""
    IDLE = 0
    LEFT_SWING = 2
    RIGHT_SWING = 1

class RaspberryPiActuatorController:
    """树莓派执行器控制器 - 使用SocketCAN和双舵机串口"""

    def __init__(self, can_interface='can0', servo1_port='/dev/ttyUSB0', servo2_port='/dev/ttyUSB1',
                 servo_baudrate=1000000):
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
            "ID2_motor_run": [0xA2, 0x00, 0x00, 0x00, 0x20, 0xBF, 0x02, 0x00],  # 电机2命令
            "motor_stop": [0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]      # 停止
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
            print(f"  {port_name}命令发送: {command_key} ({bytes_written} bytes)")
            return True
        except Exception as e:
            print(f"  舵机命令发送失败 {command_key}: {e}")
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
            print(f"  电机{motor_id}命令发送: {command_key} (ID: 0x{can_id:X})")
            return True
        except Exception as e:
            print(f"  电机{motor_id}命令发送失败 {command_key}: {e}")
            return False

    def execute_left_swing(self):
        """执行左摆动作 - 阻塞版本"""
        try:
            self.current_state = ActuatorState.LEFT_SWING
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 开始执行左摆动作...")

            # 同时发送舵机锁定和电机运行命令
            print("  发送启动命令...")
            servo_success = self.send_servo_command("ID2_servo_lock")
            motor_success = self.send_motor_command(1, "ID1_motor_run")

            if not (servo_success and motor_success):
                print("  警告: 部分启动命令发送失败")

            # 持续1.5秒
            print("  动作执行中 (1.5秒)...")
            time.sleep(1.5)

            # 发送舵机解锁和电机停止命令
            print("  发送停止命令...")
            self.send_servo_command("ID2_servo_unlock")
            self.send_motor_command(1, "motor_stop")

            print("  左摆动作完成")

        except Exception as e:
            print(f"  左摆动作执行异常: {e}")
        finally:
            self.current_state = ActuatorState.IDLE

    def execute_right_swing(self):
        """执行右摆动作 - 阻塞版本"""
        try:
            self.current_state = ActuatorState.RIGHT_SWING
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 开始执行右摆动作...")

            # 同时发送舵机锁定和电机运行命令
            print("  发送启动命令...")
            servo_success = self.send_servo_command("ID1_servo_lock")
            motor_success = self.send_motor_command(2, "ID2_motor_run")

            if not (servo_success and motor_success):
                print("  警告: 部分启动命令发送失败")

            # 持续1.5秒
            print("  动作执行中 (1.5秒)...")
            time.sleep(1.5)

            # 发送舵机解锁和电机停止命令
            print("  发送停止命令...")
            self.send_servo_command("ID1_servo_unlock")
            self.send_motor_command(2, "motor_stop")

            print("  右摆动作完成")

        except Exception as e:
            print(f"  右摆动作执行异常: {e}")
        finally:
            self.current_state = ActuatorState.IDLE

    def idle_wait(self, duration=5):
        """待机等待"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 待机 {duration} 秒...")
        
        # 显示倒计时
        for i in range(duration, 0, -1):
            print(f"  倒计时: {i} 秒", end='\r')
            time.sleep(1)
        print("  待机结束          ")  # 清除倒计时显示

    def cleanup(self):
        """清理资源"""
        print("\n正在清理执行器控制器资源...")

        # 停止所有电机
        try:
            self.send_motor_command(1, "motor_stop")
            self.send_motor_command(2, "motor_stop")
            print("所有电机已停止")
        except:
            pass

        # 关闭串口
        if hasattr(self, 'servo1_serial') and self.servo1_serial.is_open:
            self.servo1_serial.close()
            print("舵机1串口已关闭")
        if hasattr(self, 'servo2_serial') and self.servo2_serial.is_open:
            self.servo2_serial.close()
            print("舵机2串口已关闭")

        # 关闭CAN总线
        if hasattr(self, 'can_bus'):
            self.can_bus.shutdown()
            print("CAN总线已关闭")

        print("执行器控制器资源清理完成")

class ActuatorTestSystem:
    """执行器循环测试系统"""
    
    def __init__(self, can_interface='can0', servo1_port='/dev/ttyUSB0', servo2_port='/dev/ttyUSB1'):
        """初始化测试系统"""
        
        print("=" * 80)
        print("执行器循环测试系统")
        print("=" * 80)
        print("测试序列: 待机5秒 → 左摆 → 待机5秒 → 右摆 → 循环")
        print("=" * 80)
        
        # 初始化执行器控制器
        self.actuator = RaspberryPiActuatorController(
            can_interface=can_interface,
            servo1_port=servo1_port,
            servo2_port=servo2_port
        )
        
        self.cycle_count = 0
        self.start_time = time.time()
        
        print("执行器循环测试系统初始化完成")
        print("=" * 80)

    def run_test_cycle(self, max_cycles=None):
        """运行测试循环"""
        
        print(f"\n开始执行器循环测试...")
        if max_cycles:
            print(f"将执行 {max_cycles} 个循环")
        else:
            print("无限循环 (按 Ctrl+C 停止)")
        print("-" * 80)
        
        try:
            while True:
                # 检查是否达到最大循环次数
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n已完成 {max_cycles} 个循环，测试结束")
                    break
                
                self.cycle_count += 1
                print(f"\n第 {self.cycle_count} 个循环开始")
                print("-" * 40)
                
                # 1. 初始待机5秒
                self.actuator.idle_wait(5)
                
                # 2. 执行左摆
                self.actuator.execute_left_swing()
                
                # 3. 待机5秒
                self.actuator.idle_wait(5)
                
                # 4. 执行右摆
                self.actuator.execute_right_swing()
                
                print(f"第 {self.cycle_count} 个循环完成")
                
                # 显示统计信息
                elapsed = time.time() - self.start_time
                avg_cycle_time = elapsed / self.cycle_count
                print(f"统计: 总时间 {elapsed:.1f}秒 | 平均每循环 {avg_cycle_time:.1f}秒")
                print("-" * 40)
                
        except KeyboardInterrupt:
            print(f"\n\n用户中断，已完成 {self.cycle_count} 个循环")
        except Exception as e:
            print(f"\n\n测试过程出错: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n" + "=" * 80)
        print("正在清理测试系统...")
        
        if hasattr(self, 'actuator'):
            self.actuator.cleanup()
        
        # 显示测试总结
        total_time = time.time() - self.start_time
        print(f"\n测试总结:")
        print(f"  总循环次数: {self.cycle_count}")
        print(f"  总运行时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        if self.cycle_count > 0:
            print(f"  平均每循环: {total_time/self.cycle_count:.1f} 秒")
            print(f"  总左摆次数: {self.cycle_count}")
            print(f"  总右摆次数: {self.cycle_count}")
        
        print("\n执行器循环测试系统已安全关闭")
        print("=" * 80)

def main():
    """主函数"""
    # 执行器配置
    CAN_INTERFACE = 'can0'           # CAN接口
    SERVO1_PORT = '/dev/ttyUSB0'     # 舵机1控制串口 (1000K波特率)
    SERVO2_PORT = '/dev/ttyUSB1'     # 舵机2控制串口 (1000K波特率)
    
    try:
        # 创建测试系统
        test_system = ActuatorTestSystem(
            can_interface=CAN_INTERFACE,
            servo1_port=SERVO1_PORT,
            servo2_port=SERVO2_PORT
        )
        
        # 运行循环测试
        # test_system.run_test_cycle(max_cycles=10)  # 运行10个循环后停止
        test_system.run_test_cycle()  # 无限循环，直到手动停止
        
    except Exception as e:
        print(f"系统启动失败: {e}")
        print("\n故障排查:")
        print("1. 检查CAN接口: sudo ip link set can0 up type can bitrate 1000000")
        print("2. 检查舵机串口权限: sudo usermod -a -G dialout $USER")
        print("3. 检查设备连接: ls /dev/ttyUSB*")
        print("4. 重启后重新登录以应用权限更改")

if __name__ == "__main__":
    main()