#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派硬件连接测试脚本
用于测试CAN总线和双舵机串口连接
"""

import serial
import can
import time
import sys


def test_serial_ports():
    """测试串口连接"""
    print("🔌 测试串口连接...")
    print("-" * 40)

    # 测试串口列表
    test_ports = [
        ('/dev/ttyUSB0', '舵机1'),
        ('/dev/ttyUSB1', '舵机2'),
        ('/dev/ttyACM0', 'EMG数据')
    ]

    results = {}

    for port, description in test_ports:
        try:
            # 尝试以1000K波特率连接舵机
            if '舵机' in description:
                ser = serial.Serial(port, 1000000, timeout=1)
            else:
                # EMG数据端口使用115200
                ser = serial.Serial(port, 115200, timeout=1)

            print(f"✓ {port} ({description}) - 连接成功")
            ser.close()
            results[port] = True

        except Exception as e:
            print(f"✗ {port} ({description}) - 连接失败: {e}")
            results[port] = False

    return results


def test_can_interface():
    """测试CAN总线连接"""
    print("\n🚌 测试CAN总线连接...")
    print("-" * 40)

    try:
        # 尝试连接CAN总线
        bus = can.interface.Bus(
            channel='can0',
            bustype='socketcan'
        )
        print("✓ CAN总线 (can0) - 连接成功")

        # 发送测试消息
        test_msg = can.Message(
            arbitration_id=0x141,
            is_extended_id=False,
            data=[0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  # 停止命令
        )

        bus.send(test_msg)
        print("✓ CAN消息发送测试成功")

        bus.shutdown()
        return True

    except Exception as e:
        print(f"✗ CAN总线连接失败: {e}")
        print("   提示: 请执行 sudo ip link set can0 up type can bitrate 1000000")
        return False


def test_servo_commands():
    """测试舵机命令发送"""
    print("\n🤖 测试舵机命令...")
    print("-" * 40)

    # 舵机测试命令
    test_commands = {
        '/dev/ttyUSB0': {
            'name': '舵机1',
            'lock_cmd': bytes.fromhex("FF FF 01 0B 03 28 01 D0 00 00 B8 0B D0 07 5D"),
            'unlock_cmd': bytes.fromhex("FF FF 01 0B 03 28 01 D0 FC 03 B8 0B D0 07 5E")
        },
        '/dev/ttyUSB1': {
            'name': '舵机2',
            'lock_cmd': bytes.fromhex("FF FF 02 0B 03 28 01 D0 F8 07 B8 0B D0 07 5D"),
            'unlock_cmd': bytes.fromhex("FF FF 02 0B 03 28 01 C8 E8 03 F4 01 96 00 88")
        }
    }

    for port, config in test_commands.items():
        try:
            print(f"\n测试 {config['name']} ({port}):")
            ser = serial.Serial(port, 1000000, timeout=1)

            # 发送锁定命令
            bytes_sent = ser.write(config['lock_cmd'])
            ser.flush()
            print(f"  ✓ 锁定命令发送成功 ({bytes_sent} bytes)")
            time.sleep(0.5)

            # 发送解锁命令
            bytes_sent = ser.write(config['unlock_cmd'])
            ser.flush()
            print(f"  ✓ 解锁命令发送成功 ({bytes_sent} bytes)")

            ser.close()

        except Exception as e:
            print(f"  ✗ {config['name']}测试失败: {e}")


def check_system_info():
    """检查系统信息"""
    print("🍓 系统信息检查...")
    print("-" * 40)

    import os
    import subprocess

    try:
        # 检查用户组
        result = subprocess.run(['groups'], capture_output=True, text=True)
        groups = result.stdout.strip()
        if 'dialout' in groups:
            print("✓ 用户已加入 dialout 组")
        else:
            print("✗ 用户未加入 dialout 组")
            print("   执行: sudo usermod -a -G dialout $USER")
            print("   然后重启或重新登录")

        # 检查CAN模块
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'can' in result.stdout:
            print("✓ CAN内核模块已加载")
        else:
            print("✗ CAN内核模块未加载")
            print("   执行: sudo modprobe can && sudo modprobe can_raw")

        # 检查CAN接口状态
        result = subprocess.run(['ip', 'link', 'show', 'can0'], capture_output=True, text=True)
        if result.returncode == 0:
            if 'UP' in result.stdout:
                print("✓ CAN接口 can0 已启动")
            else:
                print("✗ CAN接口 can0 未启动")
                print("   执行: sudo ip link set can0 up type can bitrate 1000000")
        else:
            print("✗ CAN接口 can0 不存在")
            print("   请检查CAN硬件连接")

    except Exception as e:
        print(f"系统检查出错: {e}")


def main():
    """主测试函数"""
    print("🔧 树莓派硬件连接全面测试")
    print("=" * 60)

    # 系统信息检查
    check_system_info()
    print()

    # 串口连接测试
    serial_results = test_serial_ports()

    # CAN总线测试
    can_result = test_can_interface()

    # 如果串口连接成功，测试舵机命令
    if serial_results.get('/dev/ttyUSB0') or serial_results.get('/dev/ttyUSB1'):
        test_servo_commands()

    # 总结
    print("\n📊 测试总结")
    print("=" * 60)

    all_good = True

    for port, result in serial_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {port}")
        if not result:
            all_good = False

    can_status = "✅" if can_result else "❌"
    print(f"{can_status} CAN总线 (can0)")
    if not can_result:
        all_good = False

    if all_good:
        print("\n🎉 所有硬件连接测试通过！可以运行主程序了。")
    else:
        print("\n⚠️  存在连接问题，请根据上述提示解决后重试。")

    print("\n💡 常见问题解决:")
    print("   1. 权限问题: sudo usermod -a -G dialout $USER (需要重新登录)")
    print("   2. CAN接口: sudo ip link set can0 up type can bitrate 1000000")
    print("   3. 设备路径: ls -la /dev/ttyUSB*")
    print("   4. 重启USB: sudo modprobe -r ftdi_sio && sudo modprobe ftdi_sio")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试脚本出错: {e}")
        sys.exit(1)