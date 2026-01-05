import time
import serial

# 按你的实际设备改：/dev/serial0（板载UART）或 /dev/ttyUSB0（USB转串口）
PORT = "/dev/ttyUSB0"   # 或 "/dev/ttyUSB0"
BAUD = 1_000_000        # 1 Mbps
TIMEOUT = 0.05          # 50 ms 读超时

ID1_servo_lock   = bytes.fromhex("FF FF 01 0B 03 28 01 D0 00 00 B8 0B D0 07 5D")
ID1_servo_unlock = bytes.fromhex("FF FF 01 0B 03 28 01 D0 FC 03 B8 0B D0 07 5E")

def main():
    ser = serial.Serial(
        PORT,
        BAUD,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT
    )

    # 若是半双工转接板，确保已切到“发送”方向（某些板子需要拉高DE/!RE）
    time.sleep(0.1)

    # 发锁定指令
    ser.write(ID1_servo_lock)
    ser.flush()
    # 读回回包（如果舵机会返回 Status Packet；没有也正常）
    time.sleep(0.01)
    resp1 = ser.read(64)
    print("Lock resp:", resp1.hex(" "))

    time.sleep(0.1)

    # 发解锁指令
    ser.write(ID1_servo_unlock)
    ser.flush()
    time.sleep(0.01)
    resp2 = ser.read(64)
    print("Unlock resp:", resp2.hex(" "))

    ser.close()

if __name__ == "__main__":
    main()
