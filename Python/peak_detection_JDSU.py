import sys
import os
from pathlib import Path

# A copied/local virtual environment does not always expose Qt's platform
# plugins through the process search path.  Configure it before pyqtgraph or
# PyQt5 imports Qt, matching the supported main application.
qt_platform_plugins = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "PyQt5"
    / "Qt5"
    / "plugins"
    / "platforms"
)
if qt_platform_plugins.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_platform_plugins)

import serial
import math
import time
import yaml
import pyvisa
import openpyxl
import threading
import statistics
from decimal import Decimal, InvalidOperation

import pyqtgraph as pg
import numpy as np

from openpyxl import load_workbook
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,QLineEdit,QPushButton,
    QVBoxLayout, QLabel, QStackedWidget, QAction, QMessageBox,
    QFileDialog, QPlainTextEdit
)
from PyQt5.QtCore import (
    QThread, pyqtSignal
)
from serial.tools import list_ports
from datetime import datetime
from collections import deque

# ====== 参数（按需改）======
ADDR = "GPIB0::7::INSTR"

PRINT_EXCEL_EVERY = 1      # 每隔N行打印一次Excel数据（1=每行都打印）
PRINT_ACK = True           # 打印接收到的ACK

VISA_TIMEOUT_MS = 3000
VISA_RETRY = 2

FLUSH_EVERY_N = 10
XLSX_BASENAME = f"AQ6150B_log_{time.strftime('%H%M%S')}.xlsx"

ACK_VALUE = 0x21
ACK_RESEND_SLEEP_S = 0.001  # 每次重发后短暂停一下，避免占满CPU

array_size = 1904

tx_size = 9
# ==========================

ser = serial.Serial(timeout=0.2)

STM32_CDC_VID_PID = (0x0483, 0x5740)

ser_open = False
ser_cond = threading.Condition()

ap_open = False
ap_cond = threading.Condition()

switch_mode_enable = True

rx_queue = deque(maxlen=4000)

rx_buffer = bytearray()

frames_queue = deque(maxlen=200)

# def serial_thread():
#     pack_size = 4+array_size*8+2+60*4+5 # 帧长度是变长的，但是接收的时候按定长为标准，接收不定长的帧
#     def process_buffer(buf):
#         while True:
#             idx = buf.find(b'\xee\xee')
#             if idx<0:
#                 if len(buf) > 1:
#                     del buf[:-1]
#                     break
#             elif idx>0:
#                 # print(buf)
#                 del buf[:idx]
            
#             end_idx = buf.find(b'\xff\xff')
#             if end_idx<0:
#                 break

#             frame = buf[idx:end_idx+2]
#             # print(frame)

#             if frame[-2:] == b'\xff\xff':
#                 frames_queue.append(frame)
#                 del buf[:end_idx+2]

#     while True:
#         with ser_cond:
#             while not ser_open:
#                 ser_cond.wait()
#                 rx_buffer.clear()
#         if ser.is_open:
#             try:
#                 data = ser.read(pack_size)
#             except Exception as e:
#                 print("Serial Error:",e)
#                 continue
#             rx_buffer.extend(data)
#             # print(rx_buffer)
#             process_buffer(rx_buffer)

# threading.Thread(target=serial_thread, daemon=True).start()

def get_desktop_path():
    return str(Path.home() / "Desktop")

def serial_write(info: bytes):
    ser.write(info)

def wait_ack_0x21_forever(regDataCommand: bytes, should_continue=None) -> bool:
    """
    一直等待直到收到 ACK=0x21。
    期间如果没收到或收到的不是0x21，就重发同一条命令，不推进。
    """
    while should_continue is None or bool(should_continue()):
        b = ser.read(1)
        if b:
            v = int.from_bytes(b, "big")
            if PRINT_ACK:
                print(f"接收到数据: {v}")
            if v == ACK_VALUE:
                return True
        # 没收到或不是0x21 -> 重发
        serial_write(regDataCommand)
        time.sleep(ACK_RESEND_SLEEP_S)
    return False

def try_write(inst, cmd):
    try:
        inst.write(cmd)
        return True
    except Exception:
        return False

def excel_operate(iter_excel):
    """
    返回：(cmd_bytes, info_str)
    """
    try:
        row = next(iter_excel)
    except StopIteration:
        return None, None
    if len(row) < 7:
        raise ValueError("波长表数据行少于7列")

    def exact_uint16(value, name):
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise ValueError(f"{name}必须是0..65535整数") from None
        if (
            not parsed.is_finite()
            or parsed != parsed.to_integral_value()
            or not 0 <= parsed <= 0xFFFF
        ):
            raise ValueError(f"{name}必须是0..65535整数")
        return int(parsed)

    readPhase = exact_uint16(row[4], "phase")
    readwaveA = exact_uint16(row[5], "waveA")
    readwaveB = exact_uint16(row[6], "waveB")

    bWrite = [0] * tx_size
    bWrite[0] = 0xFF
    bWrite[1] = 0xFF
    bWrite[2] = 0x00
    bWrite[3] = (readPhase >> 8) & 0xFF
    bWrite[4] = readPhase & 0xFF
    bWrite[5] = (readwaveA >> 8) & 0xFF
    bWrite[6] = readwaveA & 0xFF
    bWrite[7] = (readwaveB >> 8) & 0xFF
    bWrite[8] = readwaveB & 0xFF

    cmd = bytes(bWrite)
    info = f"phase={readPhase}, waveA={readwaveA}, waveB={readwaveB}"
    return cmd, info

def parse_arr(reply: str):
    if reply is None:
        return []
    s = str(reply).strip().replace("\r", "").replace("\n", "")
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

def trunc6(x):
    """mW 建议保留 6 位小数更有意义"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "nan"

def trunc3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{math.trunc(float(x) * 1000) / 1000.0:.3f}"
    except Exception:
        return "nan"

def dbm_to_mw(dbm: float) -> float:
    # mW = 10^(dBm/10)
    return 10 ** (dbm / 10.0)

def get_top2_peaks_from_arrays(wav_list, pow_list_dbm):
    n = min(len(wav_list), len(pow_list_dbm))
    wav_list = wav_list[:n]
    pow_list_dbm = pow_list_dbm[:n]

    def is_small_int(x):
        return isinstance(x, (int, float)) and abs(x - round(x)) < 1e-9 and 0 <= x <= 200

    # 去掉头部计数 N
    if n >= 2 and is_small_int(wav_list[0]) and is_small_int(pow_list_dbm[0]):
        w1 = wav_list[1]
        if (500 <= w1 <= 20000) or (1e-9 <= w1 <= 1e-3):
            wav_list = wav_list[1:]
            pow_list_dbm = pow_list_dbm[1:]
            n -= 1

    # 单位判断：m->nm / nm直接用
    wav_nm = []
    for w in wav_list:
        w = float(w)
        if 1e-9 <= w <= 1e-3:
            wav_nm.append(w * 1e9)
        else:
            wav_nm.append(w)

    valid = [i for i in range(n) if 500 <= wav_nm[i] <= 20000]
    if not valid:
        return float("nan"), float("nan"), float("nan"), float("nan")

    valid.sort(key=lambda i: float(pow_list_dbm[i]), reverse=True)

    i1 = valid[0]
    wav1_nm = wav_nm[i1]
    pow1_mw = dbm_to_mw(float(pow_list_dbm[i1]))

    wav2_nm = pow2_mw = float("nan")
    if len(valid) >= 2:
        i2 = valid[1]
        wav2_nm = wav_nm[i2]
        pow2_mw = dbm_to_mw(float(pow_list_dbm[i2]))

    return wav1_nm, pow1_mw, wav2_nm, pow2_mw

def read_two_peaks_stable(inst):
    inst.write(":INIT")
    inst.write("*TRG")
    wav_reply = inst.query(":FETC:ARR:POW:WAV?")
    pow_reply = inst.query(":FETC:ARR:POW?")  # dBm
    wav_list = parse_arr(wav_reply)
    pow_list_dbm = parse_arr(pow_reply)
    return get_top2_peaks_from_arrays(wav_list, pow_list_dbm)

class MQComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        self.refresh_ports()

    def showPopup(self):
        self.refresh_ports()
        super().showPopup()

    def refresh_ports(self):
        current_port = self.currentData()

        self.blockSignals(True)
        self.clear()

        native_port = None
        for p in list_ports.comports():
            text = f"{p.device} {p.description}"
            self.addItem(text, p.device)
            if (
                getattr(p, "vid", None),
                getattr(p, "pid", None),
            ) == STM32_CDC_VID_PID and native_port is None:
                native_port = p.device

        if current_port:
            index = self.findData(current_port)
            if index >= 0:
                self.setCurrentIndex(index)
                self.blockSignals(False)
                return
        if native_port is not None:
            self.setCurrentIndex(self.findData(native_port))
        else:
            self.setCurrentIndex(-1)
        
        self.blockSignals(False)

class LogWidget(QPlainTextEdit):
    def __init__(self, max_lines=1000, readOnly=True):
        super().__init__()

        self.setReadOnly(readOnly)
        self.max_lines = max_lines

        self.setStyleSheet("""
        QPlainTextEdit{
            background-color: #1e1e1e;
            color: #dddddd;
            font-family: Consolas;
            font-size: 12px
        }
        """)

    def log(self, message, level="INFO"):
        time_str = datetime.now().strftime("%H:%M:%S")
        
        if level == "ERROR":
            text = f'<span style="color:#ff5555">[{time_str}] {message}</span>'
        elif level == "WARNING":
            text = f'<span style="color:#ffaa00">[{time_str}] {message}</span>'
        else:
            text = f'<span style="color:#dddddd">[{time_str}] {message}</span>'

        self.appendHtml(text)

        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        if self.blockCount() > self.max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

class peakWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        pack_size = 4+array_size*8+2+60*4+5 # 帧长度是变长的，但是接收的时候按定长为标准，接收不定长的帧
        def process_buffer(buf):
            while True:
                idx = buf.find(b'\xee\xee')
                if idx<0:
                    if len(buf) > 1:
                        del buf[:-1]
                        break
                elif idx>0:
                    # print(buf)
                    del buf[:idx]
                    idx = 0
                
                end_idx = buf.find(b'\xff\xff')
                if end_idx<0:
                    break

                frame = buf[idx:end_idx+2]
                # print(frame)

                if frame[-2:] == b'\xff\xff':
                    frames_queue.append(frame)
                    del buf[:end_idx+2]

        while self.running:
            with ser_cond:
                while not ser_open and self.running:
                    ser_cond.wait()
                    rx_buffer.clear()
            if not self.running:
                break
            if ser.is_open:
                try:
                    data = ser.read(pack_size)
                except Exception as e:
                    print("Serial Error:",e)
                    continue
                rx_buffer.extend(data)
                # print(rx_buffer)
                process_buffer(rx_buffer)


class APWorker(QThread):
    log_signal = pyqtSignal(str,str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.running = True

    def log(self, msg, level="info"):
        self.log_signal.emit(msg, level)

    def run(self):
        try:
            excel_path = self.file_path
            if not excel_path:
                self.log("未选择文件", "WARNING")
                return
            self.log(f"执行文件: {excel_path}")

            wb_in = load_workbook(excel_path, read_only=True, data_only=True)
            ws_in = wb_in.active
            iter_excel = ws_in.iter_rows(min_row=2, values_only=True)

            rm = pyvisa.ResourceManager()
            inst = rm.open_resource(ADDR)
            inst.timeout = VISA_TIMEOUT_MS
            inst.read_termination = "\n"
            inst.write_termination = "\n"

            try:
                try:
                    self.log(f"Connected: {inst.query('*IDN?').strip()}")
                except Exception:
                    self.log("提示：*IDN? 不响应也没关系。")

                # 设置单位仍然用 dBm（我们在软件里转 mW）
                try_write(inst, ":INIT:CONT OFF")
                try_write(inst, ":TRIG:SOUR BUS")
                try_write(inst, ":FORM:NDAT 0NM")
                try_write(inst, ":UNIT:POW DBM")

                desktop = get_desktop_path()
                xlsx_path = Path(desktop) / XLSX_BASENAME
                # file_exists = os.path.isfile(xlsx_path)

                # with open(xlsx_path, "a", newline="", encoding="utf-8") as f:
                #     writer = csv.writer(f)
                wb_out = openpyxl.Workbook()
                ws_out = wb_out.active

                head_data = ["timestamp_iso",
                        "peak1_wavelength_nm", "peak1_power_mW",
                        "peak2_wavelength_nm", "peak2_power_mW"]
                gap_len = len(head_data)+1

                self.log("开始记录(ACK 不成功则一直重发同一行，不推进 Excel)。Ctrl+C 结束。")

                count = 0
                # since_flush = 0
                # since_row = 2
                loop_time = -1

                # 缓存当前行（ACK 成功后才读取下一行）
                current_cmd = None
                current_info = None
                last_current = True

                while self.running:
                    # 循环写入表头
                    if last_current:
                        count=0
                        loop_time += 1
                        iter_excel = ws_in.iter_rows(min_row=2, values_only=True)
                        for i, value in enumerate(head_data):
                            ws_out.cell(row=1, column=i+1+loop_time*gap_len, value=value)
                        last_current = False
                    
                    # 只有当当前没有待发送行时，才取 Excel 下一行
                    if current_cmd is None:
                        cmd, info = excel_operate(iter_excel)
                        if cmd is None:
                            last_current = True
                            continue
                        current_cmd, current_info = cmd, info

                    # 打印 Excel 读取内容（可控频率）
                    if PRINT_EXCEL_EVERY > 0 and ((count + 1) % PRINT_EXCEL_EVERY == 0):
                        self.log("----excel----")
                        self.log(current_info)
                        self.log("----excel----")

                    # 串口发送
                    serial_write(current_cmd)

                    # 等 ACK（不成功就一直重发，不推进）
                    if not wait_ack_0x21_forever(
                        current_cmd, should_continue=lambda: self.running
                    ):
                        break

                    # ACK 成功：计数并“推进到下一行”
                    count += 1
                    self.log("计数:" + str(count))
                    current_cmd = None
                    current_info = None

                    # 读 AQ（失败短重试）
                    ts = datetime.now().isoformat(timespec="milliseconds")
                    wav1_nm = pow1_mw = wav2_nm = pow2_mw = float("nan")
                    for _ in range(VISA_RETRY + 1):
                        try:
                            wav1_nm, pow1_mw, wav2_nm, pow2_mw = read_two_peaks_stable(inst)
                            break
                        except Exception:
                            time.sleep(0.01)

                    # 写 CSV（波长保留3位，功率mW保留6位）
                    cur_row = [ts, trunc3(wav1_nm), trunc6(pow1_mw), trunc3(wav2_nm), trunc6(pow2_mw)]
                    for i, value in enumerate(cur_row):
                        ws_out.cell(row=count+1, column=i+1+loop_time*gap_len, value=value)

            except KeyboardInterrupt:
                self.log("\n用户中断, 已停止.")
                wb_out.save(xlsx_path)

            finally:
                try:
                    inst.close()
                finally:
                    rm.close()
                wb_out.save(xlsx_path)
                self.log("结束.")

        except Exception as e:
            self.log(str(e), "ERROR")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("可调谐激光器")
        self.resize(1000, 800)

        self.stack = QStackedWidget()
        self.stack.currentChanged.connect(self.on_switch_mode)
        
        self.page_peak = GraphWindow()
        self.page_ap6150 = ap6150bWindow()

        self.stack.addWidget(self.page_peak)
        self.stack.addWidget(self.page_ap6150)

        self.setCentralWidget(self.stack)

        self.MCU_mode = 0

        self.init_menu()

    def init_menu(self):
        menubar = self.menuBar()

        menu_page = menubar.addMenu("工作模式")

        action_peak = QAction("寻峰模式", self)
        action_ap6150 = QAction("扫波长模式", self)

        menu_page.addAction(action_peak)
        menu_page.addAction(action_ap6150)

        action_peak.triggered.connect(self.open_peak_page)
        action_ap6150.triggered.connect(self.open_ap6150_page)

    def open_peak_page(self):
        if switch_mode_enable:
            self.stack.setCurrentIndex(0)
        else:
            QMessageBox.warning(self, "警告", "先将当前页面工作流停止再进行切换")

    def open_ap6150_page(self):
        if switch_mode_enable:
            self.stack.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "警告", "先将当前页面工作流停止再进行切换")

    def on_switch_mode(self, index):
        self.MCU_mode = index
        command_frame = [0]*9
        command_frame[0] = 0xFF
        command_frame[1] = 0xFF
        command_frame[2] = 0x01
        command_frame[3] = 0x02
        command_frame[8] = self.MCU_mode & 0xFF

        if not ser.is_open:
            try:
                ser.open()
                ser.write(bytes(command_frame))
                ser.close()
            except Exception as e:
                QMessageBox.warning(self, "警告", "改工作模式前确保串口端口和波特率选对")
                QMessageBox.warning(self, "crash", f"发生异常:\n{str(e)}")
                return
        else:
            ser.write(bytes(command_frame))
        # QMessageBox.information(self, "提示", "工作模式已修改")

class ap6150bWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.ctrl_panel = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(5,5,5,5)
        self.ctrl_layout.setSpacing(10)
        self.ctrl_panel.setLayout(self.ctrl_layout)
        self.ctrl_panel.setMaximumHeight(80)

        lab_com = QtWidgets.QLabel("COM端口:")
        self.combo_com = MQComboBox()
        self.combo_com.setMinimumWidth(100)
        if self.combo_com.count() > 0:
            self.combo_com.setCurrentIndex(0)

        lab_baud = QtWidgets.QLabel("波特率:")
        self.combo_baud = QtWidgets.QComboBox()
        self.combo_baud.setMinimumWidth(100)
        self.combo_baud.addItems([
            "9600",
            "115200",
            "2000000",
            "3000000",
            "4000000",
            "6000000"
        ])
        self.combo_baud.setCurrentText("2000000")

        self.fileText_edit = QLineEdit()
        self.fileText_edit.setPlaceholderText("选择波长数据文件")
        self.file_button = QPushButton("...")
        self.file_path = ""

        self.com_btn = QtWidgets.QPushButton("开始")
        self.com_btn.setMinimumWidth(100)

        self.clear_btn = QtWidgets.QPushButton("清空日志")
        self.clear_btn.setMinimumWidth(100)

        self.file_button.clicked.connect(self.select_file)
        self.com_btn.clicked.connect(self.ap_thread)
        self.clear_btn.clicked.connect(self.on_clear)

        self.ctrl_layout.addWidget(lab_com)
        self.ctrl_layout.addWidget(self.combo_com)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(lab_baud)
        self.ctrl_layout.addWidget(self.combo_baud)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(self.fileText_edit)
        self.ctrl_layout.addWidget(self.file_button)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(self.com_btn)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(self.clear_btn)

        layout.addWidget(self.ctrl_panel, 0, 0)

        self.printf_area = LogWidget()
        layout.addWidget(self.printf_area)

        # self.worker = APWorker(self.file_path)
        # self.worker.log_signal.connect(self.printf_area.log)

    def select_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择Excel文件",
            get_desktop_path(),
            "ALL Files (*);;Text Files (*.xlsx)"
        )
        self.fileText_edit.setText(self.file_path)

    def ap_thread(self):
        global ser
        global ap_open
        global switch_mode_enable

        do_open = False
        do_close = False

        if ap_open == False:
            if not self.file_path:
                QMessageBox.warning(self, "未选择文件", "请先选择波长 Excel 表")
                return
            do_open = True
            ap_open = True
        else: 
            do_close = True
            ap_open = False

        if do_open:
            try:
                ser.open()
            except Exception as e:
                print("Serial Error:",str(e))
                with ap_cond:
                    ap_open = False
                return

            switch_mode_enable = False
            self.combo_com.setEnabled(False)
            self.combo_baud.setEnabled(False)

            self.com_btn.setText("关闭")
            self.worker = APWorker(self.file_path)
            self.worker.log_signal.connect(self.printf_area.log)
            self.worker.start()
        elif do_close:
            try:
                ser.close()
            except Exception as e:
                print("Serial Error:",str(e))
                with ap_cond:
                    ap_open = True
                return
            
            switch_mode_enable = True
            self.combo_com.setEnabled(True)
            self.combo_baud.setEnabled(True)

            self.worker.running = False
            self.worker.quit()
            self.worker.wait()

            self.com_btn.setText("打开")

    def on_clear(self):
        self.printf_area.clear()

class GraphWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.paused = False
        self.process_down = True
        self.peaks_lines = [[] for _ in range(4)]
        self.visible_lines = [False for _ in range(4)]

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.ctrl_panel = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(5,5,5,5)
        self.ctrl_layout.setSpacing(10)
        self.ctrl_panel.setLayout(self.ctrl_layout)
        self.ctrl_panel.setMaximumHeight(80)

        lab_com = QtWidgets.QLabel("COM端口:")
        self.combo_com = MQComboBox()
        self.combo_com.setMinimumWidth(100)
        if self.combo_com.count() > 0:
            self.combo_com.setCurrentIndex(0)

        lab_baud = QtWidgets.QLabel("波特率:")
        self.combo_baud = QtWidgets.QComboBox()
        self.combo_baud.setMinimumWidth(100)
        self.combo_baud.addItems([
            "9600",
            "115200",
            "2000000",
            "3000000",
            "4000000",
            "6000000"
        ])
        self.combo_baud.setCurrentText("2000000")

        self.com_btn = QtWidgets.QPushButton("打开")
        self.com_btn.setMinimumWidth(100)

        lab_mode = QtWidgets.QLabel("工作模式:")
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.setMinimumWidth(100)
        self.combo_mode.addItems([
            "寻峰模式",
            "扫波长模式",
        ])
        self.combo_mode.setCurrentText("寻峰模式")
        self.MCU_mode = 0

        lab_interval = QtWidgets.QLabel("单数据间隔(us):")
        self.interval_text = QtWidgets.QTextEdit("10")
        self.interval_text.setMinimumWidth(100)
        self.interval_text.setMaximumHeight(25)

        self.clear_btn = QtWidgets.QPushButton("清空数据")
        self.clear_btn.setMinimumWidth(100)

        self.combo_com.currentTextChanged.connect(self.on_com_changed)
        self.combo_baud.currentTextChanged.connect(self.on_baud_changed)
        self.com_btn.clicked.connect(self.on_open_changed)
        self.combo_mode.currentTextChanged.connect(self.on_switch_mode)
        self.interval_text.textChanged.connect(self.on_interval_changed)
        self.clear_btn.clicked.connect(self.on_clear_chart)
        
        self.ctrl_layout.addWidget(lab_com)
        self.ctrl_layout.addWidget(self.combo_com)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(lab_baud)
        self.ctrl_layout.addWidget(self.combo_baud)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()
        self.ctrl_layout.addWidget(lab_mode)
        self.ctrl_layout.addWidget(self.combo_mode)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()
        self.ctrl_layout.addWidget(self.com_btn)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(lab_interval)
        self.ctrl_layout.addWidget(self.interval_text)
        self.ctrl_layout.addSpacing(20)
        self.ctrl_layout.addStretch()

        self.ctrl_layout.addWidget(self.clear_btn)
        
        layout.addWidget(self.ctrl_panel, 0, 0)

        self.plot1 = pg.PlotWidget()

        layout.addWidget(self.plot1, 1, 0)

        self.adc1 = deque(maxlen=array_size)
        self.adc2 = deque(maxlen=array_size)
        self.adc3 = deque(maxlen=array_size)
        self.adc4 = deque(maxlen=array_size)

        self.data1 = deque(maxlen=array_size)
        self.data2 = deque(maxlen=array_size)
        self.data3 = deque(maxlen=array_size)
        self.data4 = deque(maxlen=array_size)

        self.waves = [[0 for _ in range(15)] for _ in range(4)]

        self.voltage_range = 5
        self.plot1.setXRange(0,array_size)
        self.plot1.setYRange(0,6)
        self.plot1.getViewBox().setLimits(xMin=-1000,xMax=5000,
                                          yMin=-self.voltage_range,yMax=self.voltage_range)
        self.plot1.showGrid(x=True, y=True)

        self.plot1.addLegend()
        self.curve1 = self.plot1.plot(pen='yellow', name='CH0')
        self.curve2 = self.plot1.plot(pen='green', name='CH1')
        self.curve3 = self.plot1.plot(pen='blue', name='CH2')
        self.curve4 = self.plot1.plot(pen='purple', name='CH3')
        # self.plot1.actionEvent()

        self.num_panel = QtWidgets.QWidget()
        self.num_layout = QtWidgets.QGridLayout()
        self.num_panel.setLayout(self.num_layout)

        header_style = "font-weight:bold; font-size:14pt; padding:3pt;"

        self.check_boxs = []

        for r in range(4):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(False)
            # cb.setAlignment(QtCore.Qt.AlignCenter)
            cb.stateChanged.connect(lambda state, i=r: self.toggle_line(i, state))
            self.check_boxs.append(cb)
            self.num_layout.addWidget(cb, r+1, 0)

        # 左侧表头（行：1~4）
        for r in range(4):
            lab = QtWidgets.QLabel(str(r+1))
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet(header_style)
            self.num_layout.addWidget(lab, r+1, 1)

        # 顶部表头（列：1~15）
        for c in range(15):
            lab = QtWidgets.QLabel(str(c+1))
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet(header_style)
            self.num_layout.addWidget(lab, 0, c+2)

        self.num_labels = [[None for _ in range(15)] for _ in range(4)]

        for r in range(4):
            for c in range(15):
                lab = QtWidgets.QLabel("0")
                lab.setAlignment(QtCore.Qt.AlignCenter)
                lab.setStyleSheet("font-size:14pt; padding:2pt;")
                self.num_labels[r][c] = lab
                self.num_layout.addWidget(lab, r+1, c+2)

        layout.addWidget(self.num_panel, 2, 0, 1, 2)

        self.port = self.combo_com.currentData()
        self.baud = int(self.combo_baud.currentText())
        self.interval = int(self.interval_text.toPlainText())
        global ser
        ser.port = self.port
        ser.baudrate = self.baud

        self.worker = peakWorker()
        
        self.frame_timer = QtCore.QTimer()    
        self.frame_timer.timeout.connect(self.process_frame)
        
        self.update_timer = QtCore.QTimer()    
        self.update_timer.timeout.connect(self.update_plot)
        
        QtWidgets.QShortcut(QtGui.QKeySequence("P"), self, activated=self.toggle_pause)

        wavelength_file = Path(__file__).resolve().with_name("wave_const.yaml")
        with wavelength_file.open('r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
            # print((yaml_data['Wave_DATA']))
            self.yaml = yaml_data['Wave_DATA']

        self.visual_index = 0
        self.visual_y = 0
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r'))
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

        self.label = pg.TextItem(color='w')
        self.plot1.addItem(self.label)

        self.plot1.proxy = pg.SignalProxy(
            self.plot1.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved
        )
        self.worker.start()

    def shutdown_worker(self):
        """Stop the receiver without leaving a live QThread at Qt shutdown."""
        global ser_open
        worker = getattr(self, "worker", None)
        if worker is None:
            return True
        worker.running = False
        with ser_cond:
            ser_open = False
            ser_cond.notify_all()
        if worker.isRunning():
            worker.wait(700)
        return not worker.isRunning()

    def closeEvent(self, event):
        self.frame_timer.stop()
        self.update_timer.stop()
        self.shutdown_worker()
        try:
            if ser.is_open:
                ser.close()
        except Exception:
            pass
        super().closeEvent(event)

    def mouseMoved(self, evt):
        pos = evt[0] 
        if self.plot1.sceneBoundingRect().contains(pos):

            mousePoint = self.plot1.plotItem.vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()

            # 找最近的x索引
            index = int(x)
            if index>=array_size or index<0 or len(self.adc1)==0:
                    return
            
            self.visual_index = index
            self.visual_y = y
            self.update_crosshair(index, y)

    def update_crosshair(self, index, y):
        if index >= array_size or index < 0 \
        or y > self.voltage_range or y < -self.voltage_range \
        or len(self.adc1) == 0:
            return
        
        adc = np.array([
            self.adc1[index],
            self.adc2[index],
            self.adc3[index],
            self.adc4[index]
        ])

        indey = np.abs(adc-y).argmin()

        x_snap = self.yaml[index][0] + self.yaml[index][1]*0.001
        y_snap = adc[indey]

        self.vLine.setPos(index)
        self.hLine.setPos(y_snap)

        self.label.setText(f"x={x_snap:.3f}\ny={y_snap:.3f}")
        self.label.setPos(index, y_snap)

    def on_com_changed(self):
        self.port = self.combo_com.currentData()
        self.update_serial()

    def on_baud_changed(self):
        self.baud = int(self.combo_baud.currentText())
        self.update_serial()
    
    def update_serial(self):
        global ser
        ser.port = self.port
        ser.baudrate = self.baud

    def on_open_changed(self):
        global ser_open
        global ser_cond
        global ser
        global switch_mode_enable

        do_open = False
        do_close = False

        with ser_cond: 
            if ser_open == False:
                do_open = True
                ser_open = True 
                ser_cond.notify_all() 
                
            else: 
                do_close = True
                ser_open = False

        if do_open:
            try:
                ser.open()
            except Exception as e:
                print("Serial Error:",e)
                with ser_cond:
                    ser_open = False
                return
            
            switch_mode_enable = False
            self.combo_com.setEnabled(False)
            self.combo_baud.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.on_interval_changed()

            self.frame_timer.start(1)
            self.update_timer.start(2)

            self.com_btn.setText("关闭")
        elif do_close:
            try:
                ser.close()
            except Exception as e:
                print("Serial Error:",e)
                with ser_cond:
                    ser_open = True
                return
            
            switch_mode_enable = True
            self.combo_com.setEnabled(True)
            self.combo_baud.setEnabled(True)
            self.clear_btn.setEnabled(True)

            self.frame_timer.stop()
            self.update_timer.stop()

            self.com_btn.setText("打开")

    def on_clear_chart(self):
        curves = [self.curve1, self.curve2, self.curve3, self.curve4]
        for idx in range(4):
            curves[idx].setData([])
            for line in self.peaks_lines[idx]:
                self.plot1.removeItem(line)
            self.peaks_lines[idx].clear()
            for label in self.num_labels[idx]:  
                label.setText("0")

    def on_switch_mode(self):
        self.MCU_mode ^= 1
        command_frame = [0]*tx_size
        command_frame[0] = 0xFF
        command_frame[1] = 0xFF
        command_frame[2] = 0x01
        command_frame[3] = 0x02
        command_frame[8] = self.MCU_mode & 0xFF

        if not ser.is_open:
            QMessageBox.information(self, "提示", "改工作模式前确保串口端口和波特率选对")
            try:
                self.com_btn.setEnabled(False)
                ser.open()
                ser.write(bytes(command_frame))
                ser.close()
                self.com_btn.setEnabled(True)
                QMessageBox.information(self, "提示", "工作模式已生效")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"发生异常:\n{str(e)}")
                return
        else:
            ser.write(bytes(command_frame))

    def on_interval_changed(self):
        self.interval = int(self.interval_text.toPlainText())
        command_frame = [0]*tx_size
        command_frame[0] = 0xFF
        command_frame[1] = 0xFF
        command_frame[2] = 0x01
        command_frame[3] = 0x01
        command_frame[7] = (self.interval>>8) & 0xFF
        command_frame[8] = self.interval & 0xFF

        if not ser.is_open:
            QMessageBox.information(self, "提示", "改间隔前确保串口端口和波特率选对")
            try:
                self.com_btn.setEnabled(False)
                ser.open()
                ser.write(bytes(command_frame))
                ser.close()
                self.com_btn.setEnabled(True)
                QMessageBox.information(self, "提示", "间隔已生效")
            except Exception as e:
                QMessageBox.information(self, "警告", f"发生异常:\n{str(e)}")
                return
        else:
            ser.write(bytes(command_frame))

    def process_frame(self):
        try:
            if len(frames_queue)==0 or self.process_down==False:
                # print("no valid frame")
                return
            raw = frames_queue.popleft()
            self.process_down = False
            single_size = 8

            # print(len(raw))
            if raw[0]!=0xEE or raw[1]!=0xEE:
                print("pack head error")
                self.process_down = True
                print(raw)
                return 
            
            for i in range(2, array_size*8+2, single_size):
                com_input = raw[i:i+single_size]

                ch1 = (com_input[0]<<8)+com_input[1]
                ch2 = (com_input[2]<<8)+com_input[3]
                ch3 = (com_input[4]<<8)+com_input[5]
                ch4 = (com_input[6]<<8)+com_input[7]

                v1 = ch1*2.5/4096
                v2 = ch2*2.5/4096
                v3 = ch3*2.5/4096
                v4 = ch4*2.5/4096

                self.adc1.append(v1)
                self.adc2.append(v2)
                self.adc3.append(v3)
                self.adc4.append(v4)

                self.data1.append(ch1)
                self.data2.append(ch2)
                self.data3.append(ch3)
                self.data4.append(ch4)

                if ((raw[array_size*8+2]<<8)+raw[array_size*8+3])!=array_size:
                    print("array_size_error")
                    self.process_down = True
                    return

                if raw[array_size*8+4]!=0xAB:
                    print("wave head error")
                    print(hex(raw[array_size*8+4]))
                    self.process_down = True
                    return
            self.waves = [[0 for _ in range(15)] for _ in range(4)]
            frame_start = array_size*8+4
            data_len = 0
            for i in range(0, 4):
                # frame_start = array_size*8+3+i*61
                frame_start += (data_len*4+1)

                data_len = raw[frame_start]
                # print(data_len)
                data_start = frame_start+1
                for j in range(data_len):
                    cell_start = data_start+j*4
                    # print(cell_start)
                    int_high = raw[cell_start]
                    int_low = raw[cell_start+1]
                    dec_high = raw[cell_start+2]
                    dec_low = raw[cell_start+3]
                    get_wave = (int_high<<8)+int_low+((dec_high<<8)+dec_low)*0.001
                    # print(i)
                    self.waves[i][j] = get_wave
            # print(self.waves)

            self.process_down = True
        except Exception as e:
            print(e)
            print(len(raw))
            self.process_down = True
            return

    def update_plot(self):
        if self.process_down == False:
            return
        self.process_down = False

        x = list(range(array_size))

        # 更新曲线
        self.curve1.setData(x,self.adc1)
        self.curve2.setData(x,self.adc2)
        self.curve3.setData(x,self.adc3)
        self.curve4.setData(x,self.adc4)

        # 更新光标
        self.update_crosshair(self.visual_index, self.visual_y)

        # 更新峰值可视化线
        for i in range(4):
            for j in range(15):
                self.num_labels[i][j].setText("{:.3f}".format(self.waves[i][j]) if not self.waves[i][j]==0 else "0")
            if self.visible_lines[i]:
                self.cal_peaks_line(i)
            else:
                for l in self.peaks_lines[i]:
                    l.setVisible(False)

        self.process_down = True

    def cal_peaks_line(self, i):
        datas = [self.data1, self.data2, self.data3, self.data4]
        _data = datas[i]
        if len(_data) == 0:
            return
        values = np.asarray(_data, dtype=float)
        ma = float(np.max(values))
        mi = float(np.min(values))
        if not ma > mi:
            return
        normalized = (values - mi) / (ma - mi)
        peaks = peak_main(normalized, float(np.mean(normalized)))
        # print(statistics.mean(_data))

        for l in self.peaks_lines[i]:
            l.setVisible(False)

        for idx,p in enumerate(peaks):
            if p==-1:
                continue
            if idx>=len(self.peaks_lines[i]):
                vline = pg.InfiniteLine(
                    pos=p,
                    angle=90,
                    pen=pg.mkPen('r',width=2,style=QtCore.Qt.DashLine)
                )
                self.plot1.addItem(vline)
                self.peaks_lines[i].append(vline)
            else:
                self.peaks_lines[i][idx].setPos(p)
                self.peaks_lines[i][idx].setVisible(True)

    def toggle_line(self, i, state):
        visible = (state == QtCore.Qt.Checked)
        self.visible_lines[i] = visible

    def toggle_pause(self):
        if self.paused: 
            self.frame_timer.start(1)
            self.update_timer.start(2)
            self.paused = False
        else: 
            self.frame_timer.stop()
            self.update_timer.stop()
            self.paused = True

def peak_initial(adc_vec, interval, adc_length, initials_length, threshold):
    initials = [0]*initials_length
    if threshold>0.15:
        return initials
    it = 0
	# for(uint16_t i=1;i<adc_length-1;i++){
    for i in range(1,adc_length-1):
        if it>=initials_length: break
        start = i-interval if i-interval>=0 else 0
        end = i+interval if i+interval<adc_length else adc_length
        front = list(adc_vec)[start:i]
        back = list(adc_vec)[i+1:end]
        if adc_vec[i]>=max(front) and adc_vec[i]>=max(back) and adc_vec[i]>0.6: 
            initials[it] = i
            i += interval
            it+=1
    # print(initials)
    return initials

def peak_main(adc_vec, threshold):
    interval = 25
    adc_length = array_size
    initials_length = 15
	
    initials = peak_initial(adc_vec, interval, adc_length, initials_length, threshold)
    peaks_vec = [-1]*initials_length

    # for(uint8_t i=0;i<10;i++){
    for i in range(initials_length):
        if initials[i]==0:
            break
        start = initials[i]-interval if initials[i]-interval>=0 else 0
        end = initials[i]+interval if initials[i]+interval<adc_length else adc_length
        sumy=0
        sumxy=0
        # for(int j=start;j<end;j++){
        for j in range(start, end):
            sumxy+=adc_vec[j]*j
            sumy+=adc_vec[j]
        peaks_vec[i] = sumxy/sumy

    # print(peaks_vec)
    return peaks_vec

if __name__=="__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton, QLabel {
            font-size: 12pt;
            font-family: Microsoft YaHei;
        }
    """)
    win = GraphWindow()
    win.setWindowTitle("ADC Plot")
    win.resize(1000,800)
    win.show()
    # win = MainWindow()
    # win.show()
    sys.exit(app.exec_())


