import os
import math
import tkinter as tk
from decimal import Decimal, InvalidOperation
from tkinter import filedialog, messagebox
from openpyxl import load_workbook
import yaml

excel_path = ""

def to_int_str(v, maximum=None):
    """Convert an Excel cell to an exact, optionally bounded integer code."""
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        value = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"DAC 码不是数字：{v!r}") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"DAC 码必须是整数：{v!r}")
    code = int(value)
    if code < 0 or (maximum is not None and code > maximum):
        raise ValueError(f"DAC 码超出安全范围 0～{maximum}：{code}")
    return str(code)
    
def split_number(x):
    try:
        s = f"{float(x):.3f}"
        integer_part, decimal_part = s.split('.')
        return int(integer_part), int(decimal_part)
    except Exception as e:
        print(f"excel读取的值:{x}")
        raise ValueError(f"无效波长：{x!r}") from e

def dac_main(ws):
    lines = []
    for row in range(2, ws.max_row + 1):
        # 从1开始计数
        g = ws.cell(row=row, column=7).value
        h = ws.cell(row=row, column=8).value
        i = ws.cell(row=row, column=9).value
        j = ws.cell(row=row, column=10).value
        k = ws.cell(row=row, column=11).value

        if g is None and h is None and i is None:
            continue

        cells = (g, h, i, j, k)
        limits = (58982, 58982, 32768, 24576, 24576)
        codes = [to_int_str(value, limit) for value, limit in zip(cells, limits)]
        g_s, h_s, i_s, j_s, k_s = codes
        lines.append(f"{{{g_s}, {h_s}, {i_s}, {j_s}, {k_s}}},")

    default_name = os.path.splitext(os.path.basename(excel_path))[0] + "_dac"
    save_path = filedialog.asksaveasfilename(
        title="保存为TXT",
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not save_path:
        return

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("完成", f"已生成TXT：\n{save_path}")
    except Exception as ex:
        messagebox.showerror("保存失败", f"保存TXT时出错：\n{ex}")

def wave_main(ws):
    results = []
    yaml_results = []
    for row in range(2, ws.max_row + 1):
        # 空值跳过（你也可以改成写 {0,0}）
        value = ws.cell(row=row, column=13).value
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        integer_part, decimal_part = split_number(value)
        results.append(f"{{{integer_part},{decimal_part}}},")
        yaml_results.append([integer_part,decimal_part])

    default_name = os.path.splitext(os.path.basename(excel_path))[0] + "_wave"
    save_path = filedialog.asksaveasfilename(
        title="保存为txt",
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[("Text files", "*.txt")]
    )

    # print(results)
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for line in results:
                    f.write(line + "\n")
            messagebox.showinfo("完成", f"已生成TXT：\n{save_path}")
        except Exception as ex:
            messagebox.showerror("保存失败", f"保存TXT时出错：\n{ex}")

    yaml_path = filedialog.asksaveasfilename(
        title="保存为yaml",
        defaultextension=".yaml",
        initialfile=default_name,
        filetypes=[("Yaml files", "*.yaml")]
    )

    if yaml_path:
        yaml_data = {"Wave_DATA":yaml_results}
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    yaml_data,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False
                )
            messagebox.showinfo("完成", f"已生成TXT：\n{yaml_path}")
        except Exception as ex:
            messagebox.showerror("保存失败", f"保存TXT时出错：\n{ex}")

def excel_main():
    root = tk.Tk()
    root.withdraw()
    root.update()

    global excel_path
    excel_path = filedialog.askopenfilename(
        title="选择Excel文件",
        filetypes=[
            ("Excel files", "*.xlsx *.xlsm *.xltx *.xltm"),
            ("All files", "*.*"),
        ],
    )
    if not excel_path:
        return

    try:
        wb = load_workbook(excel_path, data_only=True)
        ws = wb.active
    except Exception as ex:
        messagebox.showerror("读取失败", f"读取Excel时出错：\n{ex}")
        return
    
    try:
        dac_main(ws)
        wave_main(ws)
    except ValueError as ex:
        messagebox.showerror("数据无效", str(ex))
    finally:
        wb.close()

if __name__ == "__main__":
    excel_main()
