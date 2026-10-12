"""Merge bootloader and relocated application Intel HEX images safely."""

from __future__ import annotations

import argparse
from pathlib import Path

from fbg_ota_protocol import (
    OTA_APPLICATION_END,
    OTA_LINK_ADDRESS,
    OTA_RAM_END,
    OTA_RAM_START,
)


BOOTLOADER_START = 0x08000000

def _record(record_type: int, address: int, data: bytes) -> str:
    body = bytes((len(data), (address >> 8) & 0xFF, address & 0xFF, record_type)) + data
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes((checksum,))).hex().upper()


def read_hex(path: Path) -> dict[int, int]:
    memory: dict[int, int] = {}
    upper = 0
    saw_eof = False
    for number, raw_line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if saw_eof:
            raise ValueError(f"{path.name}:{number} EOF之后仍有记录")
        if not line.startswith(":"):
            raise ValueError(f"{path.name}:{number} 不是 Intel HEX 记录")
        raw = bytes.fromhex(line[1:])
        if len(raw) < 5 or len(raw) != raw[0] + 5 or sum(raw) & 0xFF:
            raise ValueError(f"{path.name}:{number} 长度或校验和错误")
        length = raw[0]
        address = (raw[1] << 8) | raw[2]
        kind = raw[3]
        data = raw[4 : 4 + length]
        if kind == 0:
            absolute = upper + address
            for offset, value in enumerate(data):
                location = absolute + offset
                previous = memory.get(location)
                if previous is not None and previous != value:
                    raise ValueError(f"0x{location:08X} 出现互相冲突的数据")
                memory[location] = value
        elif kind == 4:
            if length != 2 or address != 0:
                raise ValueError("无效的扩展线性地址记录")
            upper = int.from_bytes(data, "big") << 16
        elif kind == 1:
            if length != 0 or address != 0:
                raise ValueError("无效的 Intel HEX EOF 记录")
            saw_eof = True
        elif kind in (3, 5):
            if length != 4:
                raise ValueError(f"无效的 Intel HEX 记录类型 {kind}")
        elif kind == 2:
            if length != 2 or address != 0:
                raise ValueError("无效的扩展段地址记录")
            upper = int.from_bytes(data, "big") << 4
        else:
            raise ValueError(f"不支持的 Intel HEX 记录类型 {kind}")
    if not saw_eof:
        raise ValueError(f"{path.name}缺少 Intel HEX EOF 记录")
    return memory


def _uint32_le(memory: dict[int, int], address: int, label: str) -> int:
    try:
        raw = bytes(memory[address + offset] for offset in range(4))
    except KeyError as exc:
        raise ValueError(f"{label}缺少完整向量表") from exc
    return int.from_bytes(raw, "little")


def _validate_vectors(
    memory: dict[int, int], start: int, end: int, label: str
) -> None:
    initial_sp = _uint32_le(memory, start, label)
    reset_vector = _uint32_le(memory, start + 4, label)
    if not OTA_RAM_START <= initial_sp <= OTA_RAM_END or initial_sp & 3:
        raise ValueError(f"{label}初始栈指针无效：0x{initial_sp:08X}")
    reset_address = reset_vector & ~1
    if not reset_vector & 1 or not start <= reset_address < end:
        raise ValueError(f"{label}复位向量无效：0x{reset_vector:08X}")
    if reset_address not in memory:
        raise ValueError(f"{label}复位向量指向未包含的程序地址")


def write_hex(memory: dict[int, int], output: Path) -> None:
    lines: list[str] = []
    addresses = sorted(memory)
    index = 0
    current_upper = None
    while index < len(addresses):
        start = addresses[index]
        upper = start >> 16
        if upper != current_upper:
            lines.append(_record(4, 0, upper.to_bytes(2, "big")))
            current_upper = upper
        low = start & 0xFFFF
        block = bytearray((memory[start],))
        index += 1
        while (
            index < len(addresses)
            and addresses[index] == start + len(block)
            and addresses[index] >> 16 == upper
            and len(block) < 16
        ):
            block.append(memory[addresses[index]])
            index += 1
        lines.append(_record(0, low, bytes(block)))
    lines.append(_record(1, 0, b""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def merge(bootloader: Path, application: Path, output: Path) -> None:
    if output.resolve() in {bootloader.resolve(), application.resolve()}:
        raise ValueError("输出 HEX 不能覆盖引导或应用源文件")
    memory = read_hex(bootloader)
    application_memory = read_hex(application)
    if not memory or min(memory) != BOOTLOADER_START:
        raise ValueError("引导固件不从 0x08000000 开始")
    if any(not BOOTLOADER_START <= address < OTA_LINK_ADDRESS for address in memory):
        raise ValueError("引导固件超出 0x08000000..0x08007FFF 保留区")
    if not application_memory:
        raise ValueError("应用 HEX 为空")
    if any(
        not OTA_LINK_ADDRESS <= address < OTA_APPLICATION_END
        for address in application_memory
    ):
        raise ValueError("应用固件超出 0x08008000..0x0803FFFF 应用区")
    _validate_vectors(memory, BOOTLOADER_START, OTA_LINK_ADDRESS, "引导固件")
    _validate_vectors(
        application_memory, OTA_LINK_ADDRESS, OTA_APPLICATION_END, "应用固件"
    )
    for address, value in application_memory.items():
        if address in memory and memory[address] != value:
            raise ValueError(f"引导区与应用区在 0x{address:08X} 重叠")
        memory[address] = value
    write_hex(memory, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成首次有线安装 OTA HEX")
    parser.add_argument("bootloader", type=Path)
    parser.add_argument("application", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    merge(args.bootloader, args.application, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
