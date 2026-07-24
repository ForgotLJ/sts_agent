from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Callable
from zipfile import ZipFile


TARGET_CLASS = "communicationmod/GameStateConverter.class"
TARGET_METHOD = "convertMapRoomNodeToJson"


@dataclass(frozen=True, slots=True)
class ConstantPool:
    count: int
    end: int
    entries: tuple[tuple[int, tuple[int | str, ...]] | None, ...]

    def utf8(self, index: int) -> str:
        entry = self.entries[index]
        if entry is None or entry[0] != 1:
            raise ValueError(f"constant #{index} is not UTF-8")
        return str(entry[1][0])

    def find_utf8(self, value: str) -> int:
        for index, entry in enumerate(self.entries):
            if entry is not None and entry[0] == 1 and entry[1][0] == value:
                return index
        raise ValueError(f"UTF-8 constant is missing: {value}")

    def find_class(self, value: str) -> int:
        for index, entry in enumerate(self.entries):
            if entry is not None and entry[0] == 7 and self.utf8(int(entry[1][0])) == value:
                return index
        raise ValueError(f"class constant is missing: {value}")

    def find_member(self, tag: int, owner: str, name: str, descriptor: str) -> int:
        for index, entry in enumerate(self.entries):
            if entry is None or entry[0] != tag:
                continue
            class_index, name_type_index = (int(value) for value in entry[1])
            if self.utf8(int(self.entries[class_index][1][0])) != owner:
                continue
            name_type = self.entries[name_type_index]
            if name_type is None or name_type[0] != 12:
                continue
            name_index, descriptor_index = (int(value) for value in name_type[1])
            if self.utf8(name_index) == name and self.utf8(descriptor_index) == descriptor:
                return index
        raise ValueError(f"member constant is missing: {owner}.{name}:{descriptor}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add burning_elite map state to a CommunicationMod 1.2.1 jar."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def u1(data: bytes, offset: int) -> tuple[int, int]:
    return data[offset], offset + 1


def u2(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def u4(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def pack_u2(value: int) -> bytes:
    return struct.pack(">H", value)


def pack_u4(value: int) -> bytes:
    return struct.pack(">I", value)


def parse_constant_pool(data: bytes) -> ConstantPool:
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("target is not a Java class file")
    count, offset = u2(data, 8)
    entries: list[tuple[int, tuple[int | str, ...]] | None] = [None] * count
    index = 1
    while index < count:
        tag, offset = u1(data, offset)
        if tag == 1:
            length, offset = u2(data, offset)
            value = data[offset : offset + length].decode("utf-8")
            entries[index] = (tag, (value,))
            offset += length
        elif tag in {3, 4}:
            entries[index] = (tag, (int.from_bytes(data[offset : offset + 4], "big"),))
            offset += 4
        elif tag in {5, 6}:
            entries[index] = (tag, (int.from_bytes(data[offset : offset + 8], "big"),))
            offset += 8
            index += 1
        elif tag in {7, 8, 16, 19, 20}:
            value, offset = u2(data, offset)
            entries[index] = (tag, (value,))
        elif tag in {9, 10, 11, 12, 18}:
            first, offset = u2(data, offset)
            second, offset = u2(data, offset)
            entries[index] = (tag, (first, second))
        elif tag == 15:
            first, offset = u1(data, offset)
            second, offset = u2(data, offset)
            entries[index] = (tag, (first, second))
        else:
            raise ValueError(f"unsupported constant-pool tag: {tag}")
        index += 1
    return ConstantPool(count=count, end=offset, entries=tuple(entries))


def skip_attributes(data: bytes, offset: int) -> int:
    count, offset = u2(data, offset)
    for _ in range(count):
        _, offset = u2(data, offset)
        length, offset = u4(data, offset)
        offset += length
    return offset


def locate_code_attribute(data: bytes, pool: ConstantPool) -> tuple[int, int]:
    offset = pool.end + 6
    interface_count, offset = u2(data, offset)
    offset += interface_count * 2
    field_count, offset = u2(data, offset)
    for _ in range(field_count):
        offset += 6
        offset = skip_attributes(data, offset)
    method_count, offset = u2(data, offset)
    for _ in range(method_count):
        _, offset = u2(data, offset)
        name_index, offset = u2(data, offset)
        _, offset = u2(data, offset)
        attribute_count, offset = u2(data, offset)
        method_name = pool.utf8(name_index)
        for _ in range(attribute_count):
            attribute_start = offset
            attribute_name_index, offset = u2(data, offset)
            length, offset = u4(data, offset)
            attribute_end = offset + length
            if method_name == TARGET_METHOD and pool.utf8(attribute_name_index) == "Code":
                return attribute_start, attribute_end
            offset = attribute_end
    raise ValueError(f"method code was not found: {TARGET_METHOD}")


def append_constants(pool: ConstantPool) -> tuple[bytes, dict[str, int]]:
    burning_utf8 = pool.count
    burning_string = burning_utf8 + 1
    field_name_utf8 = burning_utf8 + 2
    field_name_type = burning_utf8 + 3
    field_reference = burning_utf8 + 4
    descriptor_index = pool.find_utf8("Z")
    map_node_class = pool.find_class("com/megacrit/cardcrawl/map/MapRoomNode")
    encoded = b"".join(
        (
            b"\x01" + pack_u2(len("burning_elite")) + b"burning_elite",
            b"\x08" + pack_u2(burning_utf8),
            b"\x01" + pack_u2(len("hasEmeraldKey")) + b"hasEmeraldKey",
            b"\x0c" + pack_u2(field_name_utf8) + pack_u2(descriptor_index),
            b"\x09" + pack_u2(map_node_class) + pack_u2(field_name_type),
        )
    )
    return encoded, {
        "count": pool.count + 5,
        "burning_string": burning_string,
        "field_reference": field_reference,
    }


def shift_local_variables(payload: bytes, insertion: int, delta: int) -> bytes:
    count, offset = u2(payload, 0)
    result = bytearray(payload[:2])
    for _ in range(count):
        start, offset = u2(payload, offset)
        length, offset = u2(payload, offset)
        tail = payload[offset : offset + 6]
        offset += 6
        if start >= insertion:
            start += delta
        elif start + length > insertion:
            length += delta
        result.extend(pack_u2(start))
        result.extend(pack_u2(length))
        result.extend(tail)
    return bytes(result)


def shift_line_numbers(payload: bytes, insertion: int, delta: int) -> bytes:
    count, offset = u2(payload, 0)
    result = bytearray(payload[:2])
    for _ in range(count):
        start, offset = u2(payload, offset)
        line, offset = u2(payload, offset)
        if start >= insertion:
            start += delta
        result.extend(pack_u2(start))
        result.extend(pack_u2(line))
    return bytes(result)


def rebuild_code_attribute(
    attribute: bytes,
    pool: ConstantPool,
    burning_string: int,
    field_reference: int,
) -> bytes:
    name_index = attribute[:2]
    info = attribute[6:]
    max_stack, offset = u2(info, 0)
    max_locals, offset = u2(info, offset)
    code_length, offset = u4(info, offset)
    code = info[offset : offset + code_length]
    offset += code_length
    if code[-2:] != b"\x2b\xb0":
        raise ValueError("unexpected target method return sequence")
    boolean_value_of = pool.find_member(
        10,
        "java/lang/Boolean",
        "valueOf",
        "(Z)Ljava/lang/Boolean;",
    )
    hash_map_put = pool.find_member(
        10,
        "java/util/HashMap",
        "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
    )
    inserted = b"".join(
        (
            b"\x2b\x13" + pack_u2(burning_string),
            b"\x2a\xb4" + pack_u2(field_reference),
            b"\xb8" + pack_u2(boolean_value_of),
            b"\xb6" + pack_u2(hash_map_put),
            b"\x57",
        )
    )
    insertion = len(code) - 2
    code = code[:insertion] + inserted + code[insertion:]
    delta = len(inserted)
    exception_count, offset = u2(info, offset)
    exception_table = bytearray()
    for _ in range(exception_count):
        start, offset = u2(info, offset)
        end, offset = u2(info, offset)
        handler, offset = u2(info, offset)
        catch_type, offset = u2(info, offset)
        if start >= insertion:
            start += delta
        if end > insertion:
            end += delta
        if handler >= insertion:
            handler += delta
        exception_table.extend(struct.pack(">HHHH", start, end, handler, catch_type))
    nested_count, offset = u2(info, offset)
    nested = bytearray()
    transformers: dict[str, Callable[[bytes, int, int], bytes]] = {
        "LocalVariableTable": shift_local_variables,
        "LocalVariableTypeTable": shift_local_variables,
        "LineNumberTable": shift_line_numbers,
    }
    for _ in range(nested_count):
        nested_name_index, offset = u2(info, offset)
        nested_length, offset = u4(info, offset)
        payload = info[offset : offset + nested_length]
        offset += nested_length
        name = pool.utf8(nested_name_index)
        if name == "StackMapTable":
            raise ValueError("target method unexpectedly contains a StackMapTable")
        transformer = transformers.get(name)
        if transformer is not None:
            payload = transformer(payload, insertion, delta)
        nested.extend(pack_u2(nested_name_index))
        nested.extend(pack_u4(len(payload)))
        nested.extend(payload)
    rebuilt_info = b"".join(
        (
            pack_u2(max(max_stack, 3)),
            pack_u2(max_locals),
            pack_u4(len(code)),
            code,
            pack_u2(exception_count),
            bytes(exception_table),
            pack_u2(nested_count),
            bytes(nested),
        )
    )
    return name_index + pack_u4(len(rebuilt_info)) + rebuilt_info


def patch_class(data: bytes) -> bytes:
    if b"burning_elite" in data:
        return data
    pool = parse_constant_pool(data)
    attribute_start, attribute_end = locate_code_attribute(data, pool)
    constants, indices = append_constants(pool)
    rebuilt_attribute = rebuild_code_attribute(
        data[attribute_start:attribute_end],
        pool,
        indices["burning_string"],
        indices["field_reference"],
    )
    return b"".join(
        (
            data[:8],
            pack_u2(indices["count"]),
            data[10 : pool.end],
            constants,
            data[pool.end:attribute_start],
            rebuilt_attribute,
            data[attribute_end:],
        )
    )


def patch_jar(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(input_path) as source, ZipFile(output_path, "w") as target:
        names = set(source.namelist())
        if TARGET_CLASS not in names:
            raise ValueError(f"jar lacks {TARGET_CLASS}")
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == TARGET_CLASS:
                payload = patch_class(payload)
            target.writestr(info, payload)
    with ZipFile(output_path) as patched:
        payload = patched.read(TARGET_CLASS)
        if b"burning_elite" not in payload or b"hasEmeraldKey" not in payload:
            raise RuntimeError("patched CommunicationMod jar failed verification")


def main() -> int:
    args = parse_args()
    patch_jar(args.input.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
