"""Length-delimited JSON messages over a worker pipe.

Each message is one UTF-8 JSON object on its own line, bounded in size. The
protocol carries no pickled objects and requires no third-party dependency in
the worker process.
"""

from __future__ import annotations

import json

MAX_MESSAGE_BYTES = 1_000_000
PROTOCOL_VERSION = 1


class ProtocolError(Exception):
    pass


def encode(message):
    if not isinstance(message, dict):
        raise ProtocolError("消息必须是对象")
    payload = dict(message)
    payload.setdefault("protocol", PROTOCOL_VERSION)
    line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("消息超过大小上限")
    return line


def decode(line):
    if not line:
        raise ProtocolError("空消息")
    if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("消息超过大小上限")
    try:
        message = json.loads(line)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"无法解析消息：{error.msg}") from error
    if not isinstance(message, dict):
        raise ProtocolError("消息必须是对象")
    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("协议版本不匹配")
    return message


def validate_command(command, command_id, args=None):
    if command not in {"resume", "pause", "step", "jog", "jog_start", "jog_stop", "edit", "reset", "stop"}:
        raise ProtocolError("未知命令")
    if not isinstance(command_id, str) or not 1 <= len(command_id) <= 100:
        raise ProtocolError("command_id 无效")
    if args is not None and not isinstance(args, dict):
        raise ProtocolError("args 必须是对象")
    if command not in {"jog", "jog_start", "jog_stop", "edit"} and args:
        raise ProtocolError("只有 jog 系列命令接受参数")
    return command, command_id
