"""Guest side of a karotte Firecracker VM, written into the base drive at
/.karotte/agent.py and run by the init with the image's Python.

Standard library only: it runs with ``-I -S`` before karotte starts.

- ``lo-up``: bring up loopback (the kernel's ``ip=`` does it only when the VM
  has a network interface).
- ``serve``: answer the host's liveness probe on a vsock port, and relay the
  host's vsock connections on the websocket port to karotte on 127.0.0.1.
  Only connections from the host (CID 2) are served, so a guest process
  can't use the relay to reach a port the firewall blocks.
- ``poweroff``: reboot(2), which with ``reboot=k`` makes Firecracker exit.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import os
import socket
import struct
import sys
import threading

# vsock goes through libc on raw fds: Python builds without
# <linux/vm_sockets.h> (python-build-standalone among them) can't bind or
# accept AF_VSOCK sockets.
AF_VSOCK = 40
VMADDR_CID_ANY = 0xFFFFFFFF
VMADDR_CID_HOST = 2
SOCK_CLOEXEC = 0o2000000
SHUT_RDWR = 2
LINUX_REBOOT_CMD_RESTART = 0x01234567
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1


def lo_up() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        req = struct.pack("16sh", b"lo", 0)
        flags = struct.unpack("16sh", fcntl.ioctl(s, SIOCGIFFLAGS, req))[1]
        fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sh", b"lo", flags | IFF_UP))


def poweroff() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sync()
    libc.reboot(LINUX_REBOOT_CMD_RESTART)
    raise OSError(ctypes.get_errno(), "reboot failed")


_libc = ctypes.CDLL(None, use_errno=True)


def _check(result: int, what: str) -> int:
    if result < 0:
        raise OSError(ctypes.get_errno(), what)
    return result


def sockaddr_vm(port: int, cid: int = VMADDR_CID_ANY) -> bytes:
    """struct sockaddr_vm: family, reserved, port, cid, flags, padding."""
    return struct.pack("=HHIIB3x", AF_VSOCK, 0, port, cid, 0)


def peer_cid(address: bytes) -> int:
    return struct.unpack_from("=I", address, 8)[0]


def _listen(port: int) -> int:
    fd = _check(_libc.socket(AF_VSOCK, socket.SOCK_STREAM | SOCK_CLOEXEC, 0), "socket")
    addr = sockaddr_vm(port)
    _check(_libc.bind(fd, addr, len(addr)), f"bind vsock port {port}")
    _check(_libc.listen(fd, 128), "listen")
    return fd


def _accept_from_host(server: int):
    """Connections from the host only; the rest are closed."""
    while True:
        buf = ctypes.create_string_buffer(16)
        size = ctypes.c_uint32(16)
        conn = _check(
            _libc.accept4(server, buf, ctypes.byref(size), SOCK_CLOEXEC), "accept"
        )
        if peer_cid(buf.raw) == VMADDR_CID_HOST:
            yield conn
        else:
            os.close(conn)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _pipe(src: int, dst: int) -> None:
    try:
        while chunk := os.read(src, 65536):
            _write_all(dst, chunk)
    except OSError:
        pass
    finally:
        for fd in (src, dst):
            _libc.shutdown(fd, SHUT_RDWR)


def _heartbeat(port: int) -> None:
    for conn in _accept_from_host(_listen(port)):
        try:
            _write_all(conn, b"ok\n")
        except OSError:
            pass
        finally:
            os.close(conn)


def _relay_one(conn: int, port: int) -> None:
    try:
        upstream = socket.create_connection(("127.0.0.1", port), timeout=5)
    except OSError:
        os.close(conn)
        return
    upstream.settimeout(None)
    with upstream:
        back = threading.Thread(
            target=_pipe, args=(upstream.fileno(), conn), daemon=True
        )
        back.start()
        _pipe(conn, upstream.fileno())
        back.join()
    os.close(conn)


def _relay(port: int) -> None:
    for conn in _accept_from_host(_listen(port)):
        threading.Thread(target=_relay_one, args=(conn, port), daemon=True).start()


def serve(heartbeat_port: int, relay_port: int | None) -> None:
    threads = [threading.Thread(target=_heartbeat, args=(heartbeat_port,))]
    if relay_port is not None:
        threads.append(threading.Thread(target=_relay, args=(relay_port,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="agent.py")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lo-up")
    sub.add_parser("poweroff")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--heartbeat-port", type=int, required=True)
    serve_parser.add_argument("--relay-port", type=int)
    args = parser.parse_args(argv)
    if args.command == "lo-up":
        lo_up()
    elif args.command == "poweroff":
        poweroff()
    else:
        serve(args.heartbeat_port, args.relay_port)


if __name__ == "__main__":
    main(sys.argv[1:])
