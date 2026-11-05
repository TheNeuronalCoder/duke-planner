"""
Forwards a local port to the production Postgres over SSH, so a local
`runserver` or `shell` can use the real database.

Uses the system `ssh` client with key-based auth: an explicit identity file
if given, otherwise your ~/.ssh keys / agent.
"""

import atexit
import os
import socket
import subprocess
import time


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def open_tunnel(ssh_host, ssh_port, ssh_user, remote_host, remote_port, local_port, key_file=None, timeout=15):
    """Returns the local (host, port) to connect to in place of the remote database."""
    # runserver's autoreloader loads settings in both the watcher and the
    # server process, so the second one reuses the tunnel the first opened.
    if _port_open(local_port):
        return '127.0.0.1', local_port

    key_args = []
    if key_file:
        key_args = ['-i', os.path.expanduser(key_file), '-o', 'IdentitiesOnly=yes']

    proc = subprocess.Popen(
        [
            'ssh', '-N',
            '-L', f'127.0.0.1:{local_port}:{remote_host}:{remote_port}',
            '-p', str(ssh_port),
            *key_args,
            '-o', 'BatchMode=yes',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=30',
            f'{ssh_user}@{ssh_host}',
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    atexit.register(proc.terminate)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            error = proc.stderr.read().decode().strip()
            raise RuntimeError(f'SSH tunnel to {ssh_user}@{ssh_host} failed: {error}')
        if _port_open(local_port):
            return '127.0.0.1', local_port
        time.sleep(0.2)

    proc.terminate()
    raise RuntimeError(f'SSH tunnel to {ssh_user}@{ssh_host} timed out after {timeout}s')
