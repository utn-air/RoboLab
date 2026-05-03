from __future__ import annotations

import time
import pickle
import socket
import struct
import traceback
from pathlib import Path

from .base_client import InferenceClient

################################ CLIENT ############################################

_HEADER = struct.Struct("!Q")

def _send_message(sock: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("VALP policy socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))

class VALPDroidEEClient(InferenceClient):
    """Local RoboLab inference client for the VALP world model on DroidIK envs."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
    ) -> None:
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.sock = self._connect()

    def _connect(self) -> socket.socket:
        print(f"[{self.__class__.__name__}] Waiting for VALP server on {self.remote_host}:{self.remote_port}...")
        while True:
            try:
                sock = socket.create_connection((self.remote_host, self.remote_port), timeout=10)
                sock.settimeout(None)
                print(f"[{self.__class__.__name__}] Connected to VALP server.")
                return sock
            except OSError:
                time.sleep(2)

    def _request(self, payload: dict) -> dict:
        _send_message(self.sock, payload)
        response = _recv_message(self.sock)
        if error := response.get("error"):
            raise RuntimeError(f"VALP server error:\n{error}")
        return response

    def reset(self):
        self._request({"method": "reset"})

    def set_goal_images(self, external_image, wrist_image, *, env_id: int = 0, instruction: str = "goal"):
        # SEND EXACT PARAMS PASSED TO THIS FUNCTION TO SERVER
        self._request(
            {
                "method": "set_goal_images",
                "external_image": external_image,
                "wrist_image": wrist_image,
                "env_id": env_id,
                "instruction": instruction
            }
        )

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        # SEND EXACT PARAMS PASSED TO THIS FUNCTION TO SERVER 
        
        return self._request(
            {
                "method": "infer",
                "obs": obs,
                "instruction": instruction,
                "env_id": env_id,
            }
        )

MyPolicyClient = VALPDroidEEClient

