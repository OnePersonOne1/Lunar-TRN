"""Unity 렌더 서버 TCP 클라이언트: pose·태양각 전송 → 프레임(numpy BGR) 수신.

프로토콜: 요청 = JSON 한 줄, 응답 = 4바이트 big-endian 길이 + PNG(또는 JSON {"error": ...}).
"""
from __future__ import annotations

import json
import socket
import struct

import cv2
import numpy as np

RETRIES = 3


class UnityRenderError(RuntimeError):
    pass


class RenderClient:
    def __init__(self, cfg: dict) -> None:
        u = cfg["unity"]
        self.host = str(u["host"])
        self.port = int(u["port"])
        self.timeout = float(u["timeout_s"])
        self._sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
            s.settimeout(self.timeout)
            self._sock = s
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _recv_exact(self, n: int) -> bytes:
        s = self._connect()
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise UnityRenderError("서버가 연결을 닫았다")
            buf += chunk
        return buf

    def render(
        self, r_L: np.ndarray, sun_az_deg: float, sun_el_deg: float, frame_id: int = 0, t: float = 0.0
    ) -> np.ndarray:
        """한 프레임 렌더. 반환: (H, W, 3) BGR uint8. 실패 시 재시도 후 UnityRenderError."""
        req = json.dumps({
            "frame_id": int(frame_id), "t": float(t),
            "r_L": [float(v) for v in r_L],
            "sun_az_deg": float(sun_az_deg), "sun_el_deg": float(sun_el_deg),
        }) + "\n"
        last_exc: Exception | None = None
        for _ in range(RETRIES):
            try:
                s = self._connect()
                s.sendall(req.encode("utf-8"))
                (length,) = struct.unpack(">I", self._recv_exact(4))
                payload = self._recv_exact(length)
                if payload[:4] == b"\x89PNG":
                    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise UnityRenderError("PNG 디코드 실패")
                    return img
                raise UnityRenderError(json.loads(payload.decode("utf-8")).get("error", "unknown"))
            except (OSError, struct.error) as exc:
                last_exc = exc
                self.close()
        raise UnityRenderError(f"렌더 실패 (재시도 {RETRIES}회): {last_exc}")
