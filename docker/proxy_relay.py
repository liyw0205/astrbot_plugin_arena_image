"""Small unauthenticated-to-authenticated HTTP CONNECT proxy relay.

Chromium rejects proxy URLs that contain ``user:password@`` credentials
(``ERR_NO_SUPPORTED_PROXIES``).  This relay keeps credentials server-side and
adds Proxy-Authorization when talking to the configured upstream proxy.
"""

from __future__ import annotations

import base64
import os
import select
import socket
import threading
from urllib.parse import unquote, urlsplit


def _upstream() -> tuple[str, int, str]:
    raw = os.environ.get("LM_BRIDGE_PROXY_URL", "").strip()
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise SystemExit("proxy relay requires an http(s) LM_BRIDGE_PROXY_URL")
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    auth = ""
    if parts.username is not None:
        token = f"{unquote(parts.username)}:{unquote(parts.password or '')}".encode()
        auth = "Basic " + base64.b64encode(token).decode("ascii")
    return host, port, auth


UPSTREAM_HOST, UPSTREAM_PORT, PROXY_AUTH = _upstream()


def _pipe(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 120)
            if not readable:
                continue
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                (right if source is left else left).sendall(data)
    except OSError:
        return


def _headers(raw: bytes) -> tuple[str, dict[str, str], bytes]:
    head, remainder = raw.split(b"\r\n\r\n", 1)
    lines = head.decode("latin-1").split("\r\n")
    first = lines[0]
    values: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.lower()] = value.strip()
    return first, values, remainder


def handle(client: socket.socket) -> None:
    upstream = None
    try:
        client.settimeout(10)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = client.recv(4096)
            if not chunk:
                return
            data += chunk
        if b"\r\n\r\n" not in data:
            return
        first, headers, remainder = _headers(data)
        fields = first.split()
        if len(fields) < 2:
            return
        upstream = socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), 10)
        if fields[0].upper() == "CONNECT":
            target = fields[1]
            target_lower = target.lower()
            target_host = target_lower.rsplit(":", 1)[0] if ":" in target_lower else target_lower
            upstream_target = target
            if target_lower == "www.google.com:443":
                # The upstream proxy returns an empty reCAPTCHA bootstrap.
                upstream_target = "www.recaptcha.net:443"
            # The upstream proxy resets direct CONNECTs to Google's regional
            # OAuth hosts. oauth2.googleapis.com serves the same TLS endpoint;
            # the browser-side tunnel is untouched, so its original SNI and
            # Host remain accounts.google.*.
            if target_host.startswith("accounts.google.") and target_lower.endswith(":443"):
                upstream_target = "oauth2.googleapis.com:443"
            request = f"CONNECT {upstream_target} HTTP/1.1\r\nHost: {upstream_target}\r\n"
            if PROXY_AUTH:
                request += f"Proxy-Authorization: {PROXY_AUTH}\r\n"
            upstream.sendall((request + "\r\n").encode("latin-1"))
            response = b""
            while b"\r\n\r\n" not in response and len(response) < 65536:
                chunk = upstream.recv(4096)
                if not chunk:
                    return
                response += chunk
            status = response.split(b"\r\n", 1)[0]
            if b" 200 " not in status:
                client.sendall(response)
                return
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(None)
            upstream.settimeout(None)
            # Preserve only bytes after the complete upstream header block.
            # The previous implementation accidentally forwarded the blank
            # line itself, which made Chromium report ERR_TUNNEL_CONNECTION_FAILED.
            marker = response.find(b"\r\n\r\n")
            remainder = response[marker + 4 :] if marker >= 0 else b""
            if remainder:
                client.sendall(remainder)
            _pipe(client, upstream)
            return

        # Fallback for plain HTTP requests: forward absolute-form requests.
        lines = [first]
        for key, value in headers.items():
            if key in {"proxy-connection", "proxy-authorization", "connection"}:
                continue
            lines.append(f"{key}: {value}")
        if PROXY_AUTH:
            lines.append(f"Proxy-Authorization: {PROXY_AUTH}")
        upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + remainder)
        while True:
            chunk = upstream.recv(65536)
            if not chunk:
                break
            client.sendall(chunk)
    except OSError:
        return
    finally:
        for sock in (client, upstream):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass


def main() -> None:
    with socket.create_server(("127.0.0.1", 18080), reuse_port=True) as server:
        while True:
            client, _ = server.accept()
            threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
