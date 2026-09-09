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
import ssl
import threading
from urllib.parse import unquote, urlsplit


def _upstream() -> tuple[str, int, str, bool]:
    raw = (os.environ.get("LM_BRIDGE_PROXY_URL") or os.environ.get("HTTP_PROXY") or "").strip()
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise SystemExit("proxy relay requires an http(s) LM_BRIDGE_PROXY_URL or HTTP_PROXY")
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    auth = ""
    if parts.username is not None:
        token = f"{unquote(parts.username)}:{unquote(parts.password or '')}".encode()
        auth = "Basic " + base64.b64encode(token).decode("ascii")
    return host, port, auth, parts.scheme == "https"


UPSTREAM_HOST, UPSTREAM_PORT, PROXY_AUTH, UPSTREAM_TLS = _upstream()
GOOGLE_WORKAROUNDS = os.environ.get("LM_BRIDGE_PROXY_GOOGLE_WORKAROUNDS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _connect_upstream() -> socket.socket:
    connection = socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), 10)
    if not UPSTREAM_TLS:
        return connection
    try:
        # Authenticate the HTTPS proxy before sending its Basic credentials.
        # The default context also honours SSL_CERT_FILE for a private proxy CA.
        return ssl.create_default_context().wrap_socket(connection, server_hostname=UPSTREAM_HOST)
    except Exception:
        connection.close()
        raise


def _pipe(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        while True:
            # TLS may already have decrypted bytes buffered even when the
            # underlying socket is not readable anymore.
            readable = [
                source
                for source in sockets
                if isinstance(source, ssl.SSLSocket) and source.pending()
            ]
            if not readable:
                readable, _, _ = select.select(sockets, [], [], 120)
            if not readable:
                return
            for source in readable:
                data = source.recv(65536)
                if not data:
                    if source is right:
                        return
                    # A request sender may half-close after its POST body and
                    # still expect the response. Keep reading the upstream.
                    sockets.remove(source)
                    if not isinstance(right, ssl.SSLSocket):
                        right.shutdown(socket.SHUT_WR)
                    continue
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
        upstream = _connect_upstream()
        if fields[0].upper() == "CONNECT":
            target = fields[1]
            target_lower = target.lower()
            target_host = target_lower.rsplit(":", 1)[0] if ":" in target_lower else target_lower
            upstream_target = target
            if GOOGLE_WORKAROUNDS and target_lower == "www.google.com:443":
                # The upstream proxy returns an empty reCAPTCHA bootstrap.
                upstream_target = "www.recaptcha.net:443"
            # The upstream proxy resets direct CONNECTs to Google's regional
            # OAuth hosts. oauth2.googleapis.com serves the same TLS endpoint;
            # the browser-side tunnel is untouched, so its original SNI and
            # Host remain accounts.google.*.
            if (
                GOOGLE_WORKAROUNDS
                and target_host.startswith("accounts.google.")
                and target_lower.endswith(":443")
            ):
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
            if b"\r\n\r\n" not in response:
                return
            if status.split()[1:2] != [b"200"]:
                client.sendall(response)
                return
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(120)
            upstream.settimeout(120)
            # A CONNECT client may pipeline its first tunnel bytes with the
            # headers. They belong upstream, not to the proxy response.
            if remainder:
                upstream.sendall(remainder)
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
        # Authenticate each HTTP request on a fresh upstream connection. A raw
        # persistent pipe would forward later requests without injecting auth.
        lines.extend(("Connection: close", "Proxy-Connection: close"))
        upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + remainder)
        client.settimeout(120)
        upstream.settimeout(120)
        # Read both peers: a POST body need not arrive with its headers, and
        # Expect: 100-continue requires an upstream response before that body.
        _pipe(client, upstream)
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
    with socket.create_server(("127.0.0.1", 18080)) as server:
        while True:
            client, _ = server.accept()
            threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
