"""Loopback-only regression tests for Chromium's authenticated proxy relay."""

from __future__ import annotations

import base64
import importlib.util
import ipaddress
import os
import socket
import ssl
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


RELAY = Path(__file__).resolve().parents[1] / "proxy_relay.py"
AUTH = b"Basic " + base64.b64encode(b"fixture-user:fixture-pass")


def load_relay(env):
    spec = importlib.util.spec_from_file_location("fixture_relay", RELAY)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(module)
    return module


def receive_headers(connection):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def receive_body(connection, size, initial=b""):
    data = initial
    while len(data) < size:
        chunk = connection.recv(min(65536, size - len(data)))
        if not chunk:
            break
        data += chunk
    return data


def exchange(peer, browser, *, tls_context=None, extra_env=None, allow_tls_failure=False):
    errors = []
    peer_results = []
    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(5)
        scheme = "https" if tls_context else "http"
        env = {
            "LM_BRIDGE_PROXY_URL": (
                f"{scheme}://fixture-user:fixture-pass@127.0.0.1:{listener.getsockname()[1]}"
            ),
            **(extra_env or {}),
        }
        relay = load_relay(env)

        def serve():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    if tls_context:
                        connection = tls_context.wrap_socket(connection, server_side=True)
                    with connection:
                        peer_results.append(peer(connection))
            except Exception as exc:
                errors.append(exc)

        with patch.dict(os.environ, env, clear=True):
            peer_thread = threading.Thread(target=serve, daemon=True)
            peer_thread.start()
            client, relay_side = socket.socketpair()
            with client, relay_side:
                client.settimeout(5)
                relay_thread = threading.Thread(
                    target=relay.handle, args=(relay_side,), daemon=True
                )
                relay_thread.start()
                try:
                    result = browser(client)
                finally:
                    client.close()
                    relay_thread.join(6)
                    peer_thread.join(6)
            assert not relay_thread.is_alive()
            assert not peer_thread.is_alive()
    if allow_tls_failure:
        # Windows may deliver a TCP reset rather than the TLS alert after the
        # verifying client closes. In both cases no authenticated HTTP arrived.
        assert errors and all(
            isinstance(exc, (ssl.SSLError, ConnectionResetError)) for exc in errors
        )
    else:
        assert not errors, errors
    return result, peer_results


def test_http_proxy_fallback_and_lm_bridge_precedence():
    fallback = load_relay({"HTTP_PROXY": "http://fixture-user:fixture-pass@127.0.0.1:8001"})
    assert (fallback.UPSTREAM_HOST, fallback.UPSTREAM_PORT) == ("127.0.0.1", 8001)
    assert fallback.PROXY_AUTH.encode() == AUTH
    explicit = load_relay(
        {
            "HTTP_PROXY": "http://127.0.0.1:8001",
            "LM_BRIDGE_PROXY_URL": "https://fixture-user:fixture-pass@localhost:8002",
        }
    )
    assert (explicit.UPSTREAM_HOST, explicit.UPSTREAM_PORT, explicit.UPSTREAM_TLS) == (
        "localhost",
        8002,
        True,
    )


def test_credentials_are_url_decoded_before_basic_auth():
    relay = load_relay({"LM_BRIDGE_PROXY_URL": "http://user%40fixture:pass%3Aword@localhost"})
    assert relay.PROXY_AUTH == "Basic " + base64.b64encode(b"user@fixture:pass:word").decode()


@pytest.mark.parametrize(
    "target,enabled,expected",
    [
        ("www.google.com:443", False, "www.google.com:443"),
        ("accounts.google.co.jp:443", False, "accounts.google.co.jp:443"),
        ("www.google.com:443", True, "www.recaptcha.net:443"),
        ("accounts.google.co.jp:443", True, "oauth2.googleapis.com:443"),
        ("target.invalid:443", True, "target.invalid:443"),
    ],
)
def test_connect_routing_is_unchanged_unless_explicitly_enabled(target, enabled, expected):
    def peer(connection):
        request = receive_headers(connection)
        assert request.startswith(f"CONNECT {expected} HTTP/1.1\r\n".encode())
        assert b"Proxy-Authorization: " + AUTH in request
        connection.sendall(b"HTTP/1.1 200\r\n\r\n")
        assert receive_body(connection, 4) == b"ping"
        connection.sendall(b"pong")

    def browser(connection):
        connection.sendall(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
        assert b" 200 " in receive_headers(connection)
        connection.sendall(b"ping")
        assert receive_body(connection, 4) == b"pong"

    exchange(peer, browser, extra_env={"LM_BRIDGE_PROXY_GOOGLE_WORKAROUNDS": str(int(enabled))})


def test_connect_preserves_pipelined_client_bytes_and_upstream_remainder():
    def peer(connection):
        request = receive_headers(connection)
        initial = request.split(b"\r\n\r\n", 1)[1]
        connection.sendall(b"HTTP/1.1 200 OK\r\n\r\nserver-first")
        assert receive_body(connection, 4, initial) == b"ping"
        connection.sendall(b"pong")

    def browser(connection):
        connection.sendall(b"CONNECT target.invalid:443 HTTP/1.1\r\n\r\nping")
        response = receive_headers(connection)
        initial = response.split(b"\r\n\r\n", 1)[1]
        assert receive_body(connection, len(b"server-firstpong"), initial) == b"server-firstpong"

    exchange(peer, browser)


@pytest.mark.parametrize("chunked", [False, True])
def test_plain_http_streams_body_after_headers(chunked):
    saw_headers = threading.Event()
    body = (b"10000\r\n" + b"x" * 65536 + b"\r\n0\r\n\r\n") if chunked else b"x" * 131072

    def peer(connection):
        request = receive_headers(connection)
        head, initial = request.split(b"\r\n\r\n", 1)
        assert b"Proxy-Authorization: " + AUTH in head
        assert b"Connection: close" in head
        assert b"browser-supplied-key" not in head
        saw_headers.set()
        assert receive_body(connection, len(body), initial) == body
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")

    def browser(connection):
        framing = "Transfer-Encoding: chunked" if chunked else f"Content-Length: {len(body)}"
        connection.sendall(
            (
                "POST http://target.invalid/upload HTTP/1.1\r\nHost: target.invalid\r\n"
                "Proxy-Authorization: browser-supplied-key\r\n"
                f"{framing}\r\n\r\n"
            ).encode()
        )
        assert saw_headers.wait(5)
        connection.sendall(body)
        assert b"200 OK" in receive_headers(connection)

    exchange(peer, browser)


def test_expect_continue_and_client_half_close_keep_the_response():
    body = b"fixture-body"

    def peer(connection):
        request = receive_headers(connection)
        connection.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
        initial = request.split(b"\r\n\r\n", 1)[1]
        assert receive_body(connection, len(body), initial) == body
        assert connection.recv(1) == b""
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")

    def browser(connection):
        connection.sendall(
            (
                "POST http://target.invalid/upload HTTP/1.1\r\nHost: target.invalid\r\n"
                f"Content-Length: {len(body)}\r\nExpect: 100-continue\r\n\r\n"
            ).encode()
        )
        assert b"100 Continue" in receive_headers(connection)
        connection.sendall(body)
        connection.shutdown(socket.SHUT_WR)
        assert b"200 OK" in receive_headers(connection)

    exchange(peer, browser)


def test_connect_forwards_proxy_authentication_failure():
    def peer(connection):
        receive_headers(connection)
        connection.sendall(
            b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n"
        )

    def browser(connection):
        connection.sendall(b"CONNECT target.invalid:443 HTTP/1.1\r\n\r\n")
        assert b"407 Proxy Authentication Required" in receive_headers(connection)

    exchange(peer, browser)


@pytest.fixture
def tls_proxy(tmp_path):
    # Ephemeral test-only keys; no private key is stored in the repository.
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "proxy.crt", tmp_path / "proxy.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context, cert_path


def test_https_proxy_uses_verified_tls_and_drains_buffered_records(tls_proxy):
    context, cert_path = tls_proxy
    body = b"x" * (256 * 1024)

    def peer(connection):
        request = receive_headers(connection)
        assert b"Proxy-Authorization: " + AUTH in request
        connection.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
        assert receive_body(connection, len(body)) == body
        connection.sendall(body)

    def browser(connection):
        connection.sendall(b"CONNECT target.invalid:443 HTTP/1.1\r\n\r\n")
        assert b"200 Connection Established" in receive_headers(connection)
        connection.sendall(body)
        assert receive_body(connection, len(body)) == body

    exchange(peer, browser, tls_context=context, extra_env={"SSL_CERT_FILE": str(cert_path)})


def test_https_proxy_rejects_untrusted_certificate_before_sending_auth(tls_proxy):
    context, _ = tls_proxy

    def peer(connection):
        raise AssertionError("Untrusted TLS peer must not receive authenticated HTTP")

    def browser(connection):
        connection.sendall(b"CONNECT target.invalid:443 HTTP/1.1\r\n\r\n")
        assert connection.recv(4096) == b""

    _, received = exchange(peer, browser, tls_context=context, allow_tls_failure=True)
    assert received == []
