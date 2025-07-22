"""Enhanced security middleware"""
import time
import hashlib
import hmac
from typing import Set
import ipaddress
import os
from fastapi.responses import JSONResponse

class SecurityHeadersMiddleware:
    """Add security headers to all responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))

                    # Security headers
                    security_headers = {
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                        b"content-security-policy": (
                            b"default-src 'self'; "
                            b"script-src 'self' 'unsafe-inline'; "
                            b"style-src 'self' 'unsafe-inline'"
                        ),
                        b"referrer-policy": b"strict-origin-when-cross-origin",
                        b"permissions-policy": b"geolocation=(), microphone=(), camera=()"
                    }


                    headers.update(security_headers)
                    message["headers"] = list(headers.items())

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

class IPWhitelistMiddleware:
    """IP whitelist middleware for admin endpoints"""

    def __init__(self, app, allowed_ips: Set[str] = None, admin_paths: Set[str] = None):
        self.app = app
        self.allowed_ips = allowed_ips or set()
        self.admin_paths = admin_paths or {"/api/sandbox", "/api/admin"}

        # Parse CIDR ranges
        self.allowed_networks = []
        for ip in self.allowed_ips:
            try:
                self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
            except ValueError:
                pass

    def is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP is in whitelist"""
        if not self.allowed_networks:
            return True  # No restrictions if no IPs configured

        try:
            client_ip_obj = ipaddress.ip_address(client_ip)
            return any(client_ip_obj in network for network in self.allowed_networks)
        except ValueError:
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]

            # Check if this is an admin path
            is_admin_path = any(path.startswith(admin_path) for admin_path in self.admin_paths)

            if is_admin_path:
                client_ip = None
                for header_name, header_value in scope.get("headers", []):
                    if header_name == b"x-forwarded-for":
                        client_ip = header_value.decode().split(",")[0].strip()
                        break

                if not client_ip and scope.get("client"):
                    client_ip = scope["client"][0]

                if client_ip and not self.is_ip_allowed(client_ip):
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "IP not allowed"}
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)

class RequestSignatureMiddleware:
    """Verify request signatures for API calls"""

    def __init__(self, app, secret_key: str = None, require_signature: bool = False):
        self.app = app
        self.secret_key = secret_key or os.getenv("API_SIGNATURE_SECRET")
        self.require_signature = require_signature

    def verify_signature(self, body: bytes, signature: str, timestamp: str) -> bool:
        """Verify HMAC signature"""
        if not self.secret_key:
            return not self.require_signature

        try:
            # Check timestamp (prevent replay attacks)
            request_time = int(timestamp)
            current_time = int(time.time())
            if abs(current_time - request_time) > 300:  # 5 minutes
                return False

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                f"{timestamp}.{body.decode()}".encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except (ValueError, TypeError):
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ["POST", "PUT", "PATCH"]:
            # Read the request body
            body = b""

            # Check for signature headers
            headers = dict(scope.get("headers", []))
            signature = headers.get(b"x-signature", b"").decode()
            timestamp = headers.get(b"x-timestamp", b"").decode()

            if self.require_signature and (not signature or not timestamp):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Missing signature"}
                )
                await response(scope, receive, send)
                return

            if signature and timestamp:
                # We need to read the full body first
                while True:
                    message = await receive()
                    if message["type"] == "http.request":
                        body += message.get("body", b"")
                        if not message.get("more_body", False):
                            break

                if not self.verify_signature(body, signature, timestamp):
                    response = JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid signature"}
                    )
                    await response(scope, receive, send)
                    return

                # Recreate receive function with cached body
                body_sent = False
                async def receive_wrapper():
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False
                        }
                    return {"type": "http.disconnect"}

                await self.app(scope, receive_wrapper, send)
            else:
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)

# CSRF Protection
class CSRFMiddleware:
    """CSRF protection middleware"""

    def __init__(self, app, secret_key: str = None):
        self.app = app
        self.secret_key = secret_key or os.getenv("CSRF_SECRET_KEY", "fallback-csrf-secret")

    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}:{signature}"

    def verify_csrf_token(self, token: str, session_id: str) -> bool:
        """Verify CSRF token"""
        try:
            timestamp, signature = token.split(":", 1)

            # Check if token is not too old (1 hour)
            token_time = int(timestamp)
            if time.time() - token_time > 3600:
                return False

            expected_data = f"{session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                expected_data.encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except (ValueError, TypeError):
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ["POST", "PUT", "DELETE", "PATCH"]:
            headers = dict(scope.get("headers", []))
            csrf_token = headers.get(b"x-csrf-token", b"").decode()

            # Extract session ID from cookies (simplified)
            cookies = headers.get(b"cookie", b"").decode()
            session_id = None
            for cookie in cookies.split(";"):
                if "session_token=" in cookie:
                    session_id = cookie.split("session_token=")[1].split(";")[0]
                    break

            if session_id and not self.verify_csrf_token(csrf_token, session_id):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token invalid"}
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
