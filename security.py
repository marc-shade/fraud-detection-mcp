#!/usr/bin/env python3
"""
Production-Grade Security Layer for Fraud Detection MCP
Implements OWASP best practices for authentication, authorization, and API security

OWASP Top 10 2021 Mitigations:
- A01:2021 - Broken Access Control: RBAC + Token validation
- A02:2021 - Cryptographic Failures: Strong encryption, secure hashing
- A03:2021 - Injection: Input sanitization and validation
- A04:2021 - Insecure Design: Defense in depth architecture
- A05:2021 - Security Misconfiguration: Secure defaults, hardening
- A07:2021 - Identification and Authentication Failures: JWT + MFA ready
- A08:2021 - Software and Data Integrity Failures: Token signing
"""

import secrets
import hashlib
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from functools import wraps

# Cryptography and authentication
import bcrypt
from jose import JWTError, jwt
from passlib.context import CryptContext

# FastAPI and validation
from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, Field

# Rate limiting and caching
import redis.asyncio as redis
from datetime import datetime as dt

# Configuration
from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# OWASP Password Requirements
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGIT = True
PASSWORD_REQUIRE_SPECIAL = True

# Security headers as per OWASP Secure Headers Project
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
}


class UserRole(str, Enum):
    """User roles for RBAC (Role-Based Access Control)"""
    ADMIN = "admin"
    ANALYST = "analyst"
    API_USER = "api_user"
    READ_ONLY = "read_only"


class TierLevel(str, Enum):
    """API tier levels for rate limiting"""
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


class TokenType(str, Enum):
    """Token types for different purposes"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


# Password hashing context with bcrypt (OWASP recommended)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # OWASP minimum recommendation
)


class User(BaseModel):
    """User model with security attributes"""
    user_id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole = UserRole.API_USER
    tier: TierLevel = TierLevel.FREE
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    lockout_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


class APIKey(BaseModel):
    """API Key model for API authentication"""
    key_id: str
    key_hash: str  # Never store raw keys
    user_id: str
    name: str
    scopes: List[str] = Field(default_factory=list)
    tier: TierLevel = TierLevel.FREE
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0


class TokenData(BaseModel):
    """JWT token payload data"""
    user_id: str
    username: str
    role: UserRole
    tier: TierLevel
    token_type: TokenType = TokenType.ACCESS
    scopes: List[str] = Field(default_factory=list)


class PasswordValidator:
    """OWASP-compliant password validation"""

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against OWASP recommendations

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Length check
        if len(password) < PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
        if len(password) > PASSWORD_MAX_LENGTH:
            errors.append(f"Password must not exceed {PASSWORD_MAX_LENGTH} characters")

        # Complexity checks
        if PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if PASSWORD_REQUIRE_DIGIT and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if PASSWORD_REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        # Common password check (basic - extend with compromised password list)
        common_passwords = ['password', '12345678', 'qwerty', 'abc123']
        if password.lower() in common_passwords:
            errors.append("Password is too common")

        return len(errors) == 0, errors


class InputSanitizer:
    """Input sanitization to prevent injection attacks (OWASP A03)"""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """
        Sanitize string input to prevent injection attacks

        - Removes control characters
        - Limits length
        - Prevents null bytes
        """
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Remove null bytes (can cause issues in C-based systems)
        value = value.replace('\x00', '')

        # Remove or replace control characters (except newline, tab, carriage return)
        value = ''.join(char for char in value if char.isprintable() or char in '\n\t\r')

        # Truncate to maximum length
        value = value[:max_length]

        # Strip whitespace
        value = value.strip()

        return value

    @staticmethod
    def sanitize_dict(data: Dict[str, Any], allowed_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Sanitize dictionary to prevent mass assignment vulnerabilities

        Only allows whitelisted keys if provided
        """
        if allowed_keys:
            return {k: v for k, v in data.items() if k in allowed_keys}
        return data

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format (basic validation)"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) and len(email) <= 255


class AuthManager:
    """
    Authentication Manager - Handles JWT tokens and API keys
    Implements OWASP authentication best practices
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize AuthManager with secret key

        Args:
            secret_key: JWT signing key (generated if not provided)
        """
        self.secret_key = secret_key or config.JWT_SECRET_KEY or secrets.token_urlsafe(32)
        self.algorithm = config.JWT_ALGORITHM
        self.access_token_expire_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES

        # In-memory storage (use database in production)
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._revoked_tokens: set = set()  # Token revocation list

        logger.info("AuthManager initialized with secure defaults")

    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt

        OWASP: Use strong adaptive hashing with salt
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash

        Constant-time comparison to prevent timing attacks
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def create_access_token(
        self,
        data: TokenData,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            data: Token payload data
            expires_delta: Token expiration time

        Returns:
            Signed JWT token
        """
        to_encode = data.dict()

        # Set expiration
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),  # Unique token ID for revocation
        })

        # Sign token
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        logger.info(f"Access token created for user: {data.user_id}")
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token to verify

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            # Check if token is revoked
            if token in self._revoked_tokens:
                logger.warning("Attempted use of revoked token")
                return None

            # Decode and verify signature
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}  # Verify expiration
            )

            # Extract token data
            token_data = TokenData(
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                role=UserRole(payload.get("role")),
                tier=TierLevel(payload.get("tier")),
                token_type=TokenType(payload.get("token_type", "access")),
                scopes=payload.get("scopes", [])
            )

            return token_data

        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (add to revocation list)

        In production, use Redis or database for token revocation
        """
        try:
            self._revoked_tokens.add(token)
            logger.info("Token revoked successfully")
            return True
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return False

    def generate_api_key(
        self,
        user_id: str,
        name: str,
        tier: TierLevel = TierLevel.FREE,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None
    ) -> Tuple[str, APIKey]:
        """
        Generate API key for user

        Returns:
            Tuple of (raw_key, api_key_object)

        Note: Raw key is only shown once at creation
        """
        # Generate cryptographically secure key
        raw_key = f"fd_{secrets.token_urlsafe(32)}"

        # Hash the key (never store raw keys)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Create API key object
        api_key = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            tier=tier,
            scopes=scopes or [],
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None
        )

        # Store API key
        self._api_keys[api_key.key_id] = api_key

        logger.info(f"API key generated for user {user_id}: {api_key.key_id}")

        return raw_key, api_key

    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Verify API key and return associated key object

        Args:
            raw_key: Raw API key provided by client

        Returns:
            APIKey object if valid, None otherwise
        """
        try:
            # Hash provided key
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            # Find matching key
            for api_key in self._api_keys.values():
                if api_key.key_hash == key_hash:
                    # Check if active
                    if not api_key.is_active:
                        logger.warning(f"Inactive API key used: {api_key.key_id}")
                        return None

                    # Check expiration
                    if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                        logger.warning(f"Expired API key used: {api_key.key_id}")
                        return None

                    # Update usage stats
                    api_key.last_used = datetime.utcnow()
                    api_key.usage_count += 1

                    return api_key

            logger.warning("Invalid API key attempted")
            return None

        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return None

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.API_USER,
        tier: TierLevel = TierLevel.FREE
    ) -> Optional[User]:
        """
        Create new user with password validation

        Args:
            username: Username (will be sanitized)
            email: Email address
            password: Plain password (will be hashed)
            role: User role
            tier: API tier level

        Returns:
            User object if created successfully
        """
        try:
            # Sanitize inputs
            username = InputSanitizer.sanitize_string(username, max_length=50)
            email = InputSanitizer.sanitize_string(email, max_length=255)

            # Validate email
            if not InputSanitizer.validate_email(email):
                logger.error("Invalid email format")
                return None

            # Validate password
            is_valid, errors = PasswordValidator.validate_password(password)
            if not is_valid:
                logger.error(f"Password validation failed: {errors}")
                return None

            # Check for duplicate username/email
            for user in self._users.values():
                if user.username == username or user.email == email:
                    logger.error("Username or email already exists")
                    return None

            # Create user
            user = User(
                user_id=secrets.token_urlsafe(16),
                username=username,
                email=email,
                hashed_password=self.hash_password(password),
                role=role,
                tier=tier
            )

            self._users[user.user_id] = user

            logger.info(f"User created successfully: {user.user_id}")
            return user

        except Exception as e:
            logger.error(f"User creation error: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/password

        Implements account lockout after failed attempts
        """
        try:
            # Find user
            user = None
            for u in self._users.values():
                if u.username == username:
                    user = u
                    break

            if not user:
                logger.warning(f"Authentication failed: user not found - {username}")
                return None

            # Check if account is locked
            if user.lockout_until and datetime.utcnow() < user.lockout_until:
                logger.warning(f"Account locked: {username}")
                return None

            # Verify password
            if not self.verify_password(password, user.hashed_password):
                # Increment failed attempts
                user.failed_login_attempts += 1

                # Lock account after 5 failed attempts (OWASP recommendation)
                if user.failed_login_attempts >= 5:
                    user.lockout_until = datetime.utcnow() + timedelta(minutes=30)
                    logger.warning(f"Account locked due to failed attempts: {username}")

                return None

            # Successful authentication
            user.failed_login_attempts = 0
            user.lockout_until = None
            user.last_login = datetime.utcnow()

            logger.info(f"User authenticated successfully: {username}")
            return user

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None


class RateLimiter:
    """
    Redis-backed rate limiter with tier-based limits
    Prevents abuse and DoS attacks (OWASP)
    """

    def __init__(self, redis_url: str = None):
        """
        Initialize rate limiter with Redis connection

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or config.REDIS_URL
        self._redis_client: Optional[redis.Redis] = None

        # Rate limit definitions per tier
        self.rate_limits = {
            TierLevel.FREE: (10, 60),        # 10 requests per minute
            TierLevel.PAID: (1000, 60),      # 1000 requests per minute
            TierLevel.ENTERPRISE: (10000, 60),  # 10000 requests per minute
            TierLevel.INTERNAL: (100000, 60),   # Unlimited (practical limit)
        }

        logger.info("RateLimiter initialized")

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self._redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis_client.ping()
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._redis_client = None

    async def check_rate_limit(
        self,
        identifier: str,
        tier: TierLevel = TierLevel.FREE
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit

        Args:
            identifier: Unique identifier (user_id, API key, IP)
            tier: User tier level

        Returns:
            Tuple of (allowed, limit_info)
        """
        if not self._redis_client:
            # Fallback: allow request if Redis unavailable
            logger.warning("Rate limiting unavailable - Redis not connected")
            return True, {"limit": -1, "remaining": -1, "reset": -1}

        try:
            max_requests, window_seconds = self.rate_limits[tier]

            # Redis key
            key = f"rate_limit:{identifier}"

            # Get current count
            current = await self._redis_client.get(key)

            if current is None:
                # First request in window
                await self._redis_client.setex(key, window_seconds, 1)
                return True, {
                    "limit": max_requests,
                    "remaining": max_requests - 1,
                    "reset": int(datetime.utcnow().timestamp()) + window_seconds
                }

            current = int(current)

            if current >= max_requests:
                # Rate limit exceeded
                ttl = await self._redis_client.ttl(key)
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "reset": int(datetime.utcnow().timestamp()) + ttl
                }

            # Increment counter
            await self._redis_client.incr(key)

            return True, {
                "limit": max_requests,
                "remaining": max_requests - current - 1,
                "reset": int(datetime.utcnow().timestamp()) + await self._redis_client.ttl(key)
            }

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open on errors
            return True, {"limit": -1, "remaining": -1, "reset": -1}

    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier"""
        if not self._redis_client:
            return False

        try:
            key = f"rate_limit:{identifier}"
            await self._redis_client.delete(key)
            logger.info(f"Rate limit reset for {identifier}")
            return True
        except Exception as e:
            logger.error(f"Rate limit reset error: {e}")
            return False


class SecurityMiddleware:
    """
    FastAPI middleware for security enforcement

    Implements:
    - Authentication validation
    - Authorization (RBAC)
    - Rate limiting
    - Security headers
    - Request logging
    - Input validation
    """

    def __init__(
        self,
        auth_manager: AuthManager,
        rate_limiter: RateLimiter
    ):
        """
        Initialize security middleware

        Args:
            auth_manager: Authentication manager instance
            rate_limiter: Rate limiter instance
        """
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.security_bearer = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request, call_next):
        """
        Process request through security checks
        """
        try:
            # Add security headers to response
            response = await call_next(request)
            for header, value in SECURITY_HEADERS.items():
                response.headers[header] = value

            return response

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security processing error"
            )

    async def authenticate_request(self, request: Request) -> Optional[TokenData]:
        """
        Authenticate request using JWT or API key

        Checks:
        1. Bearer token (JWT)
        2. API key header
        3. API key query parameter (discouraged but supported)
        """
        # Try Bearer token first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_data = self.auth_manager.verify_token(token)
            if token_data:
                return token_data

        # Try API key header
        api_key = request.headers.get(config.API_KEY_HEADER)
        if api_key:
            api_key_obj = self.auth_manager.verify_api_key(api_key)
            if api_key_obj:
                # Convert API key to token data
                user = self.auth_manager._users.get(api_key_obj.user_id)
                if user:
                    return TokenData(
                        user_id=user.user_id,
                        username=user.username,
                        role=user.role,
                        tier=api_key_obj.tier,
                        token_type=TokenType.API_KEY,
                        scopes=api_key_obj.scopes
                    )

        # Try API key query parameter (last resort)
        api_key = request.query_params.get("api_key")
        if api_key:
            logger.warning("API key passed in query parameter - use header instead")
            api_key_obj = self.auth_manager.verify_api_key(api_key)
            if api_key_obj:
                user = self.auth_manager._users.get(api_key_obj.user_id)
                if user:
                    return TokenData(
                        user_id=user.user_id,
                        username=user.username,
                        role=user.role,
                        tier=api_key_obj.tier,
                        token_type=TokenType.API_KEY,
                        scopes=api_key_obj.scopes
                    )

        return None

    async def check_rate_limit(
        self,
        request: Request,
        token_data: Optional[TokenData] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for request

        Uses:
        - User ID if authenticated
        - IP address if not authenticated
        """
        if token_data:
            identifier = token_data.user_id
            tier = token_data.tier
        else:
            identifier = request.client.host
            tier = TierLevel.FREE

        return await self.rate_limiter.check_rate_limit(identifier, tier)

    def check_permission(
        self,
        token_data: TokenData,
        required_role: UserRole,
        required_scopes: Optional[List[str]] = None
    ) -> bool:
        """
        Check if user has required permissions (RBAC)

        Role hierarchy: ADMIN > ANALYST > API_USER > READ_ONLY
        """
        # Role hierarchy
        role_levels = {
            UserRole.ADMIN: 3,
            UserRole.ANALYST: 2,
            UserRole.API_USER: 1,
            UserRole.READ_ONLY: 0
        }

        # Check role level
        user_level = role_levels.get(token_data.role, 0)
        required_level = role_levels.get(required_role, 0)

        if user_level < required_level:
            return False

        # Check scopes if specified
        if required_scopes:
            if not all(scope in token_data.scopes for scope in required_scopes):
                return False

        return True


def require_auth(
    required_role: UserRole = UserRole.API_USER,
    required_scopes: Optional[List[str]] = None
):
    """
    Decorator for protecting endpoints with authentication

    Usage:
        @require_auth(required_role=UserRole.ADMIN)
        async def admin_only_endpoint():
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs
            request = kwargs.get('request')
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )

            # Get middleware from app state
            middleware = request.app.state.security_middleware

            # Authenticate
            token_data = await middleware.authenticate_request(request)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Check permissions
            if not middleware.check_permission(token_data, required_role, required_scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )

            # Check rate limit
            allowed, limit_info = await middleware.check_rate_limit(request, token_data)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(limit_info["limit"]),
                        "X-RateLimit-Remaining": str(limit_info["remaining"]),
                        "X-RateLimit-Reset": str(limit_info["reset"])
                    }
                )

            # Add token data to kwargs
            kwargs['token_data'] = token_data

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 70)
        print("Security Layer - Production-Grade Authentication & Authorization")
        print("=" * 70)

        # Initialize components
        auth_manager = AuthManager()
        rate_limiter = RateLimiter()
        await rate_limiter.initialize()

        print("\n1. Creating test user...")
        user = auth_manager.create_user(
            username="test_analyst",
            email="analyst@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.ANALYST,
            tier=TierLevel.PAID
        )

        if user:
            print(f"   User created: {user.username} (ID: {user.user_id})")
            print(f"   Role: {user.role.value}, Tier: {user.tier.value}")

        print("\n2. Testing authentication...")
        authenticated_user = auth_manager.authenticate_user("test_analyst", "SecureP@ssw0rd123!")
        if authenticated_user:
            print(f"   Authentication successful for: {authenticated_user.username}")

        print("\n3. Generating JWT token...")
        token_data = TokenData(
            user_id=user.user_id,
            username=user.username,
            role=user.role,
            tier=user.tier,
            scopes=["read:fraud_detection", "write:fraud_detection"]
        )
        access_token = auth_manager.create_access_token(token_data)
        print(f"   Token generated: {access_token[:50]}...")

        print("\n4. Verifying token...")
        verified_data = auth_manager.verify_token(access_token)
        if verified_data:
            print(f"   Token verified for user: {verified_data.username}")
            print(f"   Scopes: {', '.join(verified_data.scopes)}")

        print("\n5. Generating API key...")
        raw_key, api_key = auth_manager.generate_api_key(
            user_id=user.user_id,
            name="Test API Key",
            tier=TierLevel.PAID,
            scopes=["read:fraud_detection"],
            expires_in_days=30
        )
        print(f"   API Key: {raw_key[:30]}...")
        print(f"   Key ID: {api_key.key_id}")

        print("\n6. Verifying API key...")
        verified_key = auth_manager.verify_api_key(raw_key)
        if verified_key:
            print(f"   API key valid for user: {verified_key.user_id}")
            print(f"   Usage count: {verified_key.usage_count}")

        print("\n7. Testing rate limiting...")
        for i in range(3):
            allowed, limit_info = await rate_limiter.check_rate_limit(
                user.user_id,
                user.tier
            )
            print(f"   Request {i+1}: Allowed={allowed}, "
                  f"Remaining={limit_info['remaining']}/{limit_info['limit']}")

        print("\n8. Testing password validation...")
        test_passwords = [
            "weak",
            "WeakPassword",
            "WeakP@ssw0rd",
            "StrongP@ssw0rd123!"
        ]
        for pwd in test_passwords:
            is_valid, errors = PasswordValidator.validate_password(pwd)
            print(f"   '{pwd}': {'VALID' if is_valid else 'INVALID'}")
            if errors:
                print(f"      Errors: {', '.join(errors)}")

        print("\n9. Security Headers:")
        for header, value in SECURITY_HEADERS.items():
            print(f"   {header}: {value[:50]}...")

        print("\n" + "=" * 70)
        print("Security Layer Testing Complete")
        print("=" * 70)

    # Run tests
    asyncio.run(main())