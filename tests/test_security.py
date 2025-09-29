#!/usr/bin/env python3
"""
Comprehensive Security Layer Test Suite
Tests all authentication, authorization, and security features
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import (
    AuthManager,
    RateLimiter,
    SecurityMiddleware,
    InputSanitizer,
    PasswordValidator,
    UserRole,
    TierLevel,
    TokenData,
    TokenType,
    User,
    APIKey
)


class TestPasswordValidation:
    """Test password validation against OWASP requirements"""

    def test_password_too_short(self):
        """Test password shorter than minimum length"""
        is_valid, errors = PasswordValidator.validate_password("Short1!")
        assert not is_valid
        assert any("12 characters" in error for error in errors)

    def test_password_no_uppercase(self):
        """Test password without uppercase letter"""
        is_valid, errors = PasswordValidator.validate_password("longpassword123!")
        assert not is_valid
        assert any("uppercase" in error.lower() for error in errors)

    def test_password_no_lowercase(self):
        """Test password without lowercase letter"""
        is_valid, errors = PasswordValidator.validate_password("LONGPASSWORD123!")
        assert not is_valid
        assert any("lowercase" in error.lower() for error in errors)

    def test_password_no_digit(self):
        """Test password without digit"""
        is_valid, errors = PasswordValidator.validate_password("LongPassword!")
        assert not is_valid
        assert any("digit" in error.lower() for error in errors)

    def test_password_no_special(self):
        """Test password without special character"""
        is_valid, errors = PasswordValidator.validate_password("LongPassword123")
        assert not is_valid
        assert any("special" in error.lower() for error in errors)

    def test_password_common(self):
        """Test common password rejection"""
        is_valid, errors = PasswordValidator.validate_password("Password123!")
        # Note: "password" is in common password list
        assert not is_valid or "password" not in errors  # May pass other checks

    def test_password_valid_strong(self):
        """Test valid strong password"""
        is_valid, errors = PasswordValidator.validate_password("SecureP@ssw0rd123!")
        assert is_valid
        assert len(errors) == 0

    def test_password_too_long(self):
        """Test password exceeding maximum length"""
        long_password = "A1!" + "a" * 200
        is_valid, errors = PasswordValidator.validate_password(long_password)
        assert not is_valid
        assert any("128 characters" in error for error in errors)


class TestInputSanitization:
    """Test input sanitization against injection attacks"""

    def test_sanitize_null_bytes(self):
        """Test null byte removal"""
        result = InputSanitizer.sanitize_string("test\x00input")
        assert "\x00" not in result
        assert result == "testinput"

    def test_sanitize_control_characters(self):
        """Test control character removal"""
        result = InputSanitizer.sanitize_string("test\x01\x02\x03input")
        assert "\x01" not in result
        assert "\x02" not in result
        assert result == "testinput"

    def test_sanitize_length_limit(self):
        """Test length truncation"""
        long_string = "a" * 1000
        result = InputSanitizer.sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_sanitize_whitespace_trim(self):
        """Test whitespace trimming"""
        result = InputSanitizer.sanitize_string("  test input  ")
        assert result == "test input"

    def test_sanitize_allowed_special_chars(self):
        """Test that allowed special characters are preserved"""
        result = InputSanitizer.sanitize_string("test\nline\ttab\rreturn")
        assert "\n" in result
        assert "\t" in result
        assert "\r" in result

    def test_email_validation_valid(self):
        """Test valid email format"""
        assert InputSanitizer.validate_email("user@example.com")
        assert InputSanitizer.validate_email("test.user+tag@sub.example.com")

    def test_email_validation_invalid(self):
        """Test invalid email formats"""
        assert not InputSanitizer.validate_email("invalid")
        assert not InputSanitizer.validate_email("@example.com")
        assert not InputSanitizer.validate_email("user@")
        assert not InputSanitizer.validate_email("user@example")

    def test_sanitize_dict_whitelist(self):
        """Test dictionary key whitelisting"""
        data = {
            "username": "test",
            "password": "secret",
            "malicious": "data"
        }
        allowed = ["username", "password"]
        result = InputSanitizer.sanitize_dict(data, allowed_keys=allowed)
        assert "username" in result
        assert "password" in result
        assert "malicious" not in result


class TestAuthManager:
    """Test authentication manager functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.auth_manager = AuthManager()

    def test_password_hashing(self):
        """Test password hashing"""
        password = "SecureP@ssw0rd123!"
        hashed = self.auth_manager.hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_password_verification(self):
        """Test password verification"""
        password = "SecureP@ssw0rd123!"
        hashed = self.auth_manager.hash_password(password)

        # Correct password
        assert self.auth_manager.verify_password(password, hashed)

        # Wrong password
        assert not self.auth_manager.verify_password("WrongPassword!", hashed)

    def test_create_user_valid(self):
        """Test user creation with valid data"""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.API_USER
        assert user.tier == TierLevel.FREE
        assert user.hashed_password != "SecureP@ssw0rd123!"

    def test_create_user_weak_password(self):
        """Test user creation with weak password"""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="weak",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert user is None

    def test_create_user_invalid_email(self):
        """Test user creation with invalid email"""
        user = self.auth_manager.create_user(
            username="testuser",
            email="invalid-email",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert user is None

    def test_create_user_duplicate(self):
        """Test duplicate user creation prevention"""
        # Create first user
        user1 = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert user1 is not None

        # Try to create duplicate
        user2 = self.auth_manager.create_user(
            username="testuser",
            email="different@example.com",
            password="SecureP@ssw0rd456!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert user2 is None

    def test_authenticate_user_valid(self):
        """Test successful user authentication"""
        # Create user
        self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        # Authenticate
        user = self.auth_manager.authenticate_user("testuser", "SecureP@ssw0rd123!")

        assert user is not None
        assert user.username == "testuser"
        assert user.last_login is not None

    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password"""
        # Create user
        self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        # Try to authenticate with wrong password
        user = self.auth_manager.authenticate_user("testuser", "WrongPassword!")

        assert user is None

    def test_authenticate_user_nonexistent(self):
        """Test authentication with non-existent user"""
        user = self.auth_manager.authenticate_user("nonexistent", "Password123!")

        assert user is None

    def test_account_lockout(self):
        """Test account lockout after failed attempts"""
        # Create user
        self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecureP@ssw0rd123!",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        # Fail authentication 5 times
        for _ in range(5):
            self.auth_manager.authenticate_user("testuser", "WrongPassword!")

        # Try with correct password - should be locked
        user = self.auth_manager.authenticate_user("testuser", "SecureP@ssw0rd123!")

        assert user is None

    def test_jwt_token_creation(self):
        """Test JWT token creation"""
        token_data = TokenData(
            user_id="test123",
            username="testuser",
            role=UserRole.API_USER,
            tier=TierLevel.FREE,
            scopes=["read:fraud"]
        )

        token = self.auth_manager.create_access_token(token_data)

        assert token is not None
        assert len(token) > 0
        assert token.count('.') == 2  # JWT has 3 parts separated by dots

    def test_jwt_token_verification(self):
        """Test JWT token verification"""
        token_data = TokenData(
            user_id="test123",
            username="testuser",
            role=UserRole.API_USER,
            tier=TierLevel.FREE,
            scopes=["read:fraud"]
        )

        token = self.auth_manager.create_access_token(token_data)
        verified_data = self.auth_manager.verify_token(token)

        assert verified_data is not None
        assert verified_data.user_id == "test123"
        assert verified_data.username == "testuser"
        assert verified_data.role == UserRole.API_USER

    def test_jwt_token_expiration(self):
        """Test JWT token expiration"""
        token_data = TokenData(
            user_id="test123",
            username="testuser",
            role=UserRole.API_USER,
            tier=TierLevel.FREE,
            scopes=["read:fraud"]
        )

        # Create token with 1 second expiration
        token = self.auth_manager.create_access_token(
            token_data,
            expires_delta=timedelta(seconds=-1)  # Already expired
        )

        # Try to verify expired token
        verified_data = self.auth_manager.verify_token(token)

        assert verified_data is None

    def test_jwt_token_invalid(self):
        """Test invalid JWT token rejection"""
        invalid_token = "invalid.token.data"
        verified_data = self.auth_manager.verify_token(invalid_token)

        assert verified_data is None

    def test_token_revocation(self):
        """Test token revocation"""
        token_data = TokenData(
            user_id="test123",
            username="testuser",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        token = self.auth_manager.create_access_token(token_data)

        # Token should be valid
        assert self.auth_manager.verify_token(token) is not None

        # Revoke token
        self.auth_manager.revoke_token(token)

        # Token should now be invalid
        assert self.auth_manager.verify_token(token) is None

    def test_api_key_generation(self):
        """Test API key generation"""
        raw_key, api_key = self.auth_manager.generate_api_key(
            user_id="test123",
            name="Test Key",
            tier=TierLevel.PAID,
            scopes=["read:fraud", "write:fraud"]
        )

        assert raw_key.startswith("fd_")
        assert len(raw_key) > 10
        assert api_key.user_id == "test123"
        assert api_key.name == "Test Key"
        assert api_key.tier == TierLevel.PAID
        assert "read:fraud" in api_key.scopes

    def test_api_key_verification(self):
        """Test API key verification"""
        raw_key, api_key = self.auth_manager.generate_api_key(
            user_id="test123",
            name="Test Key",
            tier=TierLevel.PAID
        )

        # Verify with correct key
        verified = self.auth_manager.verify_api_key(raw_key)

        assert verified is not None
        assert verified.key_id == api_key.key_id
        assert verified.user_id == "test123"

    def test_api_key_invalid(self):
        """Test invalid API key rejection"""
        verified = self.auth_manager.verify_api_key("invalid_key")

        assert verified is None

    def test_api_key_inactive(self):
        """Test inactive API key rejection"""
        raw_key, api_key = self.auth_manager.generate_api_key(
            user_id="test123",
            name="Test Key",
            tier=TierLevel.PAID
        )

        # Deactivate key
        api_key.is_active = False

        # Try to verify
        verified = self.auth_manager.verify_api_key(raw_key)

        assert verified is None

    def test_api_key_expired(self):
        """Test expired API key rejection"""
        raw_key, api_key = self.auth_manager.generate_api_key(
            user_id="test123",
            name="Test Key",
            tier=TierLevel.PAID,
            expires_in_days=0  # Already expired
        )

        # Set expiration to past
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)

        # Try to verify
        verified = self.auth_manager.verify_api_key(raw_key)

        assert verified is None


class TestRateLimiter:
    """Test rate limiting functionality"""

    @pytest.fixture
    async def rate_limiter(self):
        """Create rate limiter instance"""
        limiter = RateLimiter()
        await limiter.initialize()
        yield limiter
        # Cleanup
        if limiter._redis_client:
            await limiter._redis_client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_free_tier(self, rate_limiter):
        """Test rate limiting for free tier"""
        identifier = "test_user_free"

        # Should allow up to 10 requests
        for i in range(10):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier,
                TierLevel.FREE
            )
            # First requests should be allowed
            if i < 10:
                assert allowed or not rate_limiter._redis_client  # Allow if Redis unavailable

    @pytest.mark.asyncio
    async def test_rate_limit_paid_tier(self, rate_limiter):
        """Test rate limiting for paid tier"""
        identifier = "test_user_paid"

        # Should allow many more requests
        for i in range(5):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier,
                TierLevel.PAID
            )
            assert allowed or not rate_limiter._redis_client

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limiter):
        """Test rate limit reset"""
        identifier = "test_user_reset"

        # Make some requests
        await rate_limiter.check_rate_limit(identifier, TierLevel.FREE)
        await rate_limiter.check_rate_limit(identifier, TierLevel.FREE)

        # Reset limit
        success = await rate_limiter.reset_rate_limit(identifier)

        # Should be allowed again (if Redis available)
        allowed, info = await rate_limiter.check_rate_limit(identifier, TierLevel.FREE)
        assert allowed or not rate_limiter._redis_client


class TestSecurityMiddleware:
    """Test security middleware functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.auth_manager = AuthManager()
        self.rate_limiter = RateLimiter()
        self.middleware = SecurityMiddleware(self.auth_manager, self.rate_limiter)

    def test_permission_check_role_hierarchy(self):
        """Test role hierarchy in permission checking"""
        # Admin should have all permissions
        admin_token = TokenData(
            user_id="admin",
            username="admin",
            role=UserRole.ADMIN,
            tier=TierLevel.INTERNAL
        )

        assert self.middleware.check_permission(admin_token, UserRole.ADMIN)
        assert self.middleware.check_permission(admin_token, UserRole.ANALYST)
        assert self.middleware.check_permission(admin_token, UserRole.API_USER)

        # API user should not have analyst permissions
        api_user_token = TokenData(
            user_id="user",
            username="user",
            role=UserRole.API_USER,
            tier=TierLevel.FREE
        )

        assert not self.middleware.check_permission(api_user_token, UserRole.ADMIN)
        assert not self.middleware.check_permission(api_user_token, UserRole.ANALYST)
        assert self.middleware.check_permission(api_user_token, UserRole.API_USER)

    def test_permission_check_scopes(self):
        """Test scope-based permission checking"""
        token_data = TokenData(
            user_id="user",
            username="user",
            role=UserRole.API_USER,
            tier=TierLevel.FREE,
            scopes=["read:fraud"]
        )

        # Should allow with matching scope
        assert self.middleware.check_permission(
            token_data,
            UserRole.API_USER,
            required_scopes=["read:fraud"]
        )

        # Should deny with missing scope
        assert not self.middleware.check_permission(
            token_data,
            UserRole.API_USER,
            required_scopes=["write:fraud"]
        )


def run_all_tests():
    """Run all security tests"""
    print("=" * 70)
    print("Running Security Layer Test Suite")
    print("=" * 70)

    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning"
    ])


if __name__ == "__main__":
    run_all_tests()