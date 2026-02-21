#!/usr/bin/env python3
"""
Security Layer Integration Example
Demonstrates how to integrate the security layer with FastAPI endpoints
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import security components
import sys
sys.path.append('..')
from security import (
    AuthManager,
    RateLimiter,
    SecurityMiddleware,
    require_auth,
    UserRole,
    TierLevel,
    TokenData,
    InputSanitizer
)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Secure fraud detection API with authentication",
    version="2.0.0"
)

# Initialize security components
auth_manager = AuthManager()
rate_limiter = RateLimiter()
security_middleware = SecurityMiddleware(auth_manager, rate_limiter)

# Store in app state for access in endpoints
app.state.security_middleware = security_middleware
app.state.auth_manager = auth_manager

# CORS configuration (adjust for your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit methods
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Add security middleware
@app.middleware("http")
async def security_middleware_handler(request: Request, call_next):
    return await security_middleware(request, call_next)


# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[UserRole] = UserRole.API_USER
    tier: Optional[TierLevel] = TierLevel.FREE


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


class APIKeyResponse(BaseModel):
    api_key: str
    key_id: str
    message: str = "Store this key securely - it won't be shown again!"


class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    location: str
    timestamp: str
    payment_method: Optional[str] = "credit_card"


# =============================================================================
# PUBLIC ENDPOINTS (No Authentication Required)
# =============================================================================

@app.get("/")
async def root():
    """Public endpoint - API information"""
    return {
        "service": "Fraud Detection API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "authentication": "Bearer token or API key required"
    }


@app.get("/health")
async def health_check():
    """Public endpoint - Health check"""
    return {
        "status": "healthy",
        "timestamp": "2025-09-29T00:00:00Z"
    }


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: Request, credentials: LoginRequest):
    """
    Public endpoint - User authentication

    Returns JWT token for authenticated users
    """
    # Sanitize inputs
    username = InputSanitizer.sanitize_string(credentials.username, max_length=50)

    # Authenticate user
    auth_manager = request.app.state.auth_manager
    user = auth_manager.authenticate_user(username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Generate token
    token_data = TokenData(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        tier=user.tier,
        scopes=["read:fraud_detection", "write:fraud_detection"]
    )

    access_token = auth_manager.create_access_token(token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800
    }


@app.post("/auth/register")
async def register(request: Request, user_data: UserCreateRequest):
    """
    Public endpoint - User registration

    Creates new user account with validation
    """
    auth_manager = request.app.state.auth_manager

    # Create user
    user = auth_manager.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role,
        tier=user_data.tier
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="User creation failed - check password requirements or duplicate username/email"
        )

    return {
        "message": "User created successfully",
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "tier": user.tier.value
    }


# =============================================================================
# PROTECTED ENDPOINTS (Authentication Required)
# =============================================================================

@app.post("/api/fraud/detect")
@require_auth(required_role=UserRole.API_USER)
async def detect_fraud(
    request: Request,
    transaction: TransactionRequest,
    token_data: TokenData = None  # Injected by @require_auth
):
    """
    Protected endpoint - Fraud detection

    Requires: API_USER role or higher
    Scopes: read:fraud_detection, write:fraud_detection
    """
    # Sanitize inputs
    _merchant = InputSanitizer.sanitize_string(transaction.merchant, max_length=100)
    _location = InputSanitizer.sanitize_string(transaction.location, max_length=100)

    # Fraud detection logic would go here
    # For demo, return mock result

    risk_score = 0.35  # Mock risk score

    return {
        "transaction_id": transaction.transaction_id,
        "risk_score": risk_score,
        "risk_level": "LOW" if risk_score < 0.4 else "MEDIUM",
        "analyzed_by": token_data.username,
        "user_role": token_data.role.value,
        "message": "Transaction analyzed successfully"
    }


@app.get("/api/user/profile")
@require_auth(required_role=UserRole.API_USER)
async def get_profile(request: Request, token_data: TokenData = None):
    """
    Protected endpoint - Get user profile

    Requires: API_USER role or higher
    """
    return {
        "user_id": token_data.user_id,
        "username": token_data.username,
        "role": token_data.role.value,
        "tier": token_data.tier.value,
        "scopes": token_data.scopes
    }


@app.post("/api/user/api-key")
@require_auth(required_role=UserRole.API_USER)
async def generate_api_key(
    request: Request,
    key_name: str,
    expires_in_days: Optional[int] = 365,
    token_data: TokenData = None
):
    """
    Protected endpoint - Generate API key

    Requires: API_USER role or higher
    """
    auth_manager = request.app.state.auth_manager

    raw_key, api_key = auth_manager.generate_api_key(
        user_id=token_data.user_id,
        name=key_name,
        tier=token_data.tier,
        scopes=token_data.scopes,
        expires_in_days=expires_in_days
    )

    return APIKeyResponse(
        api_key=raw_key,
        key_id=api_key.key_id
    )


# =============================================================================
# ANALYST ENDPOINTS (Higher Privilege Required)
# =============================================================================

@app.get("/api/fraud/analytics")
@require_auth(required_role=UserRole.ANALYST)
async def get_fraud_analytics(request: Request, token_data: TokenData = None):
    """
    Protected endpoint - Fraud analytics

    Requires: ANALYST role or higher
    """
    return {
        "message": "Analytics data",
        "total_transactions": 10000,
        "fraud_detected": 150,
        "fraud_rate": 0.015,
        "accessed_by": token_data.username
    }


@app.post("/api/fraud/investigate")
@require_auth(required_role=UserRole.ANALYST)
async def investigate_transaction(
    request: Request,
    transaction_id: str,
    token_data: TokenData = None
):
    """
    Protected endpoint - Detailed fraud investigation

    Requires: ANALYST role or higher
    """
    # Investigation logic here
    return {
        "transaction_id": transaction_id,
        "investigation_status": "in_progress",
        "assigned_to": token_data.username,
        "priority": "high"
    }


# =============================================================================
# ADMIN ENDPOINTS (Highest Privilege Required)
# =============================================================================

@app.post("/api/admin/users")
@require_auth(required_role=UserRole.ADMIN)
async def create_user_admin(
    request: Request,
    user_data: UserCreateRequest,
    token_data: TokenData = None
):
    """
    Protected endpoint - Admin user creation

    Requires: ADMIN role
    """
    auth_manager = request.app.state.auth_manager

    user = auth_manager.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role,
        tier=user_data.tier
    )

    if not user:
        raise HTTPException(status_code=400, detail="User creation failed")

    return {
        "message": "User created by admin",
        "user_id": user.user_id,
        "created_by": token_data.username
    }


@app.get("/api/admin/users")
@require_auth(required_role=UserRole.ADMIN)
async def list_users(request: Request, token_data: TokenData = None):
    """
    Protected endpoint - List all users

    Requires: ADMIN role
    """
    auth_manager = request.app.state.auth_manager

    users = [
        {
            "user_id": u.user_id,
            "username": u.username,
            "email": u.email,
            "role": u.role.value,
            "tier": u.tier.value,
            "is_active": u.is_active
        }
        for u in auth_manager._users.values()
    ]

    return {
        "total_users": len(users),
        "users": users,
        "accessed_by": token_data.username
    }


@app.post("/api/admin/revoke-token")
@require_auth(required_role=UserRole.ADMIN)
async def revoke_token(
    request: Request,
    token: str,
    token_data: TokenData = None
):
    """
    Protected endpoint - Revoke user token

    Requires: ADMIN role
    """
    auth_manager = request.app.state.auth_manager
    success = auth_manager.revoke_token(token)

    return {
        "message": "Token revoked" if success else "Token revocation failed",
        "revoked_by": token_data.username
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error handler - prevents information leakage"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler - fail securely"""
    # Log the real error server-side
    print(f"Unhandled error: {exc}")

    # Return generic error to client
    return {
        "error": "An internal error occurred",
        "status_code": 500
    }


# =============================================================================
# STARTUP AND INITIALIZATION
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting Fraud Detection API...")

    # Initialize rate limiter Redis connection
    await rate_limiter.initialize()

    # Create default admin user if needed
    auth_manager = app.state.auth_manager
    if not auth_manager._users:
        admin_user = auth_manager.create_user(
            username="admin",
            email="admin@example.com",
            password="Admin@SecureP@ss123!",
            role=UserRole.ADMIN,
            tier=TierLevel.INTERNAL
        )
        print(f"Default admin user created: {admin_user.username}")

    print("Fraud Detection API started successfully")
    print("API Documentation: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Fraud Detection API...")

    # Close Redis connection
    if rate_limiter._redis_client:
        await rate_limiter._redis_client.close()

    print("Fraud Detection API shut down successfully")


# =============================================================================
# MAIN - Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Fraud Detection API - Secure Authentication Example")
    print("=" * 70)
    print("\nStarting server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nDefault admin credentials:")
    print("  Username: admin")
    print("  Password: Admin@SecureP@ss123!")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )