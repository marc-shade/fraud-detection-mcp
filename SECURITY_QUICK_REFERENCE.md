# Security Layer - Quick Reference Guide

## Installation

```bash
# Dependencies already in requirements.txt
pip install -r requirements.txt

# Start Redis (required for rate limiting)
redis-server

# Test the security layer
python security.py
```

---

## Basic Setup (5 minutes)

```python
from security import AuthManager, RateLimiter, SecurityMiddleware

# 1. Initialize components
auth_manager = AuthManager()
rate_limiter = RateLimiter()
security_middleware = SecurityMiddleware(auth_manager, rate_limiter)

# 2. Initialize Redis connection
await rate_limiter.initialize()

# Done! Ready to use
```

---

## Common Tasks

### Create User
```python
user = auth_manager.create_user(
    username="john_doe",
    email="john@example.com",
    password="SecureP@ssw0rd123!",  # Min 12 chars, upper+lower+digit+special
    role=UserRole.API_USER,
    tier=TierLevel.FREE
)
```

### Authenticate User
```python
user = auth_manager.authenticate_user("john_doe", "SecureP@ssw0rd123!")
if user:
    print(f"Welcome {user.username}!")
```

### Generate JWT Token
```python
from security import TokenData

token_data = TokenData(
    user_id=user.user_id,
    username=user.username,
    role=user.role,
    tier=user.tier,
    scopes=["read:fraud", "write:fraud"]
)

token = auth_manager.create_access_token(token_data)
# Use token in Authorization header: Bearer <token>
```

### Verify JWT Token
```python
token_data = auth_manager.verify_token(token)
if token_data:
    print(f"Valid token for {token_data.username}")
```

### Generate API Key
```python
raw_key, api_key = auth_manager.generate_api_key(
    user_id=user.user_id,
    name="Production Key",
    tier=TierLevel.PAID,
    expires_in_days=365
)
# IMPORTANT: Store raw_key - won't be shown again!
```

### Verify API Key
```python
api_key = auth_manager.verify_api_key(raw_key)
if api_key:
    print(f"Valid key: {api_key.name}")
```

### Check Rate Limit
```python
allowed, info = await rate_limiter.check_rate_limit(
    identifier=user.user_id,
    tier=user.tier
)

if not allowed:
    print(f"Rate limit exceeded. Reset in {info['reset']} seconds")
```

---

## FastAPI Integration

### Basic Setup
```python
from fastapi import FastAPI, Request
from security import (
    AuthManager, RateLimiter, SecurityMiddleware,
    require_auth, UserRole
)

app = FastAPI()

# Initialize
auth_manager = AuthManager()
rate_limiter = RateLimiter()
security_middleware = SecurityMiddleware(auth_manager, rate_limiter)

# Store in app state
app.state.security_middleware = security_middleware
app.state.auth_manager = auth_manager

# Add middleware
@app.middleware("http")
async def security_handler(request: Request, call_next):
    return await security_middleware(request, call_next)

# Initialize on startup
@app.on_event("startup")
async def startup():
    await rate_limiter.initialize()
```

### Protect Endpoint
```python
@app.post("/api/fraud/detect")
@require_auth(required_role=UserRole.API_USER)
async def detect_fraud(
    request: Request,
    data: dict,
    token_data = None  # Injected by decorator
):
    # Your code here
    return {"result": "success"}
```

### Login Endpoint
```python
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
async def login(request: Request, credentials: LoginRequest):
    auth = request.app.state.auth_manager

    user = auth.authenticate_user(
        credentials.username,
        credentials.password
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token_data = TokenData(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        tier=user.tier
    )

    token = auth.create_access_token(token_data)

    return {"access_token": token, "token_type": "bearer"}
```

---

## Role Hierarchy

```
ADMIN (Level 3)          - Full system access
  └─ ANALYST (Level 2)   - Analytics + detection
      └─ API_USER (Level 1) - Basic fraud detection
          └─ READ_ONLY (Level 0) - Read-only access
```

### Usage
```python
# Admin endpoint
@require_auth(required_role=UserRole.ADMIN)
async def admin_function():
    pass

# Analyst endpoint
@require_auth(required_role=UserRole.ANALYST)
async def analyst_function():
    pass

# Any authenticated user
@require_auth(required_role=UserRole.API_USER)
async def user_function():
    pass
```

---

## Rate Limit Tiers

| Tier | Limit | Use Case |
|------|-------|----------|
| FREE | 10/min | Free tier users |
| PAID | 1000/min | Standard customers |
| ENTERPRISE | 10000/min | Large organizations |
| INTERNAL | 100000/min | Internal services |

---

## Password Requirements

✓ Minimum 12 characters
✓ At least 1 uppercase letter
✓ At least 1 lowercase letter
✓ At least 1 digit
✓ At least 1 special character (!@#$%^&*(),.?":{}|<>)
✗ Cannot be common password (password, 12345678, etc.)

**Valid Examples:**
- `SecureP@ssw0rd123!`
- `MyStr0ng!P@ssphrase`
- `C0mpl3x&Secure#Pass`

**Invalid Examples:**
- `weak` (too short)
- `password123` (no uppercase, no special)
- `Password` (too short, no digit, no special)

---

## Authentication Methods

### 1. JWT Bearer Token (Recommended)
```bash
curl -H "Authorization: Bearer <token>" https://api.example.com/endpoint
```

### 2. API Key Header
```bash
curl -H "X-API-Key: fd_<key>" https://api.example.com/endpoint
```

### 3. API Key Query (Not Recommended)
```bash
curl "https://api.example.com/endpoint?api_key=fd_<key>"
```

---

## Security Headers (Automatic)

All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Cause |
|------|---------|-------|
| 401 | Unauthorized | Invalid/missing token |
| 403 | Forbidden | Insufficient permissions |
| 429 | Too Many Requests | Rate limit exceeded |
| 400 | Bad Request | Invalid input |
| 500 | Internal Error | Server error |

### Example Error Response
```json
{
  "error": "Authentication required",
  "status_code": 401
}
```

---

## Testing

### Run All Tests
```bash
pytest tests/test_security.py -v
```

### Run Specific Test
```bash
pytest tests/test_security.py::TestAuthManager::test_create_user_valid -v
```

### Test Coverage
```bash
pytest tests/test_security.py --cov=security --cov-report=html
```

---

## Environment Variables

### Required for Production
```bash
ENVIRONMENT=production
JWT_SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
REDIS_URL=redis://your-redis-server:6379
```

### Optional
```bash
JWT_ALGORITHM=HS256  # Default
ACCESS_TOKEN_EXPIRE_MINUTES=30  # Default
DEBUG=False  # Default for production
LOG_LEVEL=WARNING  # Default for production
```

### Generate Secret
```python
import secrets
print(secrets.token_urlsafe(32))
```

---

## Common Issues & Solutions

### Issue: "Rate limiting unavailable - Redis not connected"
**Solution:** Start Redis server: `redis-server`

### Issue: "Password validation failed"
**Solution:** Ensure password meets all requirements (12+ chars, upper+lower+digit+special)

### Issue: "JWT verification failed"
**Solution:** Check JWT_SECRET_KEY is set and consistent

### Issue: "Account locked"
**Solution:** Wait 30 minutes or reset failed attempts in database

### Issue: "Token expired"
**Solution:** Generate new token using refresh flow

---

## Security Checklist

### Development
- [ ] Set up Redis locally
- [ ] Configure JWT secret
- [ ] Test authentication flow
- [ ] Test rate limiting
- [ ] Run security tests

### Production
- [ ] Set ENVIRONMENT=production
- [ ] Use strong JWT secret (32+ bytes)
- [ ] Configure production Redis
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring
- [ ] Review security logs
- [ ] Configure CORS properly

---

## Performance Tips

1. **Cache user lookups** - Store authenticated user in request state
2. **Use API keys for machines** - JWT for users, API keys for services
3. **Monitor Redis** - Ensure Redis is healthy for rate limiting
4. **Adjust token expiry** - Balance security vs UX
5. **Use connection pooling** - For Redis connections

---

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Token Contents
```python
import jwt
payload = jwt.decode(token, options={"verify_signature": False})
print(payload)
```

### Test Rate Limit
```python
# Check current count
allowed, info = await rate_limiter.check_rate_limit(user_id, tier)
print(f"Remaining: {info['remaining']}/{info['limit']}")

# Reset if needed
await rate_limiter.reset_rate_limit(user_id)
```

---

## References

- Full Documentation: `SECURITY_AUDIT.md`
- Implementation Summary: `SECURITY_IMPLEMENTATION_SUMMARY.md`
- Integration Example: `examples/security_integration_example.py`
- Test Suite: `tests/test_security.py`
- Main Module: `security.py`

---

## Quick Commands

```bash
# Test security layer
python security.py

# Run tests
pytest tests/test_security.py -v

# Start example API
python examples/security_integration_example.py

# Check Redis
redis-cli ping

# Generate secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Security scan
bandit -r . -ll

# Dependency check
safety check
```

---

**Last Updated:** 2025-09-29
**Version:** 1.0.0