from fastapi import Header, HTTPException, status, Depends
import os

# default token (override with SVC_TOKEN env var)
STATIC_TOKEN = os.getenv("SVC_TOKEN", "super-secret-token")

async def require_token(authorization: str = Header(...)):
    # Expect header: "Bearer <token>"
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    token = authorization.split(" ", 1)[1].strip()
    if token != STATIC_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return token
