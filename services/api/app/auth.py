"""Authentication: log a user in, mint their token, verify it per request.

Tokens are JWTs signed with a shared secret (HS256), carrying an expiry.
The POST /token route verifies a username and bcrypt-hashed password and
mints a token; every protected route then verifies the Bearer token.
scripts/mint_token.py remains as the operator-side mint (no login needed
when you hold the secret).
"""

from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from services.api.app.config import Settings

# fastapi helper on our server
# reads the Authorization: Bearer gsdbs header the client sent
# hands the token string
_bearer = HTTPBearer()  # rejects requests with no Authorization header


async def verify_login(user_store, username: str, password: str) -> bool:
    """Check credentials against the users table.

    Args:
        user_store: Where password hashes live (the users table adapter).
        username: The username presented at login.
        password: The plaintext password presented at login.

    Returns:
        True only when the user exists and the password verifies against
        the stored bcrypt hash. The plaintext is never stored anywhere.
    """
    stored_hash = await user_store.get_password_hash(username)
    if stored_hash is None:
        return False
    return bcrypt.checkpw(
        password.encode(), stored_hash.encode()
    )  # transform the str to bytes


def mint_access_token(subject: str) -> str:
    """Create a signed JWT for a verified user.

    Args:
        subject: Who the token is for (the ``sub`` claim).

    Returns:
        The encoded token, expiring after the configured lifetime.
    """
    settings = Settings()
    claims = {
        "sub": subject,
        "exp": datetime.now(UTC) + timedelta(hours=settings.jwt_expiry_hours),
    }
    return jwt.encode(
        claims,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Verify the request's Bearer JWT and return its claims.

    Args:
        credentials: The parsed Authorization header.

    Returns:
        The token's claims (e.g. ``sub``, ``exp``) once verified.

    Raises:
        HTTPException: 401 if the signature is wrong, the token is expired,
            or the token is malformed.
    """
    settings = Settings()  # env-read per verification: cheap, test-friendly
    try:
        return jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )  # returns a dictionary of claims the statements the token
        # says about who you are: the fields that mint access wrote
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid or expired token")
