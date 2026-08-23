"""Login: exchange a username and password for a Bearer JWT."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from services.api.app.auth import mint_access_token, verify_login
from services.api.app.stores.postgres_users import PostgresUserStore

router = APIRouter()


def get_user_store(request: Request) -> PostgresUserStore:
    """Dependency: the users table adapter built at startup."""
    return request.app.state.user_store


@router.post("/token")
async def token(
    form: OAuth2PasswordRequestForm = Depends(),
    users: PostgresUserStore = Depends(get_user_store),
) -> dict:
    """Log in with username and password; receive a Bearer JWT."""
    if not await verify_login(users, form.username, form.password):
        raise HTTPException(status_code=401, detail="wrong username or password")
    return {
        "access_token": mint_access_token(subject=form.username),
        "token_type": "bearer",
    }
