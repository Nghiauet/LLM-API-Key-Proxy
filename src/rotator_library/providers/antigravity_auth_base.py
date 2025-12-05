# src/rotator_library/providers/antigravity_auth_base.py

from .google_oauth_base import GoogleOAuthBase

class AntigravityAuthBase(GoogleOAuthBase):
    """
    Antigravity OAuth2 authentication implementation.
    
    Inherits all OAuth functionality from GoogleOAuthBase with Antigravity-specific configuration.
    Uses Antigravity's OAuth credentials and includes additional scopes for cclog and experimentsandconfigs.
    """
    
    CLIENT_ID = "REPLACE_WITH_ANTIGRAVITY_OAUTH_CLIENT_ID"
    CLIENT_SECRET = "REPLACE_WITH_ANTIGRAVITY_OAUTH_CLIENT_SECRET"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",  # Antigravity-specific
        "https://www.googleapis.com/auth/experimentsandconfigs",  # Antigravity-specific
    ]
    ENV_PREFIX = "ANTIGRAVITY"
    CALLBACK_PORT = 51121
    CALLBACK_PATH = "/oauthcallback"
