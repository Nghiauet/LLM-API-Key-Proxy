# src/rotator_library/providers/gemini_auth_base.py

from .google_oauth_base import GoogleOAuthBase

class GeminiAuthBase(GoogleOAuthBase):
    """
    Gemini CLI OAuth2 authentication implementation.
    
    Inherits all OAuth functionality from GoogleOAuthBase with Gemini-specific configuration.
    """
    
    CLIENT_ID = "REPLACE_WITH_GEMINI_CLI_OAUTH_CLIENT_ID"
    CLIENT_SECRET = "REPLACE_WITH_GEMINI_CLI_OAUTH_CLIENT_SECRET"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    ENV_PREFIX = "GEMINI_CLI"
    CALLBACK_PORT = 8085
    CALLBACK_PATH = "/oauth2callback"