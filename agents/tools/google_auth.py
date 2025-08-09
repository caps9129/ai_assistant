# agents/tools/google_auth.py

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# --- MODIFIED: Import paths from the central config file ---
from config import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH

# Define your application's required permissions
SCOPES = [
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/calendar"
]


def get_credentials():
    """
    Handles user authentication using paths from the central config.
    """
    creds = None

    # Use the reliable path for the token file
    if GOOGLE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(GOOGLE_TOKEN_PATH), SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use the reliable path for the credentials file
            flow = InstalledAppFlow.from_client_secrets_file(
                str(GOOGLE_CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(GOOGLE_TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return creds


if __name__ == '__main__':
    print("Attempting to perform first-time Google authentication...")
    print("Your web browser should open for you to log in and grant permissions.")

    # Ensure you are running this from the project's root directory
    # where credentials.json is located.
    try:
        creds = get_credentials()
        if creds:
            print("\n✅ Authentication successful!")
            print("A 'token.json' file has been created in your project directory.")
            print("You do not need to run this script directly again.")
        else:
            print("\n❌ Authentication failed. No credentials returned.")
    except FileNotFoundError:
        print("\n❌ ERROR: 'credentials.json' not found.")
        print("Please ensure you have downloaded the file from Google Cloud Console and placed it in your project's root directory.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
