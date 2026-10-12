# Local setup: Bedrock, Gemini and YouTube

This setup stores generated media under `output/` and does not require an S3
bucket. Start by copying `.env.example` to `.env`.

## 1. Amazon Bedrock

For a local trial, open Amazon Bedrock in the AWS console and create a Bedrock
API key. Put the value in:

```dotenv
AWS_BEARER_TOKEN_BEDROCK=your_key
BEDROCK_ENABLED=1
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

Confirm that Amazon Nova Lite is available in the selected region. The app uses
the Bedrock Converse API. No bucket, S3 permission or S3 variable is needed.

If the console does not offer an API key for the account, use IAM credentials
instead (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally
`AWS_SESSION_TOKEN`). Keep `LOCAL_STORAGE_ONLY=1`; those credentials will still
never be used for S3.

## 2. Gemini fallback

Create a key in Google AI Studio and set:

```dotenv
GEMINI_API_KEY=your_key
```

The app tries Bedrock up to three times and then Gemini up to three times.

## 3. YouTube OAuth

YouTube uploads do not use a normal API key. They need an OAuth desktop client
and a refresh token for the YouTube channel.

1. Create or select a project in Google Cloud Console.
2. Enable **YouTube Data API v3** in APIs & Services > Library.
3. Configure the OAuth consent screen. For a personal test choose External,
   leave it in Testing, and add the Google account that owns the YouTube channel
   as a test user.
4. In Credentials, create an **OAuth client ID** with application type
   **Desktop app**.
5. Download its JSON and save it exactly as
   `secrets/youtube_client_secrets.json`.
6. From the repository root run:

```powershell
New-Item -ItemType Directory -Force secrets
# Move the downloaded Google JSON into secrets/youtube_client_secrets.json.
uv run --with google-auth-oauthlib --with google-auth python youtube_auth.py
```

7. Sign in to the channel-owning Google account and approve the upload scope.
   The script writes `secrets/youtube_token.json`.

Both JSON files are ignored by Git. Never commit or share them.

## 4. Start and verify

```powershell
Copy-Item .env.example .env
# Edit .env and fill AWS_BEARER_TOKEN_BEDROCK and GEMINI_API_KEY first.
docker compose up --build
```

Open `http://localhost:5175`. In Settings, Bedrock and Gemini should appear as
configured and YouTube as ready. The backend capability can also be checked:

```powershell
Invoke-RestMethod http://localhost:8000/api/social/capabilities
```

The response should only list `youtube`. Upload the first test Short as
`private` or `unlisted` before publishing anything publicly.

### OAuth testing caveat

While the OAuth consent app remains in Testing, Google can expire a test user's
authorization after seven days. If uploads later start failing with an OAuth
refresh error, run `youtube_auth.py` again. A personal test does not need to be
made public just to perform this local validation.
