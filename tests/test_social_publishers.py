from pathlib import Path

import httpx

from social_publishers import (
    MetaGraphPublisher,
    PublishInput,
    SocialPublishingService,
    YouTubeDirectPublisher,
)
from youtube_utils import YouTubeUploadError, YouTubeUploader


class _FakeClient:
    def __init__(self, responses, calls, **kwargs):
        self.responses = responses
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def request(self, method, url, **kwargs):
        content = kwargs.get("content")
        if hasattr(content, "read"):
            content.read()
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def _response(status, data, headers=None):
    return httpx.Response(status, json=data, headers=headers or {})


def test_facebook_reel_uses_three_phase_graph_protocol_and_retries(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []
    responses = [
        _response(503, {"error": {"message": "temporary"}}),
        _response(200, {"video_id": "123", "upload_url": "https://upload.facebook.test/123"}),
        _response(200, {"success": True}),
        _response(200, {"success": True}),
    ]
    publisher = MetaGraphPublisher(
        "page-id", "server-secret", attempts=3,
        client_factory=lambda **kwargs: _FakeClient(responses, calls, **kwargs),
        sleep=lambda _: None,
    )

    result = publisher.publish_facebook(PublishInput(str(video), "title", "caption"))

    assert result["id"] == "123"
    assert len(calls) == 4
    assert calls[0][2]["params"]["upload_phase"] == "start"
    assert calls[2][1] == "https://upload.facebook.test/123"
    assert calls[3][2]["params"]["upload_phase"] == "finish"
    assert "server-secret" not in str(result)


def test_youtube_retries_up_to_configured_attempts(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    attempts = []

    class FakeUploader:
        def __init__(self, token_path):
            assert token_path == "secret-token.json"

        def upload_video(self, *args):
            attempts.append(1)
            if len(attempts) < 3:
                raise YouTubeUploadError("temporary")
            return "yt-id", "https://youtu.be/yt-id"

    publisher = YouTubeDirectPublisher(
        "secret-token.json", attempts=3, uploader_factory=FakeUploader
    )
    publisher_result = publisher.publish(PublishInput(str(video), "title"))
    assert publisher_result["id"] == "yt-id"
    assert len(attempts) == 3


def test_youtube_applies_thumbnail_in_same_direct_flow(tmp_path):
    video = tmp_path / "clip.mp4"
    thumbnail = tmp_path / "thumb.jpg"
    video.write_bytes(b"video")
    thumbnail.write_bytes(b"image")
    calls = []

    class FakeUploader:
        def __init__(self, token_path):
            calls.append(("token", token_path))

        def upload_video(self, *args):
            return "yt-id", "https://youtu.be/yt-id"

        def set_thumbnail(self, video_id, path):
            calls.append(("thumbnail", video_id, path))

    publisher = YouTubeDirectPublisher(
        "secret-token.json", attempts=3, uploader_factory=FakeUploader
    )
    result = publisher.publish(PublishInput(
        str(video), "title", thumbnail_path=str(thumbnail)
    ))

    assert result["id"] == "yt-id"
    assert ("thumbnail", "yt-id", str(thumbnail)) in calls


def test_capabilities_report_configuration_not_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCIAL_PLATFORMS", "youtube,facebook,instagram")
    youtube_token = tmp_path / "youtube.json"
    youtube_token.write_text("{}")
    monkeypatch.setenv("YOUTUBE_TOKEN_PATH", str(youtube_token))
    monkeypatch.setenv("META_PAGE_ID", "page-id")
    monkeypatch.setenv("META_PAGE_ACCESS_TOKEN", "very-secret")
    monkeypatch.setenv("META_INSTAGRAM_USER_ID", "ig-id")
    monkeypatch.delenv("SOCIAL_PUBLIC_MEDIA_BASE_URL", raising=False)

    result = SocialPublishingService.from_env().capabilities()

    assert result["backend"] == "direct"
    assert result["platforms"]["youtube"] == {"configured": True, "scheduling": True}
    assert result["platforms"]["facebook"] == {"configured": True, "scheduling": False}
    # Graph can be configured, but Instagram ingestion still needs a public URL.
    assert result["platforms"]["instagram"]["configured"] is False
    assert "very-secret" not in str(result)


def test_social_platform_allowlist_exposes_only_youtube(monkeypatch, tmp_path):
    youtube_token = tmp_path / "youtube.json"
    youtube_token.write_text("{}")
    monkeypatch.setenv("SOCIAL_PLATFORMS", "youtube")
    monkeypatch.setenv("YOUTUBE_TOKEN_PATH", str(youtube_token))
    monkeypatch.setenv("META_PAGE_ID", "page-id")
    monkeypatch.setenv("META_PAGE_ACCESS_TOKEN", "very-secret")

    result = SocialPublishingService.from_env().capabilities()

    assert result["configured"] == ["youtube"]
    assert list(result["platforms"]) == ["youtube"]


def test_youtube_uses_official_upload_endpoint():
    assert YouTubeUploader.RESUMABLE_UPLOAD_ENDPOINT.startswith(
        "https://www.googleapis.com/upload/youtube/v3/videos"
    )


def test_unconfigured_platform_fails_independently():
    result = SocialPublishingService({}).publish(
        ["youtube", "facebook"], PublishInput("missing.mp4", "title")
    )
    assert result["success"] is False
    assert [row["platform"] for row in result["results"]] == ["youtube", "facebook"]
    assert all(row["ok"] is False for row in result["results"])
