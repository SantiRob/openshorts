"""Direct social publishing adapters.

Credentials are deliberately read from the server environment.  Request models
must never accept access tokens: this module is also used by the HTTP API, where
echoing a vendor exception could otherwise disclose one in a response or log.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import httpx

from youtube_utils import YouTubeQuotaExceeded, YouTubeUploadError, YouTubeUploader


logger = logging.getLogger(__name__)


class SocialPublishError(RuntimeError):
    """A safe, user-displayable publishing failure."""

    def __init__(self, platform: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.platform = platform
        self.retryable = retryable


@dataclass(frozen=True)
class PublishInput:
    video_path: str
    title: str
    description: str = ""
    tags: tuple[str, ...] = ()
    scheduled_at: Optional[datetime] = None
    public_video_url: Optional[str] = None
    thumbnail_path: Optional[str] = None


def _positive_int(value: Optional[str], default: int) -> int:
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _safe_vendor_message(response: httpx.Response) -> str:
    """Return useful vendor context without reflecting tokens or request URLs."""
    try:
        data = response.json()
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "vendor request failed")[:300]
        if isinstance(error, str):
            return error[:300]
    except Exception:
        pass
    return f"vendor request failed with HTTP {response.status_code}"


class YouTubeDirectPublisher:
    platform = "youtube"

    def __init__(self, token_path: str, attempts: int = 3, uploader_factory=YouTubeUploader):
        self.token_path = token_path
        self.attempts = attempts
        self._uploader_factory = uploader_factory

    def publish(self, item: PublishInput) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.attempts + 1):
            try:
                uploader = self._uploader_factory(self.token_path)
                video_id, url = uploader.upload_video(
                    item.video_path,
                    item.title,
                    item.description,
                    list(item.tags),
                    item.scheduled_at,
                    False,
                )
                thumbnail_warning = None
                if item.thumbnail_path:
                    try:
                        uploader.set_thumbnail(video_id, item.thumbnail_path)
                    except YouTubeUploadError as exc:
                        # The video already exists at this point. Retrying the
                        # whole transaction would create duplicate uploads.
                        thumbnail_warning = "Video published, but the custom thumbnail failed"
                        logger.warning("YouTube thumbnail failed for %s: %s", video_id, exc)
                result = {"platform": self.platform, "id": video_id, "url": url,
                          "status": "scheduled" if item.scheduled_at else "published"}
                if thumbnail_warning:
                    result["warning"] = thumbnail_warning
                return result
            except YouTubeQuotaExceeded as exc:
                raise SocialPublishError(self.platform, "YouTube upload quota exceeded") from exc
            except (YouTubeUploadError, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        logger.warning("YouTube publishing failed after %s attempts: %s", self.attempts,
                       type(last_error).__name__ if last_error else "unknown")
        raise SocialPublishError(
            self.platform,
            f"YouTube upload failed after {self.attempts} attempts",
            retryable=True,
        ) from last_error


class MetaGraphPublisher:
    """Facebook Page Reels and optional Instagram Reels via Meta Graph API."""

    def __init__(
        self,
        page_id: str,
        access_token: str,
        *,
        instagram_user_id: Optional[str] = None,
        graph_version: str = "v23.0",
        attempts: int = 3,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.page_id = page_id
        self.access_token = access_token
        self.instagram_user_id = instagram_user_id
        self.graph_version = graph_version
        self.attempts = attempts
        self.client_factory = client_factory
        self.sleep = sleep

    @property
    def graph_base(self) -> str:
        return f"https://graph.facebook.com/{self.graph_version}"

    def _request(self, client: httpx.Client, method: str, url: str, **kwargs) -> httpx.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = client.request(method, url, **kwargs)
                if response.status_code < 400:
                    return response
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable:
                    raise SocialPublishError("meta", _safe_vendor_message(response))
                last_error = RuntimeError(_safe_vendor_message(response))
            except SocialPublishError:
                raise
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < self.attempts:
                retry_after = 0
                if "response" in locals():
                    try:
                        retry_after = int(response.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0
                self.sleep(max(retry_after, min(2 ** (attempt - 1), 4)))
        raise SocialPublishError(
            "meta", f"Meta request failed after {self.attempts} attempts", retryable=True
        ) from last_error

    def publish_facebook(self, item: PublishInput) -> dict:
        if item.scheduled_at:
            raise SocialPublishError("facebook", "Direct Facebook Reel scheduling is not supported")
        path = Path(item.video_path)
        if not path.is_file():
            raise SocialPublishError("facebook", "Video file not found")

        endpoint = f"{self.graph_base}/{self.page_id}/video_reels"
        auth = {"Authorization": f"OAuth {self.access_token}"}
        with self.client_factory(timeout=120.0) as client:
            started = self._request(client, "POST", endpoint, params={
                "access_token": self.access_token,
                "upload_phase": "start",
            }).json()
            video_id, upload_url = started.get("video_id"), started.get("upload_url")
            if not video_id or not upload_url:
                raise SocialPublishError("facebook", "Meta did not create a Reel upload session")
            with path.open("rb") as video:
                self._request(client, "POST", upload_url, headers={
                    **auth,
                    "offset": "0",
                    "file_size": str(path.stat().st_size),
                    "Content-Type": "application/octet-stream",
                }, content=video)
            self._request(client, "POST", endpoint, params={
                "access_token": self.access_token,
                "video_id": video_id,
                "upload_phase": "finish",
                "video_state": "PUBLISHED",
                "description": item.description or item.title,
            })
        return {"platform": "facebook", "id": str(video_id),
                "url": f"https://www.facebook.com/reel/{video_id}", "status": "published"}

    def publish_instagram(self, item: PublishInput) -> dict:
        """Publish when a public media URL and an IG professional account exist."""
        if not self.instagram_user_id:
            raise SocialPublishError("instagram", "Instagram account is not configured")
        if not item.public_video_url:
            raise SocialPublishError("instagram", "Instagram requires a public video URL")
        if item.scheduled_at:
            raise SocialPublishError("instagram", "Direct Instagram Reel scheduling is not supported")

        with self.client_factory(timeout=60.0) as client:
            created = self._request(
                client, "POST", f"{self.graph_base}/{self.instagram_user_id}/media",
                params={"access_token": self.access_token, "media_type": "REELS",
                        "video_url": item.public_video_url,
                        "caption": item.description or item.title},
            ).json()
            container_id = created.get("id")
            if not container_id:
                raise SocialPublishError("instagram", "Meta did not create an Instagram container")

            # Container processing is asynchronous. Keep polling bounded so a
            # request cannot occupy a worker indefinitely.
            ready = False
            for _ in range(12):
                status = self._request(
                    client, "GET", f"{self.graph_base}/{container_id}",
                    params={"access_token": self.access_token, "fields": "status_code"},
                ).json().get("status_code")
                if status == "FINISHED":
                    ready = True
                    break
                if status in {"ERROR", "EXPIRED"}:
                    raise SocialPublishError("instagram", "Instagram rejected the media container")
                self.sleep(5)
            if not ready:
                raise SocialPublishError("instagram", "Instagram media processing timed out", retryable=True)
            published = self._request(
                client, "POST", f"{self.graph_base}/{self.instagram_user_id}/media_publish",
                params={"access_token": self.access_token, "creation_id": container_id},
            ).json()
        media_id = published.get("id")
        return {"platform": "instagram", "id": str(media_id), "url": None,
                "status": "published"}


class SocialPublishingService:
    """Platform registry used by HTTP endpoints and background workers."""

    aliases = {"meta": "facebook", "facebook_reels": "facebook",
               "instagram_reels": "instagram"}

    def __init__(self, publishers: Dict[str, object], public_media_base_url: str = "",
                 enabled_platforms: Optional[Iterable[str]] = None):
        self.publishers = publishers
        self.public_media_base_url = public_media_base_url.rstrip("/")
        self.enabled_platforms = set(enabled_platforms or ("youtube", "facebook", "instagram"))

    @classmethod
    def from_env(cls) -> "SocialPublishingService":
        attempts = _positive_int(os.getenv("SOCIAL_PUBLISH_MAX_ATTEMPTS"), 3)
        enabled = {
            item.strip().lower()
            for item in os.getenv("SOCIAL_PLATFORMS", "youtube,facebook,instagram").split(",")
            if item.strip()
        }
        publishers: Dict[str, object] = {}
        youtube_token = os.getenv("YOUTUBE_TOKEN_PATH", "").strip()
        if youtube_token and os.path.isfile(youtube_token) and "youtube" in enabled:
            publishers["youtube"] = YouTubeDirectPublisher(youtube_token, attempts)
        page_id = os.getenv("META_PAGE_ID", "").strip()
        meta_token = os.getenv("META_PAGE_ACCESS_TOKEN", "").strip()
        if page_id and meta_token and ({"facebook", "instagram"} & enabled):
            meta = MetaGraphPublisher(
                page_id, meta_token,
                instagram_user_id=os.getenv("META_INSTAGRAM_USER_ID", "").strip() or None,
                graph_version=os.getenv("META_GRAPH_VERSION", "v23.0").strip(),
                attempts=attempts,
            )
            if "facebook" in enabled:
                publishers["facebook"] = meta
            if "instagram" in enabled and meta.instagram_user_id:
                publishers["instagram"] = meta
        return cls(
            publishers,
            os.getenv("SOCIAL_PUBLIC_MEDIA_BASE_URL", ""),
            enabled_platforms=enabled,
        )

    def capabilities(self) -> dict:
        all_details = {
            "youtube": {"configured": "youtube" in self.publishers, "scheduling": True},
            "facebook": {"configured": "facebook" in self.publishers, "scheduling": False},
            "instagram": {
                "configured": "instagram" in self.publishers and bool(self.public_media_base_url),
                "scheduling": False,
            },
        }
        details = {
            name: info for name, info in all_details.items()
            if name in self.enabled_platforms
        }
        return {"backend": "direct", "platforms": details,
                "configured": [name for name, info in details.items() if info["configured"]]}

    def publish(self, platforms: Iterable[str], item: PublishInput, relative_video_url: str = "") -> dict:
        results = []
        for requested in dict.fromkeys(platforms):
            platform = self.aliases.get(requested.lower(), requested.lower())
            publisher = self.publishers.get(platform)
            if not publisher:
                results.append({"platform": platform, "ok": False,
                                "error": f"{platform} is not configured"})
                continue
            platform_item = item
            if platform == "instagram" and not item.public_video_url and self.public_media_base_url:
                platform_item = PublishInput(
                    **{**item.__dict__, "public_video_url":
                       f"{self.public_media_base_url}/{relative_video_url.lstrip('/')}"}
                )
            try:
                if platform == "youtube":
                    value = publisher.publish(platform_item)
                elif platform == "facebook":
                    value = publisher.publish_facebook(platform_item)
                else:
                    value = publisher.publish_instagram(platform_item)
                results.append({"ok": True, **value})
            except SocialPublishError as exc:
                results.append({"platform": platform, "ok": False, "error": str(exc),
                                "retryable": exc.retryable})
        return {"success": bool(results) and all(row["ok"] for row in results), "results": results}
