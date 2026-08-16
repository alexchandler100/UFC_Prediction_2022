"""Reliable HTTP access for UFCStats.

UFCStats currently returns a small JavaScript proof-of-work page to new
sessions.  A bare ``requests.get`` receives HTTP 200, so status checks alone
cannot distinguish that challenge from the requested statistics page.  This
module centralizes session handling, retries, timeouts, and challenge solving
so every scraper has the same behavior and fails with an actionable error.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class UFCStatsError(RuntimeError):
    """Raised when UFCStats does not return a usable statistics page."""


class UFCStatsEventNotComplete(UFCStatsError):
    """Raised when a same-day/future card has not posted every fight result."""


@dataclass(frozen=True)
class RequestTimeout:
    connect: float = 10.0
    read: float = 30.0

    def as_tuple(self) -> tuple[float, float]:
        return self.connect, self.read


class UFCStatsClient:
    _CHALLENGE_MARKER = "Checking your browser"
    _NONCE_PATTERN = re.compile(r'var\s+nonce\s*=\s*"([^"]+)"')
    _TARGET_PATTERN = re.compile(r"target\s*=\s*new\s+Array\((\d+)\s*\+\s*1\)")

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: RequestTimeout | None = None,
        max_pow_attempts: int = 10_000_000,
        min_request_interval: float = 0.1,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout or RequestTimeout()
        self.max_pow_attempts = max_pow_attempts
        self.min_request_interval = max(0.0, min_request_interval)
        self._last_request_at = 0.0

        self.session.headers.setdefault(
            "User-Agent",
            (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
            ),
        )
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, *, expected_text: str | None = None) -> requests.Response:
        """Return a validated UFCStats response, satisfying its browser check.

        ``expected_text`` should be a stable page marker such as a table class.
        It prevents a changed error/challenge page with status 200 from being
        parsed and written as valid data.
        """

        response = self._get(url)
        if self._is_challenge(response):
            self._solve_challenge(response)
            response = self._get(url)

        if self._is_challenge(response):
            raise UFCStatsError(
                f"UFCStats browser challenge persisted after solving for {url}"
            )
        if expected_text and expected_text not in response.text:
            title = self._page_title(response.text)
            raise UFCStatsError(
                f"UFCStats response for {url} did not contain {expected_text!r}; "
                f"status={response.status_code}, bytes={len(response.content)}, "
                f"title={title!r}"
            )
        return response

    def _get(self, url: str) -> requests.Response:
        self._wait_for_rate_limit()
        try:
            response = self.session.get(url, timeout=self.timeout.as_tuple())
            self._last_request_at = time.monotonic()
            response.raise_for_status()
        except requests.RequestException as error:
            raise UFCStatsError(f"UFCStats request failed for {url}: {error}") from error
        return response

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _is_challenge(self, response: requests.Response) -> bool:
        return (
            self._CHALLENGE_MARKER in response.text
            and self._NONCE_PATTERN.search(response.text) is not None
        )

    def _solve_challenge(self, response: requests.Response) -> None:
        nonce_match = self._NONCE_PATTERN.search(response.text)
        target_match = self._TARGET_PATTERN.search(response.text)
        if nonce_match is None or target_match is None:
            raise UFCStatsError("Unrecognized UFCStats browser challenge format")

        nonce = nonce_match.group(1)
        zero_count = int(target_match.group(1))
        if zero_count < 1 or zero_count > 8:
            raise UFCStatsError(
                f"Refusing unexpected UFCStats proof-of-work difficulty {zero_count}"
            )

        prefix = "0" * zero_count
        solution = None
        for candidate in range(self.max_pow_attempts):
            digest = hashlib.sha256(f"{nonce}:{candidate}".encode("utf-8")).hexdigest()
            if digest.startswith(prefix):
                solution = candidate
                break
        if solution is None:
            raise UFCStatsError(
                "Could not solve UFCStats browser challenge within the safety limit"
            )

        challenge_url = urljoin(response.url, "/__c")
        self._wait_for_rate_limit()
        try:
            challenge_response = self.session.post(
                challenge_url,
                data={"nonce": nonce, "n": solution},
                timeout=self.timeout.as_tuple(),
            )
            self._last_request_at = time.monotonic()
            challenge_response.raise_for_status()
        except requests.RequestException as error:
            raise UFCStatsError(
                f"UFCStats challenge submission failed for {challenge_url}: {error}"
            ) from error

    @staticmethod
    def _page_title(html: str) -> str | None:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else None


ufcstats_client = UFCStatsClient()
