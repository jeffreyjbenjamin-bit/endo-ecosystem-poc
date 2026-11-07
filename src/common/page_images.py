"""Preview images for article cards (the page's og:image / twitter:image).

New search results carry an image from the search provider when it has one
(see ai_search.normalize_results). For older results the Home page looks the
image up from the article page itself. Lookups are cached in the app state
directory, including misses, so each URL is fetched at most once.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import urljoin

import requests

from src.common.paths import state_path

CACHE_FILE = "page_images.json"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EvidenceGap/1.0; article preview)",
    "Accept": "text/html,application/xhtml+xml",
}
_MAX_BYTES = 400_000
_META_RE = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR_RE = re.compile(r'([a-zA-Z:_-]+)\s*=\s*("([^"]*)"|\'([^\']*)\')')
_IMAGE_KEYS = ("og:image:secure_url", "og:image", "og:image:url", "twitter:image")


def image_from_html(html: str, base_url: str) -> Optional[str]:
    """The best preview image declared in a page's <meta> tags."""
    found: Dict[str, str] = {}
    for tag in _META_RE.findall(html):
        attrs = {
            m.group(1).lower(): (m.group(3) if m.group(3) is not None else m.group(4))
            for m in _ATTR_RE.finditer(tag)
        }
        name = (attrs.get("property") or attrs.get("name") or "").lower()
        if name in _IMAGE_KEYS and attrs.get("content") and name not in found:
            found[name] = attrs["content"].strip()
    for key in _IMAGE_KEYS:
        if found.get(key):
            url = urljoin(base_url, found[key])
            if url.startswith("https://"):
                return url
    return None


def fetch_image(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        with requests.get(
            url, headers=_HEADERS, timeout=timeout, stream=True, allow_redirects=True
        ) as r:
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                return None
            chunks, size = [], 0
            for chunk in r.iter_content(16_384, decode_unicode=False):
                chunks.append(chunk)
                size += len(chunk)
                if size >= _MAX_BYTES or b"</head>" in chunk.lower():
                    break
            html = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
            return image_from_html(html, r.url)
    except Exception:
        return None


def _load_cache() -> Dict[str, str]:
    try:
        return json.loads(Path(state_path(CACHE_FILE)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def images_for(urls: Iterable[str], fetch: bool = True) -> Dict[str, str]:
    """{url: image url or ""} for the given article URLs."""
    urls = [u for u in dict.fromkeys(urls) if u and u.startswith("http")]
    cache = _load_cache()
    missing = [u for u in urls if u not in cache]
    if missing and fetch:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for u, img in zip(missing, pool.map(fetch_image, missing)):
                cache[u] = img or ""
        try:
            path = Path(state_path(CACHE_FILE))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
    return {u: cache.get(u, "") for u in urls}


def image_from_search_item(item: dict) -> Optional[str]:
    """Image the search provider returned with a result, if any."""
    pagemap = item.get("pagemap") or {}
    for key, field in (("cse_image", "src"), ("cse_thumbnail", "src")):
        for entry in pagemap.get(key) or []:
            if str(entry.get(field, "")).startswith("https://"):
                return entry[field]
    for tags in pagemap.get("metatags") or []:
        for key in _IMAGE_KEYS:
            if str(tags.get(key, "")).startswith("https://"):
                return tags[key]
    thumb = (item.get("image") or {}).get("thumbnailUrl")  # Bing
    if str(thumb or "").startswith("https://"):
        return thumb
    return None
