"""Parser of a public channel page `t.me/s/<channel>`: posts, the paging cursor and the channel header.

`?before=<id>` returns up to 20 posts with smaller ids and links the next older page with
`<link rel="prev" href="/s/<channel>?before=<min id>">`; the oldest page has no such link. Deleted
posts leave gaps in the ids. A reply carries a quote of the replied post in its own text block
(`js-message_reply_text`); only the post's own text (`js-message_text`) is the post. Author
signatures and the names of forwarded people are on the page but never read here.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

from lxml import html as lhtml

BASE = "https://t.me"
TME_LINK = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,31})(?:/(\d+))?", re.I)
_VIEWS = re.compile(r"^\s*([\d.]+)\s*([KM]?)\s*$")
_MEDIA = {
    "photo": "tgme_widget_message_photo_wrap",
    "video": "tgme_widget_message_video_wrap",
    "round_video": "tgme_widget_message_roundvideo_wrap",
    "document": "tgme_widget_message_document_wrap",
    "poll": "tgme_widget_message_poll",
    "link_preview": "tgme_widget_message_link_preview",
    "album": "tgme_widget_message_grouped_wrap",
    # Media the web preview does not render ("Please open Telegram to view this post").
    "not_in_preview": "message_media_not_supported_wrap",
}
# Job boards whose links are kept (host only) for cross-source dedup and aggregates.
JOB_HOSTS = frozenset(
    {
        "hh.ru",
        "career.habr.com",
        "u.habr.com",
        "getmatch.ru",
        "geekjob.ru",
        "superjob.ru",
        "trudvsem.ru",
        "rabota.ru",
        "zarplata.ru",
        "careerspace.app",
        "finder.work",
    }
)
_HABR_VACANCY = re.compile(r"^/vacancies/(\d+)")
_HABR_SHORT = re.compile(r"^/(\w+)$")


def _has_class(name: str) -> str:
    return f'contains(concat(" ", normalize-space(@class), " "), " {name} ")'


@dataclass
class Post:
    id: int
    date: str
    text: str
    links: list[str]
    forwarded_from: str | None  # channel username, only when the forward links to a channel post
    service: bool  # "channel created", "pinned" and the like, not a post
    forwarded: bool = False
    views: int | None = None  # as the preview shows it: "4.91K" is 4910
    edited: bool = False
    media: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Page:
    status: int
    exists: bool
    fetched_at: str
    title: str = ""
    subscribers: str | None = None
    posts: list[Post] = field(default_factory=list)
    prev_before: int | None = None


def page_url(channel: str, before: int | None = None, after: int | None = None) -> str:
    url = f"{BASE}/s/{channel}"
    if before is not None:
        url += f"?before={before}"
    elif after is not None:
        url += f"?after={after}"
    return url


def post_url(channel: str, post_id: int) -> str:
    return f"{BASE}/{channel}/{post_id}"


def parse_views(shown: str) -> int | None:
    match = _VIEWS.match(shown)
    if match is None:
        return None
    scale = {"": 1, "K": 1_000, "M": 1_000_000}[match.group(2)]
    return round(float(match.group(1)) * scale)


def job_links(links: list[str]) -> list[dict[str, Any]]:
    """Links to job boards: the host, plus the vacancy id for Habr Career (the dedup key with `habr`).

    `u.habr.com/<code>` is Habr's short link (a 301 to `career.habr.com/vacancies/<id>`); its code is
    kept so the id can be resolved later. Paths of other hosts are dropped: they may name a person.
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for link in links:
        try:
            parts = urlsplit(link.strip())
        except ValueError:
            continue
        host = (parts.hostname or "").lower().removeprefix("www.")
        if host.endswith(".hh.ru"):
            host = "hh.ru"
        if host not in JOB_HOSTS:
            continue
        entry: dict[str, Any] = {"host": host}
        if host == "career.habr.com" and (m := _HABR_VACANCY.match(parts.path)):
            entry["habr_vacancy_id"] = int(m.group(1))
        elif host == "u.habr.com" and (m := _HABR_SHORT.match(parts.path)):
            entry["habr_short"] = m.group(1)
        key = (host, str(entry.get("habr_vacancy_id", entry.get("habr_short", ""))))
        found.setdefault(key, entry)
    return [found[k] for k in sorted(found)]


def _post(m: Any) -> Post:
    dates = m.xpath('.//a[contains(@class,"tgme_widget_message_date")]/time/@datetime')
    text_nodes = m.xpath(f".//div[{_has_class('js-message_text')}]")
    text, links = "", []
    if text_nodes:
        for br in text_nodes[0].iter("br"):
            br.tail = "\n" + (br.tail or "")
        text = text_nodes[0].text_content()
        links = [a.get("href") for a in text_nodes[0].xpath(".//a[@href]")]
    forwarded = m.xpath(f".//*[{_has_class('tgme_widget_message_forwarded_from_name')}]")
    source = TME_LINK.search(forwarded[0].get("href") or "") if forwarded else None
    views = m.xpath(f".//span[{_has_class('tgme_widget_message_views')}]/text()")
    meta = "".join(m.xpath(f".//span[{_has_class('tgme_widget_message_meta')}]/text()"))
    return Post(
        id=int(m.get("data-post").rsplit("/", 1)[1]),
        date=dates[0] if dates else "",
        text=text,
        links=links,
        forwarded_from=source.group(1).lower() if source and source.group(2) else None,
        service="service_message" in (m.get("class") or ""),
        forwarded=bool(forwarded),
        views=parse_views(views[0]) if views else None,
        edited="edited" in meta,
        media=[kind for kind, cls in _MEDIA.items() if m.xpath(f".//*[{_has_class(cls)}]")],
    )


def parse(status: int, fetched_at: str, body: str) -> Page:
    # A missing channel answers with a redirect to t.me/<name>; a user or a group has no channel header.
    if status != 200:
        return Page(status, False, fetched_at)
    # The lxml stubs type xpath() results as a union of every XPath value; the paths here are fixed.
    doc: Any = lhtml.fromstring(body)
    if not doc.xpath('//div[contains(@class,"tgme_channel_info")]'):
        return Page(status, False, fetched_at)
    title = "".join(doc.xpath('//div[contains(@class,"tgme_channel_info_header_title")]//text()')).strip()
    subscribers = None
    for counter in doc.xpath('//div[contains(@class,"tgme_channel_info_counter")]'):
        if "".join(counter.xpath('.//span[@class="counter_type"]/text()')).startswith("subscriber"):
            subscribers = "".join(counter.xpath('.//span[@class="counter_value"]/text()'))
    posts = [_post(m) for m in doc.xpath('//div[contains(@class,"tgme_widget_message ") and @data-post]')]
    prev = doc.xpath('//link[@rel="prev"]/@href')
    before = re.search(r"before=(\d+)", prev[0]) if prev else None
    return Page(status, True, fetched_at, title, subscribers, posts, int(before.group(1)) if before else None)
