import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.config import settings
from src.models import DocumentChunk
from src.embeddings import generate_embeddings


def discover_links(url: str, depth: int = 1, same_domain: bool = True) -> list[dict[str, str]]:
    """Crawl a URL and discover linked pages up to a given depth."""
    parsed_base = urlparse(url)
    visited = set()
    to_visit = [(url, 0)]
    links = []

    while to_visit:
        current_url, current_depth = to_visit.pop(0)

        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {current_url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else current_url

        if current_depth > 0:
            links.append({"title": title, "url": current_url})

        if current_depth < depth:
            for a_tag in soup.select("a[href]"):
                href = a_tag.get("href", "")
                if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                    continue

                full_url = urljoin(current_url, href)
                full_url = full_url.split("#")[0]
                parsed = urlparse(full_url)

                if same_domain and parsed.netloc != parsed_base.netloc:
                    continue

                if full_url not in visited:
                    to_visit.append((full_url, current_depth + 1))

    return links


def fetch_page_content(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    main = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("div#main-col-body")
        or soup.select_one("div.content")
        or soup.body
        or soup
    )
    for tag in main.select("nav, header, footer, script, style, aside"):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    return text.replace("\x00", "")


def chunk_text(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len + (1 if current_sentences else 0) > settings.chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

                # Build overlap: take sentences from the end of the current chunk
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + (1 if overlap_sentences else 0) > settings.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + (1 if len(overlap_sentences) > 1 else 0)

                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences) + max(0, len(current_sentences) - 1)
            else:
                current_sentences = []
                current_len = 0

        current_sentences.append(sentence)
        current_len += sentence_len + (1 if len(current_sentences) > 1 else 0)

    if current_sentences:
        chunk = " ".join(current_sentences)
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def _fetch_and_chunk(link: dict[str, str]) -> dict | None:
    try:
        content = fetch_page_content(link["url"])
        chunks = chunk_text(content)
        if not chunks:
            return None
        return {"link": link, "chunks": chunks}
    except Exception as e:
        print(f"Failed to fetch {link['title']}: {e}")
        return None


def ingest_url(db: Session, url: str, depth: int = 1, clear: bool = False) -> dict[str, int]:
    """Ingest documentation starting from a URL.

    Args:
        url: Starting URL to crawl
        depth: How many levels of links to follow (0 = just this page, 1 = this page + linked pages, etc.)
        clear: If True, clear all existing chunks before ingesting
    """
    links = discover_links(url, depth=depth)

    if not links:
        content = fetch_page_content(url)
        soup = BeautifulSoup(requests.get(url, timeout=30).text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        links = [{"title": title, "url": url}]

    print(f"Found {len(links)} pages to ingest from {url}")

    if clear:
        db.query(DocumentChunk).delete()
        db.commit()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_and_chunk, link): link for link in links}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    total_chunks = 0
    total_pages = 0

    for result in results:
        link = result["link"]
        chunks = result["chunks"]
        embeddings = generate_embeddings(chunks)

        for chunk_text_content, embedding in zip(chunks, embeddings):
            doc = DocumentChunk(
                source_url=link["url"],
                resource_type=link["title"],
                title=link["title"],
                content=chunk_text_content,
                embedding=embedding,
            )
            db.add(doc)

        db.commit()
        total_chunks += len(chunks)
        total_pages += 1

    return {
        "pages_ingested": total_pages,
        "total_chunks": total_chunks,
        "pages_found": len(links),
    }
