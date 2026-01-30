import httpx


def search_web(query: str, max_results: int = 5) -> list[dict]:
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": "1"},
        timeout=10.0,
    )
    data = response.json()

    results = []

    if data.get("Abstract"):
        results.append({
            "title": data.get("Heading", ""),
            "href": data.get("AbstractURL", ""),
            "body": data.get("Abstract", ""),
        })

    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and "FirstURL" in topic:
            results.append({
                "title": topic.get("Text", "")[:100],
                "href": topic.get("FirstURL", ""),
                "body": topic.get("Text", ""),
            })

    return results
