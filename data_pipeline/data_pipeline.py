from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
DB_PATH = DATA_PIPELINE_DIR / "books_catalog.db"
CSV_PATH = DATA_PIPELINE_DIR / "books_catalog.csv"
BASE_URL = "http://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50


def fetch_html(url: str) -> str:
    """Fetch a page and raise a clear error if the site is unreachable."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def extract_category_links() -> list[tuple[str, str]]:
    """Extract the first three product categories from the home page."""
    soup = BeautifulSoup(fetch_html(BASE_URL), "html.parser")
    category_links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.select("ul.nav-list li a"):
        href = anchor.get("href")
        if not href:
            continue
        category_name = anchor.get_text(" ", strip=True)
        if category_name.lower() == "books":
            continue
        full_url = urljoin(BASE_URL, href)
        if full_url not in seen:
            seen.add(full_url)
            category_links.append((category_name, full_url))

    return category_links[:3]


def parse_star_rating(star_text: str) -> int | None:
    """Convert text such as 'Three' or 'Four' into an integer rating 1-5."""
    mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    normalized = (star_text or "").strip().lower()
    return mapping.get(normalized)


def parse_price(price_text: str) -> float | None:
    """Strip the GBP symbol and convert the text price to a float."""
    cleaned = re.sub(r"[^0-9.\-]", "", price_text or "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_availability(availability_text: str) -> bool:
    """Convert the availability text to a boolean in_stock indicator."""
    return "in stock" in (availability_text or "").lower()


def scrape_category(category_name: str, category_url: str) -> list[dict[str, object]]:
    """Scrape product listings from one category and stop cleanly when a page is missing."""
    collected_books: list[dict[str, object]] = []
    page_number = 1

    while page_number <= 2:
        current_url = category_url
        if page_number > 1:
            if "index.html" in current_url:
                current_url = current_url.replace("index.html", f"page-{page_number}.html")
            else:
                current_url = current_url.rstrip("/") + f"/page-{page_number}.html"

        try:
            soup = BeautifulSoup(fetch_html(current_url), "html.parser")
        except requests.HTTPError:
            break

        product_cards = soup.select("article.product_pod")
        if not product_cards:
            break

        for card in product_cards:
            title_tag = card.h3.a
            title = title_tag.get("title") if title_tag else None
            if not title:
                continue

            price_text = card.select_one("p.price_color")
            price_gbp = parse_price(price_text.get_text(" ", strip=True) if price_text else "")

            rating_tag = card.select_one("p.star-rating")
            rating_text = rating_tag.get("class", [""])[-1] if rating_tag else ""
            rating = parse_star_rating(rating_text)

            availability_tag = card.select_one("p.availability")
            availability_text = availability_tag.get_text(" ", strip=True) if availability_tag else ""
            in_stock = parse_availability(availability_text)

            if title and price_gbp is not None and rating is not None:
                collected_books.append(
                    {
                        "title": title,
                        "price_gbp": price_gbp,
                        "rating": rating,
                        "in_stock": in_stock,
                        "availability_text": availability_text,
                        "category": category_name,
                    }
                )

        page_number += 1

    return collected_books


def build_book_dataframe() -> pd.DataFrame:
    """Scrape the first three book categories and return a cleaned DataFrame."""
    records: list[dict[str, object]] = []

    for category_name, category_url in extract_category_links():
        records.extend(scrape_category(category_name, category_url))

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        raise ValueError("No book records were scraped from the category pages.")

    if "price_gbp" in dataframe.columns:
        dataframe["price_gbp"] = pd.to_numeric(dataframe["price_gbp"], errors="coerce")
        median_price = dataframe["price_gbp"].median()
        dataframe["price_gbp"] = dataframe["price_gbp"].fillna(median_price)

    if "rating" in dataframe.columns:
        dataframe["rating"] = pd.to_numeric(dataframe["rating"], errors="coerce")
        dataframe = dataframe.dropna(subset=["rating"]).reset_index(drop=True)

    if "in_stock" in dataframe.columns:
        dataframe["in_stock"] = dataframe["in_stock"].astype(bool)

    dataframe["price_inr"] = dataframe["price_gbp"] * FIXED_GBP_TO_INR
    dataframe = dataframe.drop_duplicates(subset=["title", "category"]).reset_index(drop=True)
    return dataframe[["title", "price_gbp", "rating", "in_stock", "category", "price_inr"]]


def create_database(database_path: Path, df: pd.DataFrame) -> None:
    """Create the SQLite schema and load the normalized category and book tables."""
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS books")
        cursor.execute("DROP TABLE IF EXISTS categories")
        cursor.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            )
            """
        )

        category_rows = sorted(df["category"].unique().tolist())
        category_lookup: dict[str, int] = {}
        for index, category_name in enumerate(category_rows, start=1):
            cursor.execute(
                "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
                (index, category_name),
            )
            category_lookup[category_name] = index

        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["title"],
                    float(row["price_gbp"]),
                    float(row["price_inr"]),
                    int(row["rating"]),
                    int(bool(row["in_stock"])),
                    category_lookup[row["category"]],
                ),
            )

        connection.commit()


def run_sql_queries(database_path: Path) -> list[tuple[str, list[tuple]]]:
    """Run a set of SQL queries that collectively cover the required clauses."""
    with sqlite3.connect(database_path) as connection:
        category_ids = [
            row[0]
            for row in connection.execute(
                "SELECT category_id FROM categories ORDER BY category_id LIMIT 3;"
            ).fetchall()
        ]
        category_id_list = ", ".join(str(category_id) for category_id in category_ids)

    queries = [
        (
            "SELECT title, price_inr FROM books WHERE in_stock = 1 LIMIT 10;",
            "SELECT",
        ),
        (
            "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
            "ORDER BY",
        ),
        (
            "SELECT DISTINCT category_id FROM books ORDER BY category_id;",
            "DISTINCT",
        ),
        (
            "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 500 AND 1000 ORDER BY price_inr DESC;",
            "BETWEEN",
        ),
        (
            f"SELECT title, rating FROM books WHERE category_id IN ({category_id_list}) ORDER BY rating DESC, title LIMIT 10;",
            "IN",
        ),
        (
            "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON c.category_id = b.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;",
            "JOIN",
        ),
    ]

    results: list[tuple[str, list[tuple]]] = []
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        for sql_query, clause in queries:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            results.append((f"{clause}: {sql_query}", rows))
    return results


def compare_join_approach(database_path: Path) -> None:
    """Read the join query using pandas and compare it with a direct DataFrame merge."""
    with sqlite3.connect(database_path) as connection:
        join_sql = pd.read_sql(
            """
            SELECT c.category_name, b.title, b.rating, b.price_inr
            FROM books b
            JOIN categories c ON c.category_id = b.category_id
            ORDER BY b.rating DESC, b.price_inr DESC
            LIMIT 10
            """,
            connection,
        )

    books_df = pd.read_sql_query(
        "SELECT book_id, title, rating, price_inr, category_id FROM books",
        sqlite3.connect(database_path),
    )
    category_df = pd.read_sql_query(
        "SELECT category_id, category_name FROM categories",
        sqlite3.connect(database_path),
    )

    merge_result = books_df.merge(category_df, on="category_id", how="inner")[[
        "category_name",
        "title",
        "rating",
        "price_inr",
    ]].sort_values(["rating", "price_inr"], ascending=[False, False]).head(10).reset_index(drop=True)

    print("SQL JOIN output via pd.read_sql:")
    print(join_sql.to_string(index=False))
    print("\nDirect pandas merge output:")
    print(merge_result.to_string(index=False))
    print("\nEquivalent outputs:", join_sql.reset_index(drop=True).equals(merge_result.reset_index(drop=True)))


def main() -> None:
    """Run the end-to-end scraping, cleaning, SQLite loading, and SQL validation flow."""
    DATA_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    books_df = build_book_dataframe()

    books_df.to_csv(CSV_PATH, index=False)
    create_database(DB_PATH, books_df)

    print(f"Scraped rows: {len(books_df)}")
    print(f"Distinct categories: {books_df['category'].nunique()}")
    print("Price conversion rate: 1 GBP = 105.50 INR")
    print("Saved CSV to:", CSV_PATH)
    print("Saved database to:", DB_PATH)

    query_results = run_sql_queries(DB_PATH)
    for query_text, rows in query_results:
        print("\n" + query_text)
        print(rows[:10])

    compare_join_approach(DB_PATH)


if __name__ == "__main__":
    main()
