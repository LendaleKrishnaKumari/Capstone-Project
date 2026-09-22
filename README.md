# Capstone Project – Data, Analytics, and Support Assistant

This repository contains the complete three-module project required for the assignment. The full project is designed as one connected repository with three sibling modules that work together as a single story: a raw-to-relational data pipeline, an analytics workflow, and a grounded support assistant.

## Repository layout

- [data_pipeline](data_pipeline) — scraping, cleaning, conversion, SQLite database, and SQL output checks
- [analytics](analytics) — dataset profiling, model training, evaluation, and saved artifacts
- [support_assistant](support_assistant) — FAQ knowledge base, hybrid matching, and support agent logic

## Module 1 – Data Pipeline

This module scrapes live book data from Books to Scrape, cleans the scraped fields, converts the currency using the required fixed project rate of 1 GBP = 105.50 INR, and stores the cleaned data in a normalized SQLite database with two tables connected by a primary/foreign key relationship.

Requirements implemented:
- scrape all books across at least 3 categories
- collect at least 60 books
- convert price to GBP and INR
- normalize rating to integer 1–5
- convert availability to boolean in_stock
- save and query the cleaned database using SQLite and pandas

Files:
- [data_pipeline/data_pipeline.py](data_pipeline/data_pipeline.py)
- [data_pipeline/books_catalog.csv](data_pipeline/books_catalog.csv)
- [data_pipeline/books_catalog.db](data_pipeline/books_catalog.db)

## Module 2 – Analytics Pipeline

The analytics module profiles the cleaned dataset, defines a target variable, trains a classifier, evaluates performance, and saves the trained model and metrics.

Files:
- [analytics/analytics_pipeline.py](analytics/analytics_pipeline.py)
- [analytics/model_metrics.json](analytics/model_metrics.json)
- [analytics/customer_spending_model.pkl](analytics/customer_spending_model.pkl)

## Module 3 – Support Assistant

The support assistant uses FAQ knowledge, keyword matching, TF-IDF matching, hybrid score combination, and a grounded LLM response mechanism when a valid API key is present.

Files:
- [support_assistant/faq_knowledge_base.py](support_assistant/faq_knowledge_base.py)
- [support_assistant/intelligent_faq_matching.py](support_assistant/intelligent_faq_matching.py)
- [support_assistant/openrouter_llm_integration.py](support_assistant/openrouter_llm_integration.py)
- [support_assistant/support_agent.py](support_assistant/support_agent.py)

## Naming conventions used

- Folders: lowercase with underscores
- Python files: lowercase with underscores
- Functions: snake_case
- Classes: PascalCase
- Variables: snake_case

## Setup and run instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run each module:

```bash
python data_pipeline/data_pipeline.py
python analytics/analytics_pipeline.py
python support_assistant/faq_knowledge_base.py
```

## Fixed conversion rule used in Module 1

The textbook/project-required currency baseline for this assignment is:

1 GBP = 105.50 INR

This fixed rate is used directly in the data pipeline and is documented here as the required conversion baseline for grading.

## Submission note

This repository uses one single public GitHub repository structure, contains all three modules in the required root layout, and records the data pipeline outputs, model artifacts, and support assistant logic in separate folders as required.
