#!/usr/bin/env python3
import csv
import sys
import os
import requests
import time
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import anthropic

def load_api_keys():
    """Load API keys from .env.local file"""
    load_dotenv('.env.local')
    scrapingbee_key = os.getenv('SCRAPINGBEE_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if not scrapingbee_key:
        raise ValueError("SCRAPINGBEE_API_KEY not found in .env.local")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env.local")

    return scrapingbee_key, anthropic_key

def search_walmart(query: str, api_key: str) -> Dict[str, Any]:
    """Search Walmart using ScrapingBee API"""
    url = "https://app.scrapingbee.com/api/v1/walmart/search"
    params = {
        "api_key": api_key,
        "query": query
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Output raw API response to stdout
        print(f"\n=== API Response for '{query}' ===")
        print(json.dumps(result, indent=2))
        print("=" * 50)

        return result
    except requests.exceptions.RequestException as e:
        print(f"Error searching for '{query}': {e}")
        return None

def extract_product_info(search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract relevant product information from search results"""
    products = []

    if not search_results:
        return products

    # Check for products in the response (actual API structure)
    items = search_results.get('products', [])

    for item in items:
        product = {
            'title': item.get('title', ''),
            'price': item.get('price', ''),
            'currency': item.get('currency', 'USD'),
            'rating': item.get('rating', ''),
            'reviews_count': item.get('rating_count', ''),
            'availability': 'In Stock' if not item.get('out_of_stock', False) else 'Out of Stock',
            'seller': item.get('seller_name', ''),
            'product_id': item.get('id', ''),
            'link': f"https://www.walmart.com{item.get('url', '')}" if item.get('url') else ''
        }
        products.append(product)

    return products

def check_categories_with_sonnet(csv_filename: str, anthropic_key: str, batch_size: int = 10):
    """Check if products belong in their categories using Sonnet with batched requests"""
    # Read the CSV
    rows = []
    with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=anthropic_key)

    # Filter products that need category checking
    products_to_check = []
    row_indices = []
    for i, row in enumerate(rows):
        # Skip if already checked or no product found
        if not row['belongs_in_category'] and row['title'] != 'No results found' and row['category']:
            products_to_check.append({
                'product_id': row['product_id'],
                'title': row['title'],
                'category': row['category']
            })
            row_indices.append(i)

    if not products_to_check:
        print("No products need category checking")
        return

    print(f"\n=== Checking {len(products_to_check)} products in batches of {batch_size} ===")

    # Define the tool for structured responses
    category_check_tool = {
        "name": "category_checker",
        "description": "Categorize products based on whether they belong in the specified category",
        "input_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The product ID from the input"
                            },
                            "belongs_in_category": {
                                "type": "boolean",
                                "description": "True if the product belongs in the specified category, False otherwise"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation (max 15 words) for the categorization decision"
                            }
                        },
                        "required": ["product_id", "belongs_in_category", "reason"]
                    }
                }
            },
            "required": ["results"]
        }
    }

    # Process in batches
    updated = False
    for batch_start in range(0, len(products_to_check), batch_size):
        batch_end = min(batch_start + batch_size, len(products_to_check))
        batch = products_to_check[batch_start:batch_end]
        batch_indices = row_indices[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//batch_size + 1} (products {batch_start+1}-{batch_end})")

        # Create prompt for batch
        products_list = "\n".join([
            f"- Product ID: {p['product_id']}, Title: \"{p['title']}\", Category: \"{p['category']}\""
            for p in batch
        ])

        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                tools=[category_check_tool],
                messages=[
                    {
                        "role": "user",
                        "content": f"""For each product below, determine if it belongs in the specified category.

Products to categorize:
{products_list}

Use the category_checker tool to provide your structured response for ALL products listed above.
Be strict about category matching - products should clearly fit the category description."""
                    }
                ]
            )

            # Process tool use response
            for content in message.content:
                if content.type == 'tool_use' and content.name == 'category_checker':
                    results = content.input.get('results', [])

                    # Create a mapping of product_id to results
                    results_map = {r['product_id']: r for r in results}

                    # Update the rows with the results
                    for idx, product in zip(batch_indices, batch):
                        if product['product_id'] in results_map:
                            result = results_map[product['product_id']]
                            rows[idx]['belongs_in_category'] = 'Yes' if result['belongs_in_category'] else 'No'
                            rows[idx]['category_check_reason'] = result['reason']
                            print(f"  {product['title'][:50]}: {rows[idx]['belongs_in_category']} - {result['reason']}")
                            updated = True
                        else:
                            print(f"  Warning: No result for product {product['product_id']}")

        except Exception as e:
            print(f"  Error processing batch: {e}")
            # Mark batch as error
            for idx in batch_indices:
                rows[idx]['belongs_in_category'] = 'Error'
                rows[idx]['category_check_reason'] = str(e)[:50]

        # Small delay between batches to avoid rate limiting
        if batch_end < len(products_to_check):
            time.sleep(1)

    # Write updated results back to CSV
    if updated:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nUpdated category checks saved to '{csv_filename}'")

def process_csv(input_filename: str, output_filename: str = None):
    """Process CSV file and search for each item"""
    if not output_filename:
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_results.csv"

    scrapingbee_key, anthropic_key = load_api_keys()

    # Read input CSV with categories
    search_items = []
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header row
            for row in reader:
                if row and len(row) >= 1:  # Check if row is not empty
                    item = {
                        'name': row[0],
                        'category': row[1] if len(row) > 1 else ''
                    }
                    search_items.append(item)
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    print(f"Found {len(search_items)} items to search")

    # Prepare output CSV
    fieldnames = [
        'search_query', 'category', 'title', 'price', 'currency', 'rating',
        'reviews_count', 'availability', 'seller', 'product_id', 'link',
        'belongs_in_category', 'category_check_reason'
    ]

    all_results = []

    # Search for each item
    for i, item in enumerate(search_items, 1):
        search_term = item['name']
        category = item['category']
        print(f"Searching {i}/{len(search_items)}: {search_term}")

        search_results = search_walmart(search_term, scrapingbee_key)

        if search_results:
            products = extract_product_info(search_results)

            if products:
                # Add all results for each search
                for product in products:
                    product['search_query'] = search_term
                    product['category'] = category
                    product['belongs_in_category'] = ''
                    product['category_check_reason'] = ''
                    all_results.append(product)
                print(f"  Found {len(products)} products")
            else:
                # Add empty result if no products found
                all_results.append({
                    'search_query': search_term,
                    'category': category,
                    'title': 'No results found',
                    'price': '',
                    'currency': '',
                    'rating': '',
                    'reviews_count': '',
                    'availability': '',
                    'seller': '',
                    'product_id': '',
                    'link': '',
                    'belongs_in_category': '',
                    'category_check_reason': ''
                })
                print(f"  No products found")
        else:
            print(f"  Search failed")

        # Add delay to avoid rate limiting
        if i < len(search_items):
            time.sleep(1)

    # Write initial results to output CSV
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nResults saved to '{output_filename}'")
        print(f"Total products found: {len(all_results)}")

    except Exception as e:
        print(f"Error writing output CSV: {e}")
        sys.exit(1)

    # Check categories with Sonnet
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(f"\n=== Checking product categories with Sonnet (batch size: {batch_size}) ===")
    check_categories_with_sonnet(output_filename, anthropic_key, batch_size)

def main():
    if len(sys.argv) < 2:
        print("Usage: python walmart_search.py <input_csv_file> [output_csv_file] [batch_size]")
        print("\nExample: python walmart_search.py items.csv walmart_results.csv 10")
        print("\nParameters:")
        print("  input_csv_file: CSV with 'Product Name' and 'Category' columns")
        print("  output_csv_file: Optional output filename (default: input_results.csv)")
        print("  batch_size: Optional batch size for category checking (default: 10)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None

    process_csv(input_file, output_file)

if __name__ == "__main__":
    main()