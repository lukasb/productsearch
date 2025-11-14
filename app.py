import streamlit as st
import pandas as pd
import io
import os
import requests
import time
import anthropic
import json
from typing import List, Dict, Any

# Page config
st.set_page_config(
    page_title="Walmart Product Category Checker",
    page_icon="🛒",
    layout="wide"
)

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
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching for '{query}': {e}")
        return None

def extract_product_info(search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract relevant product information from search results"""
    products = []

    if not search_results:
        return products

    # Check for products in the response
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

def check_categories_batch(products_batch: List[Dict], client: anthropic.Anthropic) -> Dict[str, Dict]:
    """Check a batch of products against their categories using Sonnet"""

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

    # Create prompt for batch
    products_list = "\n".join([
        f"- Product ID: {p['product_id']}, Title: \"{p['title']}\", Category: \"{p['category']}\""
        for p in products_batch
    ])

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
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
        results_map = {}
        for content in message.content:
            if content.type == 'tool_use' and content.name == 'category_checker':
                results = content.input.get('results', [])
                for r in results:
                    results_map[r['product_id']] = {
                        'belongs_in_category': 'Yes' if r['belongs_in_category'] else 'No',
                        'category_check_reason': r['reason']
                    }

        return results_map

    except Exception as e:
        st.error(f"Error processing batch: {e}")
        return {}

def process_csv_data(df: pd.DataFrame, scrapingbee_key: str, anthropic_key: str, batch_size: int = 10):
    """Process the CSV data and return results"""
    all_results = []

    # Setup progress tracking
    total_items = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process each item
    for idx, row in df.iterrows():
        search_term = row.iloc[0]  # First column
        category = row.iloc[1] if len(row) > 1 else ''  # Second column if exists

        status_text.text(f"Searching for: {search_term} ({idx+1}/{total_items})")

        # Search Walmart
        search_results = search_walmart(search_term, scrapingbee_key)

        if search_results:
            products = extract_product_info(search_results)

            if products:
                for product in products:
                    product['search_query'] = search_term
                    product['category'] = category
                    product['belongs_in_category'] = ''
                    product['category_check_reason'] = ''
                    all_results.append(product)
            else:
                # No products found
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

        # Update progress
        progress_bar.progress((idx + 1) / total_items)

        # Rate limiting
        if idx < total_items - 1:
            time.sleep(1)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Check categories if we have them
    if anthropic_key and not results_df.empty and 'category' in results_df.columns:
        status_text.text("Checking categories with Sonnet...")

        # Initialize Anthropic client once
        try:
            # Try to initialize without proxy settings
            import os
            # Clear any proxy environment variables temporarily
            proxy_vars = {}
            for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                if var in os.environ:
                    proxy_vars[var] = os.environ.pop(var)

            client = anthropic.Anthropic(api_key=anthropic_key)

            # Restore proxy vars
            for var, value in proxy_vars.items():
                os.environ[var] = value

        except Exception as e:
            st.error(f"Failed to initialize Anthropic client: {e}")
            return results_df

        # Filter products that need checking
        products_to_check = []
        for idx, row in results_df.iterrows():
            if row['title'] != 'No results found' and row['category']:
                products_to_check.append({
                    'index': idx,
                    'product_id': row['product_id'],
                    'title': row['title'],
                    'category': row['category']
                })

        if products_to_check:
            # Process in batches
            total_batches = (len(products_to_check) + batch_size - 1) // batch_size

            for i in range(0, len(products_to_check), batch_size):
                batch = products_to_check[i:i+batch_size]
                status_text.text(f"Processing category batch {i//batch_size + 1}/{total_batches}")

                results = check_categories_batch(batch, client)

                # Update DataFrame with results
                for product in batch:
                    if product['product_id'] in results:
                        idx = product['index']
                        results_df.at[idx, 'belongs_in_category'] = results[product['product_id']]['belongs_in_category']
                        results_df.at[idx, 'category_check_reason'] = results[product['product_id']]['category_check_reason']

                # Rate limiting between batches
                if i + batch_size < len(products_to_check):
                    time.sleep(1)

    status_text.text("Processing complete!")
    return results_df

# Streamlit UI
st.title("🛒 Walmart Product Category Checker")
st.markdown("""
This app searches Walmart for products from your CSV and optionally checks if they belong in specified categories.

**Required CSV format:**
- Column 1: Product names to search
- Column 2 (optional): Category to check against
""")

# Sidebar for API keys
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check for environment variables first
    scrapingbee_key = os.getenv('SCRAPINGBEE_API_KEY', '')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

    # Allow override via UI
    scrapingbee_key = st.text_input(
        "ScrapingBee API Key",
        value=scrapingbee_key,
        type="password",
        help="Required for Walmart searches"
    )

    anthropic_key = st.text_input(
        "Anthropic API Key (Optional)",
        value=anthropic_key,
        type="password",
        help="Required for category checking with Claude"
    )

    batch_size = st.slider(
        "Batch Size for Category Checking",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of products to check in each API call"
    )

    st.divider()

    # Show API status
    st.subheader("API Status")
    if scrapingbee_key:
        st.success("✅ ScrapingBee API key configured")
    else:
        st.warning("⚠️ ScrapingBee API key missing")

    if anthropic_key:
        st.success("✅ Anthropic API key configured")
    else:
        st.info("ℹ️ Anthropic API key not configured (category checking disabled)")

# Main content
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and display the CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of uploaded data:")
    st.dataframe(df.head(10))

    st.info(f"Found {len(df)} items to search")

    # Process button
    if st.button("🚀 Process CSV", type="primary", disabled=not scrapingbee_key):
        if not scrapingbee_key:
            st.error("Please provide a ScrapingBee API key in the sidebar")
        else:
            with st.spinner("Processing... This may take a few minutes."):
                result_df = process_csv_data(df, scrapingbee_key, anthropic_key, batch_size)

                # Store in session state
                st.session_state['results'] = result_df

            st.success(f"✅ Processing complete! Found {len(result_df)} total products.")

            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products", len(result_df))
            with col2:
                unique_searches = result_df['search_query'].nunique()
                st.metric("Unique Searches", unique_searches)
            with col3:
                if 'belongs_in_category' in result_df.columns:
                    yes_count = (result_df['belongs_in_category'] == 'Yes').sum()
                    st.metric("Products in Category", yes_count)

# Display results if available
if 'results' in st.session_state:
    st.divider()
    st.subheader("📋 Results")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        search_filter = st.selectbox(
            "Filter by search query",
            ["All"] + list(st.session_state['results']['search_query'].unique())
        )

    with col2:
        if 'belongs_in_category' in st.session_state['results'].columns:
            category_filter = st.selectbox(
                "Filter by category match",
                ["All", "Yes", "No", "Unchecked"]
            )
        else:
            category_filter = "All"

    # Apply filters
    filtered_df = st.session_state['results'].copy()

    if search_filter != "All":
        filtered_df = filtered_df[filtered_df['search_query'] == search_filter]

    if category_filter != "All":
        if category_filter == "Unchecked":
            filtered_df = filtered_df[filtered_df['belongs_in_category'] == '']
        else:
            filtered_df = filtered_df[filtered_df['belongs_in_category'] == category_filter]

    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)

    # Download button
    csv_buffer = io.StringIO()
    st.session_state['results'].to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results CSV",
        data=csv_string,
        file_name="walmart_search_results.csv",
        mime="text/csv"
    )

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to use this app:

    1. **Prepare your CSV file** with product names in the first column
    2. **Optional**: Add categories in the second column for category checking
    3. **Enter API keys** in the sidebar (ScrapingBee required, Anthropic optional)
    4. **Upload your CSV** using the file uploader
    5. **Click Process CSV** to start searching
    6. **Download the results** when processing is complete

    ### CSV Format Example:
    ```
    Product Name,Category
    laptop,Electronics
    wireless headphones,Electronics
    coffee maker,Kitchen Appliances
    ```

    ### Features:
    - Searches Walmart for each product
    - Returns detailed product information
    - Optionally checks if products belong in specified categories using AI
    - Batch processing for efficiency
    - Downloadable results in CSV format
    """)