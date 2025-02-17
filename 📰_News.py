import streamlit as st
import requests

# Streamlit app configuration
st.set_page_config(
    page_title="📰 Today News",
    page_icon="📰",
    layout="wide",
)

# Title
st.title("📰 Today News")

# Fetch news articles
def fetch_news(api_key, category="business", query=None):
    if query:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error(f"Failed to fetch news: {response.status_code}")
        return []

# Load API key from secrets.toml
news_api_key = st.secrets["NEWS_API_KEY"]

# Sidebar for category and search
st.sidebar.header("Settings")
category = st.sidebar.selectbox(
    "Select Category",
    ["Business", "Technology", "Sports", "Health", "Entertainment", "Science"],
    index=0,
)
search_query = st.sidebar.text_input("Search For News:")

# Fetch news articles
news_articles = fetch_news(news_api_key, category, search_query)

# Display news articles
if news_articles:
    # Main news box (top story)
    main_article = news_articles[0]
    st.header("Top Story")
    col1, col2 = st.columns([1, 2])
    with col1:
        if main_article["urlToImage"]:
            st.image(main_article["urlToImage"], use_container_width=True)
        else:
            st.image("https://via.placeholder.com/600x400.png?text=No+Image", use_container_width=True)
    with col2:
        st.subheader(main_article["title"])
        st.write(main_article["description"])
        st.markdown(f"[Read more]({main_article['url']})")
    st.write("---")

    # Secondary news boxes
    st.header("More News")
    cols = st.columns(3)  # Create 3 columns for smaller news boxes
    for i, article in enumerate(news_articles[1:]):  # Skip the first article (top story)
        with cols[i % 3]:
            if article["urlToImage"]:
                st.image(article["urlToImage"], use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x200.png?text=No+Image", use_container_width=True)
            st.subheader(article["title"])
            st.write(article["description"][:100] + "...")  # Truncate description
            st.markdown(f"[Read more]({article['url']})")
else:
    st.warning("No news articles found.")