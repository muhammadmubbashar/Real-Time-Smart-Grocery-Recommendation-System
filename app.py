import os
import sys
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="🛒 Smart Grocery Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    
    .header-container {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .product-card {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .product-card:hover {
        border-color: #10b981;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
        transform: translateY(-2px);
    }
    
    .product-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .product-info {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .cart-container {
        background-color: #f0fdf4;
        border: 2px solid #10b981;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .cart-title {
        color: #065f46;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .cart-item {
        background-color: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
        color: #1e293b;
        font-weight: 500;
    }
    
    .recommendation-card {
        background-color: white;
        border-left: 5px solid #10b981;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .high-confidence {
        border-left-color: #10b981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%);
    }
    
    .medium-confidence {
        border-left-color: #f59e0b;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.05) 0%, transparent 100%);
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .badge-high {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge-medium {
        background-color: #fed7aa;
        color: #92400e;
    }
    
    .stButton > button {
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .products-scroll {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        background-color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
ARTIFACT_PATH = Path("artifact")
INGESTED_DATA_PATH = ARTIFACT_PATH / "dataset" / "ingested_data"
SERIALIZED_OBJECTS_PATH = ARTIFACT_PATH / "serialized_objects"
MODELS_PATH = ARTIFACT_PATH / "models"
RECOMMENDATIONS_PATH = ARTIFACT_PATH / "recommendations"

# Initialize session state
if "cart_items" not in st.session_state:
    st.session_state.cart_items = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# IMPORTANT: Only keep valid product IDs in cart
# Remove any invalid/deleted products
if st.session_state.cart_items:
    st.session_state.cart_items = list(filter(None, st.session_state.cart_items))

@st.cache_resource
def load_products():
    """Load products with aisle and department info"""
    products = pd.read_csv(INGESTED_DATA_PATH / "products.csv")
    aisles = pd.read_csv(INGESTED_DATA_PATH / "aisles.csv")
    departments = pd.read_csv(INGESTED_DATA_PATH / "departments.csv")
    
    products = products.merge(aisles, on="aisle_id", how="left")
    products = products.merge(departments, on="department_id", how="left")
    
    return products

@st.cache_resource
def load_model():
    """Load trained Gradient Boosting model"""
    with open(MODELS_PATH / "gb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_transformed_data():
    """Load transformed data with features"""
    with open(SERIALIZED_OBJECTS_PATH / "transformed_data.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_precomputed_recommendations():
    """Load pre-computed recommendations from training"""
    try:
        with open(RECOMMENDATIONS_PATH / "user_recommendations.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.warning("⚠️ Pre-computed recommendations not found, using model predictions instead")
        return None

class GroceryRecommender:
    def __init__(self, transformed_data, model, products_df, precomputed_recs=None):
        self.transformed_data = transformed_data
        self.model = model
        self.products_df = products_df
        self.precomputed_recs = precomputed_recs
    
    def get_recommendations(self, user_id, cart_product_ids, top_k=8):
        """Collaborative Filtering: Find products bought by users who also bought cart items"""
        try:
            if not cart_product_ids:
                st.error("❌ Cart is empty")
                return pd.DataFrame()
            
            cart_set = set(cart_product_ids)
            st.write(f"👥 **Collaborative Filtering: Finding similar shoppers...**")
            
            # STEP 1: Find all users who bought ANY item in the cart
            cart_buyers = self.transformed_data[
                self.transformed_data["product_id"].isin(cart_product_ids)
            ].copy()
            
            similar_users = cart_buyers["user_id"].unique()
            st.write(f"✅ Found {len(similar_users):,} shoppers who bought similar items")
            
            # STEP 2: Find all products bought by these similar users
            similar_users_purchases = self.transformed_data[
                self.transformed_data["user_id"].isin(similar_users)
            ].copy()
            
            # STEP 3: Exclude items already in cart
            similar_users_purchases = similar_users_purchases[
                ~similar_users_purchases["product_id"].isin(cart_set)
            ]
            
            if len(similar_users_purchases) == 0:
                st.warning("⚠️ No additional products found")
                return pd.DataFrame()
            
            # STEP 4: Score products by how many similar users bought them
            product_scores = similar_users_purchases.groupby("product_id").agg({
                "user_id": "nunique",  # Count unique users who bought this
                "reordered": "mean"     # Average reorder likelihood
            }).reset_index()
            
            product_scores.columns = ["product_id", "buyer_count", "reorder_rate"]
            
            # Combined score: popularity + reorder rate
            max_buyers = product_scores["buyer_count"].max()
            product_scores["score"] = (
                (product_scores["buyer_count"] / max_buyers * 0.7) +  # 70% weight to popularity
                (product_scores["reorder_rate"] * 0.3)                 # 30% weight to reorder rate
            )
            
            # Sort by score
            product_scores = product_scores.sort_values("score", ascending=False)
            
            st.write(f"📊 Scoring {len(product_scores):,} candidate products")
            
            # Get top K
            top_products = product_scores.head(top_k).copy()
            
            # Merge with product details
            recs = top_products[["product_id", "score"]].copy()
            recs.columns = ["product_id", "reorder_probability"]
            
            recs = recs.merge(
                self.products_df[["product_id", "product_name", "aisle", "department"]],
                on="product_id",
                how="left"
            )
            
            st.success(f"✅ Found {len(recs)} collaborative recommendations!")
            return recs
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Debug Info"):
                st.error(traceback.format_exc())
            return pd.DataFrame()

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1>🛒 Smart Grocery Recommender</h1>
            <p>Add items to your cart and get real-time recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        products_df = load_products()
        model = load_model()
        transformed_data = load_transformed_data()
        precomputed_recs = load_precomputed_recommendations()
        recommender = GroceryRecommender(transformed_data, model, products_df, precomputed_recs)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run: python main.py")
        return
    
    # User selection
    st.sidebar.header("👤 Your Profile")
    available_users = sorted(transformed_data["user_id"].unique())[:500]
    
    # Search box for user ID
    search_user = st.sidebar.text_input("Search User ID:", placeholder="e.g., 1, 100, 5000")
    
    if search_user:
        try:
            search_id = int(search_user)
            if search_id in available_users:
                selected_user = search_id
            else:
                st.sidebar.warning(f"❌ User ID {search_id} not found")
                selected_user = available_users[0]
        except:
            st.sidebar.warning("❌ Invalid User ID")
            selected_user = available_users[0]
    else:
        selected_user = available_users[0]
    
    # Auto-select first user
    st.session_state.user_id = selected_user
    st.sidebar.success(f"✅ User: {selected_user}")
    
    st.sidebar.markdown("---")
    
    # Clear session button
    if st.sidebar.button("🔄 Reset Everything", use_container_width=True):
        st.session_state.cart_items = []
        st.session_state.user_id = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Main layout
    col_browse, col_cart = st.columns([2, 1])
    
    with col_browse:
        st.header("🛍️ Browse Groceries")
        
        # Department filter dropdown
        dept_filter = st.selectbox(
            "Select category:",
            ["All"] + sorted(products_df["department"].unique().tolist()),
            index=0
        )
        
        # Search bar
        search = st.text_input("Search products:", placeholder="e.g., Milk, Bread...")
        
        # Filter products
        filtered = products_df.copy()
        
        if dept_filter != "All":
            filtered = filtered[filtered["department"] == dept_filter]
        
        if search:
            filtered = filtered[filtered["product_name"].str.contains(search, case=False, na=False)]
        
        filtered = filtered.drop_duplicates("product_id")
        
        # Show scrollable dropdown when category selected or search typed
        show_products = (dept_filter != "All") or bool(search)
        
        if show_products and len(filtered) > 0:
            st.write(f"**📦 {len(filtered)} products found**")
            
            # Scrollable container - shows max 10 items at a time
            st.markdown('''
                <div style="max-height: 520px; overflow-y: auto; border: 1px solid #10b981; border-radius: 0.5rem; padding: 0.5rem; background: #f8fafc;">
            ''', unsafe_allow_html=True)
            
            for idx, (_, prod) in enumerate(filtered.head(10).iterrows()):
                col1, col2 = st.columns([3.5, 0.5])
                
                with col1:
                    st.markdown(f"""
                        <div style="padding: 0.75rem; background: white; margin-bottom: 0.25rem; border-radius: 0.375rem; border-left: 3px solid #10b981;">
                            <div style="font-weight: 500; color: #1e293b;">{prod['product_name']}</div>
                            <div style="font-size: 0.875rem; color: #64748b;">{prod['aisle']} • {prod['department']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    in_cart = prod["product_id"] in st.session_state.cart_items
                    if in_cart:
                        if st.button("❌", key=f"rm_{idx}"):
                            st.session_state.cart_items.remove(prod["product_id"])
                            st.rerun()
                    else:
                        if st.button("➕", key=f"add_{idx}"):
                            st.session_state.cart_items.append(prod["product_id"])
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        elif show_products and len(filtered) == 0:
            st.warning("❌ No products found")
        else:
            st.info("🔍 Select a category or type to search")
    
    with col_cart:
        st.header("🛒 Cart")
        
        if st.session_state.cart_items:
            st.markdown(f"""
                <div class="cart-container">
                    <div class="cart-title">{len(st.session_state.cart_items)} Items</div>
            """, unsafe_allow_html=True)
            
            for product_id in st.session_state.cart_items:
                prod = products_df[products_df["product_id"] == product_id]
                if len(prod) > 0:
                    name = prod.iloc[0]["product_name"][:25]
                    st.markdown(f'<div class="cart-item">✔️ {name}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("Clear Cart", use_container_width=True):
                st.session_state.cart_items = []
                st.rerun()
        else:
            st.info("Cart is empty\n\n➕ Add items from the left to get recommendations!")
    
    # Only show recommendations if cart has items
    if len(st.session_state.cart_items) > 0:
        st.markdown("---")
        st.header("✨ Recommendations For You")
        
        recs = recommender.get_recommendations(
            st.session_state.user_id,
            st.session_state.cart_items,
            top_k=8
        )
        
        if len(recs) > 0:
            cols = st.columns(2)
            
            for idx, (_, rec) in enumerate(recs.iterrows()):
                col = cols[idx % 2]
                
                with col:
                    prob = rec["reorder_probability"]
                    conf = "badge-high" if prob >= 0.7 else "badge-medium"
                    card_class = "high-confidence" if prob >= 0.7 else "medium-confidence"
                    badge_text = "🟢 High" if prob >= 0.7 else "🟡 Medium"
                    
                    st.markdown(f"""
                        <div class="recommendation-card {card_class}">
                            <div class="product-name">{rec['product_name']}</div>
                            <div class="product-info">{rec['aisle']} • {rec['department']}</div>
                            <span class="badge {conf}">{badge_text}</span>
                            <div style="text-align: center; padding: 0.5rem 0;">
                                <div style="font-size: 1.5rem; color: #10b981; font-weight: 700;">{prob:.0%}</div>
                                <div style="font-size: 0.8rem; color: #64748b;">Reorder Prob.</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if rec["product_id"] not in st.session_state.cart_items:
                        if st.button("Add", key=f"rec_{idx}", use_container_width=True):
                            st.session_state.cart_items.append(rec["product_id"])
                            st.rerun()
        else:
            st.info("No recommendations available yet.")
    
    st.markdown("""
        <div class="footer">
        🛒 Smart Grocery Recommendation System | ML-Powered Shopping
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()