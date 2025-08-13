import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import fpgrowth, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

# Load data files with caching
@st.cache_data
def load_data():
    rfm = pd.read_csv('customer_segments.csv')
    rules = pd.read_csv('association_rules.csv')
    cluster_profiles = pd.read_csv('cluster_profiles.csv')
    cleaned_data = pd.read_csv('cleaned_customer_data.csv')

    # Make sure numeric columns are numeric
    clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']
    for df in [rfm, cluster_profiles]:
        for col in clustering_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return rfm, rules, cluster_profiles, cleaned_data

rfm, rules, cluster_profiles, cleaned_data = load_data()

# Personae dictionary for clusters
PERSONAS = {
    0: {'name': 'Champions', 'description': 'Highly engaged, frequent buyers with high spending and recent activity'},
    1: {'name': 'Loyal Customers', 'description': 'Repeat customers with consistent purchases and solid spend'},
    2: {'name': 'At Risk', 'description': 'Previously valuable customers who have not shopped recently'},
    3: {'name': 'New or Low-Value Customers', 'description': 'Recently acquired or inactive customers with minimal activity'},
    4: {'name': 'Potential Loyalists', 'description': 'Moderate activity and spending, could become loyal with targeted incentives'},
}

CLUSTER_FEATURES = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']

# CSS Styling for theme
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    color: #1f2937;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, h4 {
    color: #0f172a;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #1e40af;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# App Title and footer
st.title("Online Sales Customer Analytics")
st.markdown("**Built by Nastassia Pukelik for the CIS9660 course**")

# Tabs
tabs = st.tabs(["Customer Segmentation", "Market Basket Analysis", "Business Intelligence", "Data Exploration", "Upload Your Data"])

# TAB 1: Customer Segmentation
with tabs[0]:
    st.header("Customer Segmentation Dashboard")

    plot_type = st.radio("Select Cluster Visualization Type", ['2D PCA Plot', '3D Scatter Plot'])
    if plot_type == '2D PCA Plot':
        pca_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rfm[pca_features])
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_features)
        rfm['PCA1'] = components[:, 0]
        rfm['PCA2'] = components[:, 1]
        fig_2d = px.scatter(rfm, x='PCA1', y='PCA2', color='Cluster',
                            hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue'],
                            title='2D PCA Customer Segments',
                            color_continuous_scale=px.colors.qualitative.Safe)
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        fig_3d = px.scatter_3d(
            rfm,
            x='Recency', y='Frequency', z='Monetary',
            color='Cluster',
            hover_data=['CustomerID', 'AvgOrderValue', 'UniqueProducts'],
            title="3D Customer Segmentation"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("Recency Distribution (Days since last purchase)")
    hist_recency = np.histogram(rfm['Recency'], bins=50)
    bin_edges_recency = hist_recency[1]
    bin_centers_recency = (bin_edges_recency[:-1] + bin_edges_recency[1:]) / 2
    fig_recency = go.Figure(
        go.Bar(
            x=bin_centers_recency,
            y=hist_recency[0],
            marker=dict(color='#1f77b4', line=dict(color='black', width=1)),
            width=(bin_edges_recency[1] - bin_edges_recency[0]) * 0.95,
        )
    )
    fig_recency.update_layout(
        xaxis=dict(title='Recency (Days)', showgrid=True, gridcolor='LightGrey', gridwidth=1, zeroline=False),
        yaxis=dict(title='Count'),
        margin=dict(t=40, b=40),
        bargap=0.1
    )
    st.plotly_chart(fig_recency, use_container_width=True)

    st.subheader("Monetary Distribution (Total spending amount)")
    hist_monetary = np.histogram(rfm['Monetary'], bins=50)
    bin_edges_monetary = hist_monetary[1]
    bin_centers_monetary = (bin_edges_monetary[:-1] + bin_edges_monetary[1:]) / 2
    fig_monetary = go.Figure(
        go.Bar(
            x=bin_centers_monetary,
            y=hist_monetary[0],
            marker=dict(color='#ff7f0e', line=dict(color='black', width=1)),
            width=(bin_edges_monetary[1] - bin_edges_monetary[0]) * 0.95,
        )
    )
    fig_monetary.update_layout(
        xaxis=dict(title='Monetary ($)', showgrid=True, gridcolor='LightGrey', gridwidth=1, zeroline=False),
        yaxis=dict(title='Count'),
        margin=dict(t=40, b=40),
        bargap=0.1
    )
    st.plotly_chart(fig_monetary, use_container_width=True)

    st.subheader("Segment Profiles and Comparison")
    cluster_stats = cluster_profiles.copy()
    cluster_stats['Cluster'] = cluster_stats.index.astype(str)
    selected_metrics = st.multiselect("Select metrics to compare", CLUSTER_FEATURES, default=CLUSTER_FEATURES)
    fig_bar = px.bar(cluster_stats, x='Cluster', y=selected_metrics, barmode='group', title='Cluster Profiles')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(cluster_stats.style.format({m: "{:.2f}" for m in CLUSTER_FEATURES}))

    st.subheader("Segment Summary Metrics")
    seg_stats = rfm.groupby('Cluster')[CLUSTER_FEATURES].mean().round(2)
    seg_stats['Count'] = rfm['Cluster'].value_counts().sort_index()
    st.dataframe(seg_stats.style.format("{:.2f}"))

# TAB 2: Market Basket Analysis
with tabs[1]:
    st.header("Market Basket Analysis Interface")

    st.subheader("Country Lookup")
    countries = cleaned_data['Country'].dropna().unique()
    selected_countries = st.multiselect("Select country(s)", sorted(countries), default=sorted(countries))

    country_filtered = cleaned_data[cleaned_data['Country'].isin(selected_countries)] if selected_countries else cleaned_data.copy()
    country_counts = country_filtered['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Transactions']

    fig_country = px.bar(country_counts, x='Country', y='Transactions',
                         title='Number of Transactions by Country',
                         labels={'Transactions': 'Number of Transactions'},
                         color='Transactions',
                         color_continuous_scale=px.colors.sequential.Blues)
    st.plotly_chart(fig_country, use_container_width=True)

    st.subheader("Filter Association Rules")
    min_lift = st.slider("Minimum Lift", min_value=0.3, max_value=0.44, value=0.3, step=0.01, format="%.2f")
    min_conf = st.slider("Minimum Confidence", min_value=0.034, max_value=0.05, value=0.034, step=0.001, format="%.3f")
    min_supp = st.slider("Minimum Support", min_value=0.001, max_value=0.006, value=0.001, step=0.0005, format="%.4f")

    filtered_rules = rules[
        (rules['lift'] >= min_lift) &
        (rules['confidence'] >= min_conf) &
        (rules['support'] >= min_supp)
    ]

    st.write(f"Showing {len(filtered_rules)} rules matching criteria")

    st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True))

    st.subheader("Product Recommendation Engine")
    products_input = st.text_input("Enter product(s) (comma separated):")
    if products_input:
        products_list = [p.strip() for p in products_input.split(",")]
        rec_rules = filtered_rules[filtered_rules['antecedents'].apply(lambda x: any(prod in x for prod in products_list))]
        if rec_rules.empty:
            st.info("No recommendations found for these products.")
        else:
            rec_rules_sorted = rec_rules.sort_values('lift', ascending=False).head(10)
            recommendations = []
            for _, row in rec_rules_sorted.iterrows():
                rec_products = ', '.join(row['consequents'])
                recommendations.append({
                    'Recommended Products': rec_products,
                    'Confidence': f"{row['confidence']:.1%}",
                    'Lift': f"{row['lift']:.2f}"
                })
            st.table(pd.DataFrame(recommendations))

    st.subheader("Association Rules Network Graph")
    max_rules = 50
    filtered_net_rules = filtered_rules.head(max_rules)
    G = nx.DiGraph()
    for _, row in filtered_net_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_node(antecedent, label=antecedent)
                G.add_node(consequent, label=consequent)
                G.add_edge(antecedent, consequent, weight=row['lift'])
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center', hoverinfo='text',
        marker=dict(showscale=False, color='#2563eb', size=15, line_width=2)
    )
    fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network Graph of Association Rules',
                            title_x=0.5,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
    st.plotly_chart(fig_net, use_container_width=True)

    st.subheader("Support, Confidence, and Lift Distributions")
    metric = st.selectbox("Choose metric to visualize", ["support", "confidence", "lift"])
    fig_hist = px.histogram(rules, x=metric, nbins=50, title=f"Distribution of {metric.title()}")
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 3: Business Intelligence
with tabs[2]:
    st.header("Business Intelligence Summary")

    st.subheader("Executive Summary")
    st.markdown("""
    - Customer segmentation reveals distinct groups with unique purchase behaviors.
    - Market basket analysis uncovers strong product association rules for cross-selling.
    - Actionable recommendations are tailored for each segment to improve retention and growth.
    """)

    st.subheader("Actionable Recommendations by Segment")
    for cluster_id, persona in PERSONAS.items():
        st.markdown(f"**{persona['name']} (Cluster {cluster_id})**")
        if cluster_id not in cluster_profiles.index:
            continue
        metrics = cluster_profiles.loc[cluster_id]
        if metrics['Recency'] < 50 and metrics['Frequency'] > 15:
            st.markdown("""
            - RETAIN: These are your best customers.
            - Offer VIP programs and exclusive early access.
            - Send personalized thank you messages.
            - Provide premium customer service.
            """)
        elif metrics['Recency'] > 150 and metrics['Monetary'] > 1000:
            st.markdown("""
            - RE-ENGAGE: High-value customers at risk of churning.
            - Send win-back campaigns with special offers.
            - Conduct surveys to understand why they left.
            - Offer personalized discounts on previous items.
            """)
        elif metrics['Frequency'] < 5:
            st.markdown("""
            - DEVELOP: New or low-engagement customers.
            - Create onboarding email sequences.
            - Offer first-time buyer discounts.
            - Recommend popular products.
            """)
        else:
            st.markdown("""
            - GROW: Customers with growth potential.
            - Cross-sell complementary products.
            - Implement loyalty programs.
            - Send targeted product recommendations.
            """)
        avg_order_value = metrics['Monetary'] / metrics['Frequency'] if metrics['Frequency'] > 0 else 0
        count = rfm['Cluster'].value_counts().get(cluster_id, 0)
        potential_revenue = count * avg_order_value * 0.2
        st.markdown(f"**Potential Revenue Impact:** ${potential_revenue:,.0f}")

    st.subheader("ROI Projections")
    conversion_rate = st.slider("Assumed Conversion Rate (%)", 1, 100, 20)
    avg_order_val_input = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0, step=1.0)
    customers_targeted = st.number_input("Number of Customers Targeted", min_value=0, value=1000, step=10)
    roi_estimate = customers_targeted * (conversion_rate / 100) * avg_order_val_input
    st.markdown(f"Estimated ROI from campaign: **${roi_estimate:,.2f}**")

# TAB 4: Data Exploration
with tabs[3]:
    st.header("Data Exploration")

    st.subheader("Raw Cleaned Customer Data Sample")
    st.dataframe(cleaned_data.head(100))

    st.subheader("Filter Data")
    countries = cleaned_data['Country'].dropna().unique()
    selected_countries_data = st.multiselect("Select countries", sorted(countries), default=sorted(countries))
    filtered_data = cleaned_data[cleaned_data['Country'].isin(selected_countries_data)] if selected_countries_data else cleaned_data.copy()
    st.write(f"Filtered dataset size: {filtered_data.shape[0]} rows")

    fig_scatter = px.scatter(filtered_data, x='Quantity', y='UnitPrice', color='Country', title="Quantity vs Unit Price by Country",
                             hover_data=['Description', 'CustomerID'])
    st.plotly_chart(fig_scatter, use_container_width=True)

    filtered_data['InvoiceDate'] = pd.to_datetime(filtered_data['InvoiceDate'], errors='coerce')
    sales_over_time = filtered_data.groupby(filtered_data['InvoiceDate'].dt.to_period("M"))['TotalPrice'].sum()
    sales_over_time.index = sales_over_time.index.to_timestamp()
    fig_time = px.line(sales_over_time, x=sales_over_time.index, y=sales_over_time.values,
                       title='Total Sales Over Time (Monthly)', labels={'x':'Month', 'y':'Total Sales'})
    st.plotly_chart(fig_time, use_container_width=True)

# TAB 5: Upload Your Data
with tabs[4]:
    st.header("Upload Your Own Customer Data")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        st.subheader("Preview of Uploaded Data")
        st.dataframe(user_df.head(100))

        st.subheader("RFM Analysis Settings")
        col_customer = st.selectbox("Select Customer ID column", user_df.columns)
        col_invoice = st.selectbox("Select Invoice Date column", user_df.columns)
        col_invoice_no = st.selectbox("Select Invoice Number column", user_df.columns)
        col_quantity = st.selectbox("Select Quantity column", user_df.columns)
        col_unitprice = st.selectbox("Select Unit Price column", user_df.columns)
        col_description = st.selectbox("Select Product Description column", user_df.columns)

        if st.button("Run RFM Analysis"):
            try:
                user_df[col_invoice] = pd.to_datetime(user_df[col_invoice])
                ref_date = user_df[col_invoice].max() + pd.Timedelta(days=1)
                rfm_user = user_df.groupby(col_customer).agg({
                    col_invoice: lambda x: (ref_date - x.max()).days,
                    col_invoice_no: 'nunique',
                    col_quantity: 'sum',
                    col_unitprice: 'mean'
                }).rename(columns={
                    col_invoice: 'Recency',
                    col_invoice_no: 'Frequency',
                    col_quantity: 'QuantitySum',
                                        col_unitprice: 'AvgUnitPrice'
                })

                # Calculate Monetary as total spending (QuantitySum * AvgUnitPrice)
                rfm_user['Monetary'] = rfm_user['QuantitySum'] * rfm_user['AvgUnitPrice']

                st.subheader("RFM Summary")
                st.dataframe(rfm_user[['Recency', 'Frequency', 'Monetary']].head(10))

                st.success("RFM analysis completed successfully!")

                # Optionally, you can add clustering or other analyses here...

            except Exception as e:
                st.error(f"Error during RFM analysis: {e}")

st.markdown("---")
st.markdown("2025 Nastassia Pukelik | CIS9660 | Baruch College")
