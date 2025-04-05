# This is done to import required libraries
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import pycountry

# This is used to start a Flask web application
app = Flask(__name__)
def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except:
        return code  
# This is used to load the dataset
df = pd.read_csv("hotel_bookings.csv")

# This is done to fill missing values in key columns
df.fillna({'children': 0, 'country': 'Unknown', 'agent': 0, 'company': 0}, inplace=True)

# This is done to add new columns: total nights and total guests for each booking
df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']

# This is done to prepare a smaller DataFrame with only required columns for RAG
rag_df = df[['hotel', 'arrival_date_year', 'arrival_date_month', 'country', 'lead_time',
             'is_canceled', 'adr', 'total_nights', 'total_guests']].copy()

# This is used to combine each row into a descriptive text format for semantic search
rag_df['text'] = rag_df.apply(
    lambda row: f"{row['hotel']} hotel booking from {row['country']} for {int(row['total_guests'])} guest(s) staying {int(row['total_nights'])} night(s) in {row['arrival_date_month']} {row['arrival_date_year']}. Lead time: {row['lead_time']} days. Price: {row['adr']}. Canceled: {bool(row['is_canceled'])}.",
    axis=1
)

# This is done to reduce data size and speed up the project (sample 1000 rows)
sample_rag = rag_df.sample(1000, random_state=42).reset_index(drop=True)
texts = sample_rag['text'].tolist()

# This is used to load the MiniLM model for generating sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# This is done to generate embeddings for all booking descriptions
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64).astype('float32')

# This is used to make FAISS use 4 CPU threads for faster indexing
faiss.omp_set_num_threads(4)

# This is used to initialize and build the FAISS index for fast retrieval
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# This route is used to load the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# This is the POST endpoint to get relevant bookings for a user's question
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("question", "")  # Get question from frontend
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    _, indices = index.search(query_embedding, 3)  # Get top 3 similar results

    results = []
    rows = []
    for i in indices[0]:
        raw = sample_rag.iloc[i]
        rows.append(raw.to_dict())  # Save raw row for contextual analytics
        country_full = get_country_name(raw["country"])
        canceled = " Cancelled" if raw["is_canceled"] else "Confirmed"

        # Create a human-friendly answer
        formatted = (
            f"{raw['hotel']} – {int(raw['total_guests'])} guests – {int(raw['total_nights'])} nights\n"
            f" Country: {country_full}  📅 Month: {raw['arrival_date_month']} {raw['arrival_date_year']}\n"
            f" Lead Time: {raw['lead_time']} days  💰 Price: ₹{raw['adr']}\n"
            f" Status: {canceled}"
        )
        results.append(formatted)

    # Return results and raw rows for frontend use
    return jsonify({
        "matches": results,
        "rows": rows
    })

# This is the POST endpoint to generate analytics from either full or filtered data
@app.route("/analytics", methods=["POST"])
def analytics():
    data = request.json.get("rows")  # Get context rows if any
    context_df = pd.DataFrame(data) if data else df.copy()  # Use filtered or full data

    # Handle empty edge case
    if context_df.empty:
        return jsonify({
            "average_booking_price": 0,
            "cancellation_rate": "0%",
            "top_booking_countries": {},
            "top_revenue_months": {}
        })

    # This is done to calculate revenue per booking
    context_df['revenue'] = context_df['adr'] * context_df['total_nights']
    context_df['revenue'] = context_df['revenue'].where(context_df['is_canceled'] == 0, 0)

    # These values are calculated for insights
    cancel_rate = round(context_df['is_canceled'].mean() * 100, 2)
    avg_price = round(context_df['adr'].mean(), 2)

    # This is used to get top 5 countries with most bookings
    top_countries = context_df['country'].value_counts().head(5).to_dict()
    readable_countries = {get_country_name(code): count for code, count in top_countries.items()}

    # This is used to get revenue generated per month (top 5 months)
    revenue_by_month = context_df.groupby('arrival_date_month')['revenue'].sum().sort_values(ascending=False).head(5).to_dict()

    # Final analytics returned to frontend
    return jsonify({
        "average_booking_price": avg_price,
        "cancellation_rate": f"{cancel_rate}%",
        "top_booking_countries": readable_countries,
        "top_revenue_months": revenue_by_month
    })

# This runs the Flask server
if __name__ == "__main__":
    app.run(debug=True)
