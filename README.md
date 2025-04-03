# 🏨 Hotel Booking Q&A + Analytics (Solvei8 Assignment)

This project was built as part of the Solvei8 AI/ML Internship Assignment.
It is a smart system that lets you ask natural questions about hotel bookings and also shows analytics based on those answers.

 What this project does

❓ Ask booking-related questions like:

“Show me bookings from France with 2 guests”

"Tell me bookings that got cancelled from Spain"

"Bookings with long lead time from Germany"

"Any bookings from the UK in July?"

"Give me bookings with babies from Italy"



# Shows real bookings that match your question

Gives analytics based only on those results
(like average price, cancellation rate, top countries, revenue months)


# Tech Stack Used
Python + Flask (Backend API)

Sentence Transformers + FAISS (Search system)

Pandas (Data handling & analytics)

HTML + JS (Frontend UI)

No external APIs or keys used — everything runs locally

# Folder Structure

solvei8-assignment/

├── app.py                # Main backend code (Flask)

├── hotel_bookings.csv    # Dataset

├── project.ipynb         # Optional: EDA / trials

├── requirements.txt      # Python dependencies

├── templates/
│   └── index.html        # Frontend code



# Libraries to Install

flask --------->>>>>> For creating the backend API server

pandas --------->>>>>>	For data cleaning, filtering, and analytics

numpy --------->>>>>>	For array handling (used with embeddings)

sentence-transformers --------->>>>>>	To convert booking descriptions and queries into embeddings

faiss-cpu --------->>>>>>	For fast semantic search (nearest neighbor search)

pycountry --------->>>>>>	To convert country codes like “FRA” to “France”



![image](https://github.com/user-attachments/assets/91896a94-511c-4a80-8130-f26c19f5d8c7)
