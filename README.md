🏨 Hotel Booking Q&A + Analytics (Solvei8 Assignment)
This project was built as part of the Solvei8 AI/ML Internship Assignment.
It is a smart system that lets you ask natural questions about hotel bookings and also shows analytics based on those answers.

🔧 What this project does
❓ Ask booking-related questions like:

“Show me bookings from France with 2 guests”

📌 Shows real bookings that match your question

📊 Gives analytics based only on those results
(like average price, cancellation rate, top countries, revenue months)

🛠️ Tech Stack Used
Python + Flask (Backend API)

Sentence Transformers + FAISS (Search system)

Pandas (Data handling & analytics)

HTML + JS (Frontend UI)

No external APIs or keys used — everything runs locally

📂 Folder Structure
bash
Copy
Edit
solvei8-assignment/
├── app.py                # Main backend code (Flask)
├── hotel_bookings.csv    # Dataset
├── project.ipynb         # Optional: EDA / trials
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend code
