# Social Insights Dashboard

A Dash-based interactive dashboard to visualize social media posts and comments, including **engagement metrics**, **reaction breakdown**, and **sentiment/emotion/tone/intent analysis** extracted from scraped JSON data.

---


## Features

- Displays **post content** at the top.
- Shows **key engagement metrics**: Likes, Shares, Comments, Total.
- Visualizes **reaction breakdown** (pie chart and bar chart).
- Visualizes **comment analysis**:
  - Sentiment (pie chart)
  - Emotion (pie chart & bar chart)
  - Tone (pie chart)
  - Intent (pie chart)
- Shows **top commenters**, **comment lengths**, **word frequency**, **hashtags**, **post length vs engagement**.
- Heatmap for **emotion vs sentiment** (if data available).
- Preview of **top comments** with sentiment/emotion/tone/intent info.
- Auto-reloader enabled during development for faster iteration.

---

## Project Structure
project-root/

│

├── ScrappedData/ # Contains candidate subfolders with JSON files

│ ├── candidate1/

│ │ ├── post1.json

│ │ ├── post2.json

│ └── candidate2/

│ │ ├── post1.json

│

├── social_insights_dashboard.py # Main Dash application

├── README.md

└── requirements.txt
## Installation

 **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
# Activate the environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Install dependencies:

pip install -r requirements.txt

