================================================================
        AI SMARTCOURSE - Course Recommendation System
        CS619 Final Project
        Supervisor: Muhammad Bilal
================================================================

PROJECT OVERVIEW
----------------
AI SmartCourse is an intelligent course recommendation system
that uses two AI models (TF-IDF and Neural) to recommend
relevant courses based on user queries.

----------------------------------------------------------------

FEATURES
--------
- Course Recommendation using TF-IDF Model
- Course Recommendation using Neural Model (Sentence Transformers)
- Model Comparison (TF-IDF vs Neural side by side)
- Search History (saved in database)
- Save Recommendations (for later reference)
- User Dashboard
- Delete Saved Recommendations

----------------------------------------------------------------

PROJECT STRUCTURE
-----------------
Smartcourse/
├── app.py                  -> Main Flask Application
├── db_init.py              -> Database Initialization
├── preprocess.py           -> Data Preprocessing Script
├── evaluate.py             -> Model Evaluation Script
├── requirements.txt        -> Required Libraries
├── database.db             -> SQLite Database
├── data/
│   ├── courses.csv         -> Original Dataset
│   └── clean_courses.csv   -> Preprocessed Dataset
├── templates/
│   ├── index.html          -> Home Page
│   ├── dashboard.html      -> Dashboard Page
│   ├── recommand.html      -> Recommendation Page
│   └── about.html          -> About Page
└── static/
    └── style.css         -> Stylesheet for index page
    └── styleRe.css         -> Stylesheet for recommendation page
    └── styleAb.css         -> Stylesheet for about page
    └── styleDa.css         -> Stylesheet for dashboard pae
----------------------------------------------------------------

REQUIREMENTS
------------
- Python 3.10 or above
- pip (Python package manager)

----------------------------------------------------------------

INSTALLATION & SETUP
--------------------

Step 1: Open terminal and go to project folder
        cd "D:\Smartcourse complete\Smartcourse"

Step 2: Create virtual environment
        python -m venv .venv

Step 3: Activate virtual environment
        .venv\Scripts\activate

Step 4: Install required libraries
        pip install -r requirements.txt

Step 5: Initialize database
        python db_init.py

Step 6: Preprocess dataset
        python preprocess.py

Step 7: Run the application
        python app.py

Step 8: Open browser and go to
        http://127.0.0.1:5000

----------------------------------------------------------------

MODELS USED
-----------
1. TF-IDF (Term Frequency - Inverse Document Frequency)
   - Fast and lightweight
   - Keyword-based matching
   - Library: scikit-learn

2. Neural Model (Sentence Transformers)
   - Deep learning based
   - Understands meaning and context
   - Model: all-MiniLM-L6-v2
   - Library: sentence-transformers

----------------------------------------------------------------

DATABASE TABLES
---------------
1. history
   - Stores all search queries
   - Columns: id, query, model, timestamp

2. saved_recommendations
   - Stores user saved courses
   - Columns: id, course_name, department,
              description, relevance, timestamp

----------------------------------------------------------------

HOW TO USE
----------
1. Go to Home page
2. Enter your query (e.g. "I want to learn Python")
3. Select model (TF-IDF or Neural)
4. Click Recommend
5. Save any course you like
6. View saved courses on Dashboard
7. Compare both models on Dashboard

----------------------------------------------------------------

EVALUATION
----------
To run model evaluation:
        python evaluate.py

Metrics used:
- Precision@K
- Recall@K
- Hit Rate

----------------------------------------------------------------

DEVELOPED BY
------------
Project  : AI SmartCourse
Course   : CS619 - Final Year Project
Supervisor: Muhammad Bilal
Year     : 2025

================================================================