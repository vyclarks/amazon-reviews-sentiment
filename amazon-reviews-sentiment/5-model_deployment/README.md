
# Sentiment Analysis Model Deployment
This project deploys a sentiment analysis model that predicts customer sentiment in real-time. It provides a Flask-based API and a web interface for users to input reviews or upload CSV files for batch predictions.

## 1) Features

- Text Preprocessing: Automatically cleans and preprocesses text data.
- Sentiment Prediction: Classifies reviews as positive, neutral, or negative using a trained LightGBM model.
- Graph Visualization: Displays sentiment distribution graphs.
- Web Interface: Allows users to input reviews and view predictions directly.
- File Upload: Supports bulk review predictions from CSV files.
- File Download: Support download bulk review predictions file.

## 2) Project Structure

amazon-reviews-sentiment/
├── 1-data_collection/
│   └── amazon_scraping.ipynb
├── 2-data_preprocessing/
│   └── data_preprocessing.ipynb
├── 3-exploratory_data_analysis/
│   └── exploratory_data_analysis.ipynb
├── 4-model_development/
│   └── model_development.ipynb
├── 5-model_deployment/
│   ├── __pycache__/
│   └── README.md
├── data/
├── docs/
├── models/
├── requirements.txt
├── static/
│   └── amazon-logo.png
├── templates/
│   └── index.html
├── api.py
├── main.py
└── text_preprocessing.py

### Details
#### 1. Folders:
- 1-data_collection/ to 5-model_deployment/ organize your project phases.
- data/ for storing raw or preprocessed data files.
- docs/ the related documents of this projects
- models/ for saving trained machine learning models.
- static/ for static assets (e.g., images, CSS, JavaScript).
- templates/ for HTML files (used by Flask).
#### 2. Key Files:
- api.py: The Flask API for model deployment.
- main.py: Likely the entry point or a driver script.
- text_preprocessing.py: Script for preprocessing text data.
- requirements.txt: Project dependencies.
- README.md: Project documentation.
- Notebook Files: Each phase of the project has a corresponding Jupyter Notebook (*.ipynb) for tasks like data collection, preprocessing, analysis, and model development.



## 3) Setup Instructions

### 1. Prerequisites
- Python 3.8 or later
- Libraries listed in requirements.txt
### 2. Installation
1. Download the project folder
2. Navigate to the project directory:
```
cd sentiment-analysis-deployment
```
3. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# NLTK data (first run only)
python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','wordnet']]"
```
4. Install the required dependencies:
```
pip install -r requirements.txt
```

## 4) Usage
- Step 1: Run the Flask Application
```
python api.py
```
The application will be available at http://127.0.0.1:5000
- Step 2: Access the Web Interface
+ Open your browser and go to http://127.0.0.1:5000
+ Enter a review to predict its sentiment or upload a CSV file for batch processing.

## 5) Sample Data
Use the included data/sample_reviews.csv for testing batch predictions. The file should have a Review column with customer reviews.

## 6) Visualizations
The application generates pie charts to visualize sentiment distribution, which are displayed on the web interface.

## 7) Technologies Used
- Backend: Flask
- Frontend: HTML, CSS
- Machine Learning: Logistic Regression, Random Forest, Support Vector Machines, Naive Bayes, LightGBM
- Deep Learning: LSTM- RNN based
- Visualization: Matplotlib
- Data Processing: Pandas, NLTK