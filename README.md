# Project Overview

This project predicts student placement based on various parameters. The model uses machine learning to analyze data and provide insights.

# Flask Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/chauhanpalak1079/placement-predictor.git
   cd placement-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

# Routes
- `/`: Home page
- `/predict`: Endpoint to make predictions
- `/result`: POST endpoint for getting results
- `/about`: About the project

## Model and Templates
- The model is stored in `placement_model.pkl`.
- The HTML templates are located in the `templates` directory.