# Cognifyz Restaurant Analysis Project

## Internship
Cognifyz Technologies - Machine Learning Internship

### Task 1: Restaurant Rating Prediction

#### Problem

The goal of this project was to predict the **aggregate rating of a restaurant** using information available in the restaurant dataset. This can help understand which restaurant characteristics may be associated with higher or lower ratings.

#### Approach

The project followed a basic machine learning workflow:

**Data → Cleaning → EDA → Features → Model → Evaluation**

* **Data:** Used the restaurant dataset provided by Cognifyz Technologies.
* **Cleaning:** Prepared the dataset by handling missing values and organizing the required variables for analysis.
* **EDA:** Explored relationships between restaurant ratings and variables such as votes, average cost, and price range.
* **Features:** Used features including **average cost for two, price range, and votes** to predict restaurant ratings.
* **Model:** Built a regression model to predict the aggregate rating.
* **Evaluation:** Evaluated the model using appropriate regression evaluation metrics.

#### Key Finding

**Finding:** Restaurants with a higher number of votes tended to have more reliable and informative aggregate ratings, making votes an important feature to consider when analyzing restaurant ratings.
#### Business Value

The prediction model could potentially help restaurants understand the factors associated with customer ratings and identify areas that may influence their overall rating. It could also support customers or businesses in making data-driven decisions when comparing restaurants.


## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Folium
- VS Code
- GitHub


## How to Run

Clone the repository:

```bash
git clone https://github.com/fathimadiyana/Cognifyz-Restaurant-Analysis.git
cd Cognifyz-Restaurant-Analysis




Install dependencies:

pip install -r requirements.txt

Run the Streamlit dashboard:

python -m streamlit run app.py
