# Real-Time-Smart-Grocery-Recommendation-System

## 📋 Project Overview

Built an end-to-end grocery recommendation system using the Instacart dataset. The system formulates next-basket prediction as a classification problem and uses machine learning models (Logistic Regression and Gradient Boosting) to generate personalized product recommendations.

### Key Features:
- 🔄 **Complete Pipeline**: Data ingestion → Validation → Transformation → Model Training
- 🎯 **Dual Models**: Logistic Regression (baseline) and Gradient Boosting (primary)
- 💾 **Feature Engineering**: Order gap (recency), user total orders, purchase ratio, product popularity
- 🌐 **Web Interface**: Interactive Streamlit app for viewing recommendations
- 📊 **13.3M Records**: Handles Instacart's massive dataset efficiently

## Getting Started 🚀

### 1. Create and Activate Conda Environment
```powershell
conda create -n grocery-recommendation python=3.10
conda activate grocery-recommendation
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
Generate transformed data and trained models:
```powershell
python main.py
```

This will execute all 4 pipeline stages:
- ✅ Data Ingestion (downloads Instacart dataset)
- ✅ Data Validation (cleans and validates data)
- ✅ Data Transformation (engineers recommendation features)
- ✅ Model Training (trains models and generates recommendations)

**Note:** First run takes 20-30 minutes due to large dataset processing.

### 4. Launch the Web Application
```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🏗️ Project Architecture

```
Real_Time_Smart_Grocery_Recommendation_System/
├── components/
│   ├── data_ingestion.py       # Downloads data from Kaggle
│   ├── data_validation.py      # Cleans and validates data
│   ├── data_transformation.py  # Engineers recommendation features
│   └── model_trainer.py        # Trains models & generates recommendations
├── config/
│   └── configuration.py        # YAML config management
├── entity/
│   └── config_entity.py        # Configuration namedtuples
├── pipeline/
│   └── training_pipeline.py    # Orchestrates 4-stage pipeline
├── logger/
│   └── logger.py               # Custom logging
├── exception/
│   └── exxception_handler.py   # Exception handling
└── utils/
    └── utils.py                # Utility functions
```

## 📁 Data Flow

```
Instacart Dataset (Kaggle)
    ↓
Data Ingestion
    ↓
artifact/dataset/ingested_data/ (CSV files)
    ↓
Data Validation
    ↓
artifact/dataset/clean_data/ (clean_data.csv, pickle)
    ↓
Data Transformation
    ↓
artifact/dataset/transformed_data/ (transformed_data.csv, pickle)
artifact/serialized_objects/ (feature vectors)
    ↓
Model Training
    ↓
artifact/models/ (lr_model.pkl, gb_model.pkl)
artifact/recommendations/ (recommendations.pkl, CSV)
    ↓
Streamlit Web App (app.py)
```

## 🎯 Using the Web Application

### Main Features:

**📊 Top Recommendations Tab**
- View top 10 recommended products for a selected user
- Color-coded confidence levels (High/Medium/Low)
- Reorder probability for each recommendation

**🔍 Search Products Tab**
- Search for specific grocery items
- See reorder probability for any product
- Check if product is in user's purchase history

**📈 Statistics Tab**
- Overall user analytics (items purchased, purchase ratio)
- Recommendation confidence distribution
- Visual charts and metrics

### Workflow:
1. Select a user from the sidebar
2. View their purchase history
3. Explore personalized recommendations
4. Search for specific products
5. Check predicted reorder probabilities

## 🤖 Machine Learning Models

### Feature Engineering:
- **order_gap**: Days since last order (recency)
- **user_total_orders**: Total number of orders by user
- **purchase_ratio**: Percentage of orders where user bought product
- **product_popularity**: Number of users who have purchased product

### Models Trained:
- **Logistic Regression**: Baseline model with balanced class weights
- **Gradient Boosting**: HistGradientBoostingClassifier (primary model)

### Evaluation Metrics:
- AUC Score
- Accuracy
- Precision
- Recall
- F1 Score

## 📊 Dataset Information

**Source**: Instacart Online Grocery Shopping Dataset D2
**Size**: 13.3 million rows across 6 CSV files
**Time Period**: 3+ years of Instacart transaction data

**Files**:
- `orders.csv`: 3.4M orders
- `order_products__prior.csv`: 32M prior order items
- `order_products__train.csv`: 1.38M training order items
- `products.csv`: 49K products
- `aisles.csv`: 134 grocery aisles
- `departments.csv`: 21 departments

## 🚨 Troubleshooting

**Issue**: "Module not found" error
```
Solution: Ensure conda environment is activated and all dependencies installed
conda activate grocery-recommendation
pip install -r requirements.txt
```

**Issue**: Models/recommendations not found
```
Solution: Run the pipeline first to generate artifacts
python main.py
```

**Issue**: Streamlit port already in use
```
Solution: Specify a different port
streamlit run app.py --server.port 8502
```

**Issue**: Out of memory during pipeline
```
Solution: The large dataset requires ~8GB RAM. Ensure sufficient system memory
```

## 📝 Configuration

Edit `config/config.yaml` to customize:
- Dataset paths
- Model hyperparameters
- Output directories
- Kaggle API credentials

## 🔧 Dependencies

- **pandas**: Data manipulation
- **scikit-learn**: Machine learning models
- **streamlit**: Web application framework
- **numpy**: Numerical computing
- **pyYAML**: Configuration management
- **kagglehub**: Download Instacart dataset

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Muhammad Mubashir

---

**Last Updated**: March 9, 2026
   - Check logs in the terminal for errors.

Feel free to modify or extend these instructions based on your setup.
