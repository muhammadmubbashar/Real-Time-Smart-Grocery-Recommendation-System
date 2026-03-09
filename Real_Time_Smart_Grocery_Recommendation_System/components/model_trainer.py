import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException


class ModelTrainer:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def train_model(self):
        """
        Load transformed data, train models, and generate recommendations
        """
        try:
            logging.info("Loading transformed data for model training")
            
            # Load transformed data from pickle
            with open(self.model_trainer_config.transformed_data_file_path, "rb") as f:
                df = pickle.load(f)
            
            logging.info(f"Loaded transformed data shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            # Prepare features and target
            logging.info("Preparing features and target")
            X = df.drop(columns=["user_id", "product_id", "reordered"])
            y = df["reordered"]
            
            logging.info(f"Feature columns: {X.columns.tolist()}")
            logging.info(f"Target distribution: {y.value_counts().to_dict()}")
            logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Train-test split
            logging.info("Splitting data into train and test sets (80-20)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Train Logistic Regression
            logging.info("Training Logistic Regression model")
            lr_model = LogisticRegression(
                max_iter=200,
                solver='saga',
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
            lr_model.fit(X_train, y_train)
            
            # Evaluate Logistic Regression
            y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
            auc_score_lr = roc_auc_score(y_test, y_pred_proba_lr)
            accuracy_lr = accuracy_score(y_test, lr_model.predict(X_test))
            precision_lr = precision_score(y_test, lr_model.predict(X_test), zero_division=0)
            recall_lr = recall_score(y_test, lr_model.predict(X_test), zero_division=0)
            f1_lr = f1_score(y_test, lr_model.predict(X_test), zero_division=0)
            
            logging.info(f"Logistic Regression Metrics:")
            logging.info(f"  - AUC Score: {auc_score_lr:.4f}")
            logging.info(f"  - Accuracy: {accuracy_lr:.4f}")
            logging.info(f"  - Precision: {precision_lr:.4f}")
            logging.info(f"  - Recall: {recall_lr:.4f}")
            logging.info(f"  - F1 Score: {f1_lr:.4f}")
            
            # Train Gradient Boosting Classifier
            logging.info("Training Gradient Boosting Classifier model - this may take a few minutes")
            gb_model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            logging.info("✓ Gradient Boosting model training completed")
            
            # Evaluate Gradient Boosting
            y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
            auc_score_gb = roc_auc_score(y_test, y_pred_proba_gb)
            accuracy_gb = accuracy_score(y_test, gb_model.predict(X_test))
            precision_gb = precision_score(y_test, gb_model.predict(X_test), zero_division=0)
            recall_gb = recall_score(y_test, gb_model.predict(X_test), zero_division=0)
            f1_gb = f1_score(y_test, gb_model.predict(X_test), zero_division=0)
            
            logging.info(f"Gradient Boosting Metrics:")
            logging.info(f"  - AUC Score: {auc_score_gb:.4f}")
            logging.info(f"  - Accuracy: {accuracy_gb:.4f}")
            logging.info(f"  - Precision: {precision_gb:.4f}")
            logging.info(f"  - Recall: {recall_gb:.4f}")
            logging.info(f"  - F1 Score: {f1_gb:.4f}")
            
            # Select best model
            best_auc = max(auc_score_lr, auc_score_gb)
            if auc_score_gb > auc_score_lr:
                logging.info(f"✓ Gradient Boosting selected as best model (AUC: {auc_score_gb:.4f})")
                best_model = gb_model
                best_model_name = "gb_model"
                best_metrics = {
                    "model_name": "Gradient Boosting Classifier",
                    "auc_score": auc_score_gb,
                    "accuracy": accuracy_gb,
                    "precision": precision_gb,
                    "recall": recall_gb,
                    "f1_score": f1_gb
                }
            else:
                logging.info(f"✓ Logistic Regression selected as best model (AUC: {auc_score_lr:.4f})")
                best_model = lr_model
                best_model_name = "lr_model"
                best_metrics = {
                    "model_name": "Logistic Regression",
                    "auc_score": auc_score_lr,
                    "accuracy": accuracy_lr,
                    "precision": precision_lr,
                    "recall": recall_lr,
                    "f1_score": f1_lr
                }
            
            # Save best model
            logging.info("Saving trained model")
            os.makedirs(self.model_trainer_config.model_dir, exist_ok=True)
            model_path = os.path.join(
                self.model_trainer_config.model_dir,
                f"{best_model_name}.pkl"
            )
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            logging.info(f"✓ Saved {best_model_name} to {model_path}")
            
            # Save all models for comparison
            lr_model_path = os.path.join(self.model_trainer_config.model_dir, "lr_model.pkl")
            with open(lr_model_path, "wb") as f:
                pickle.dump(lr_model, f)
            logging.info(f"✓ Saved Logistic Regression model")
            
            gb_model_path = os.path.join(self.model_trainer_config.model_dir, "gb_model.pkl")
            with open(gb_model_path, "wb") as f:
                pickle.dump(gb_model, f)
            logging.info(f"✓ Saved Gradient Boosting model")
            
            # Save model metrics
            metrics_path = os.path.join(self.model_trainer_config.model_dir, "model_metrics.pkl")
            metrics_dict = {
                "best_model": best_metrics,
                "logistic_regression": {
                    "model_name": "Logistic Regression",
                    "auc_score": auc_score_lr,
                    "accuracy": accuracy_lr,
                    "precision": precision_lr,
                    "recall": recall_lr,
                    "f1_score": f1_lr
                },
                "gradient_boosting": {
                    "model_name": "Gradient Boosting Classifier",
                    "auc_score": auc_score_gb,
                    "accuracy": accuracy_gb,
                    "precision": precision_gb,
                    "recall": recall_gb,
                    "f1_score": f1_gb
                }
            }
            with open(metrics_path, "wb") as f:
                pickle.dump(metrics_dict, f)
            logging.info(f"✓ Saved model metrics")
            
            # Generate recommendations
            logging.info("Generating top-5 recommendations per user")
            
            # Use best model for predictions
            df["reorder_probability"] = best_model.predict_proba(X)[:, 1]
            
            top_k = 5
            recommendations = (
                df.sort_values(['user_id', 'reorder_probability'], ascending=[True, False])
                .groupby('user_id')
                .head(top_k)
            )
            
            logging.info(f"Generated {len(recommendations)} recommendations for {recommendations['user_id'].nunique()} users")
            
            # Save recommendations
            logging.info("Saving recommendations")
            os.makedirs(self.model_trainer_config.recommendations_dir, exist_ok=True)
            
            # Save as pickle
            recommendations_pkl_path = os.path.join(
                self.model_trainer_config.recommendations_dir,
                "recommendations.pkl"
            )
            with open(recommendations_pkl_path, "wb") as f:
                pickle.dump(recommendations, f)
            logging.info(f"✓ Saved recommendations pickle")
            
            # Save as CSV
            recommendations_csv_path = os.path.join(
                self.model_trainer_config.recommendations_dir,
                "recommendations.csv"
            )
            recommendations.to_csv(recommendations_csv_path, index=False)
            logging.info(f"✓ Saved recommendations CSV")
            
            # Save individual user recommendations
            user_recommendations_path = os.path.join(
                self.model_trainer_config.recommendations_dir,
                "user_recommendations.pkl"
            )
            user_recs_dict = {}
            for user_id in recommendations['user_id'].unique():
                user_prods = recommendations[recommendations['user_id'] == user_id][
                    ['product_id', 'reorder_probability']
                ].values.tolist()
                user_recs_dict[int(user_id)] = user_prods
            
            with open(user_recommendations_path, "wb") as f:
                pickle.dump(user_recs_dict, f)
            logging.info(f"✓ Saved user recommendations dictionary")
            
            return {
                "best_model": best_model,
                "best_model_name": best_model_name,
                "best_metrics": best_metrics,
                "recommendations": recommendations,
                "df_with_predictions": df
            }

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise AppException(e, sys) from e

    def initiate_model_training(self):
        try:
            logging.info(f"{'='*20}Model Training log started.{'='*20}")
            self.train_model()
            logging.info(f"{'='*20}Model Training log completed.{'='*20}\n\n")
        except Exception as e:
            raise AppException(e, sys) from e
