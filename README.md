# Real-Time-Smart-Grocery-Recommendation-System

## Getting Started 🚀

Follow these step-by-step instructions to run the project:

1. **Create a conda environment**
   ```powershell
   conda create -n Groccery-Store-Recommendation python=3.14
   ```


2. **Activate the conda environment**
   ```powershell
   conda activate Groccery-Store-Recommendation
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Problem Statement**
    - Built an end-to-end grocery recommendation system using the Instacart dataset, formulating next-basket prediction as a multi-label classification and ranking problem.
    Developed baseline heuristics and logistic regression models, followed by gradient boosting and sequence-based deep learning approaches.
    Deployed the final model as a real-time FastAPI service with containerised cloud deployment and evaluation using F1 and Precision@K metrics -
4. **Configure any required settings**
   - Update configuration files or environment variables as needed.
   - See project documentation for details.

5. **Run the application**
   ```powershell
   python main.py
   ```
   *(or the appropriate entry point for this project)*

6. **Visit the application**
   - Open a browser and go to `http://localhost:8000` (or the configured port).

7. **Troubleshooting**
   - Ensure the conda environment is active.
   - Check logs in the terminal for errors.

Feel free to modify or extend these instructions based on your setup.
