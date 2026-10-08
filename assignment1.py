import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures as PolyFeat, StandardScaler as StdScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
import warnings

warnings.filterwarnings('ignore')
STUDENT_ID = "BT2024172"

def execute_hybrid_pipeline(task_number, max_poly_deg, use_lasso=False):
    model_type = "Lasso (L1)" if use_lasso else "Ridge (L2)"
    print(f">>> Initiating {model_type} Regression for variant {task_number} <<<")
    
    train_path = f"{STUDENT_ID}_train_var{task_number}.csv"
    test_path = f"{STUDENT_ID}_test_var{task_number}.csv"
    out_path = f"{STUDENT_ID}-pred_var{task_number}.csv"
    # load the data sets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    y_train = df_train['y']
    X_train = df_train.drop(columns=['y'])
    
    X_test = df_test.drop(columns=['y']) if 'y' in df_test.columns else df_test
        
    best_deg = 1
    highest_r2 = -float('inf')
    lowest_mse = float('inf')
    
    # cross-validation 80-20 split
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=101)
    # define a range of penalty values (alphas) for the Ridge regression
    alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    # loop through all the polynomial degrees 
    for d in range(1, max_poly_deg + 1):
        if use_lasso:
            regressor = LassoCV(cv=3, random_state=42, max_iter=2500)
        else:
            regressor = RidgeCV(alphas=alpha_range)
        # build our machine learning model pipeline:
        # 1. Scale features so they have a mean of 0 and variance of 1
        # 2. Generate polynomial features (combinations of our variables up to degree 'd')
        # 3. Apply the regression model  
        steps = [
            ('scaler', StdScaler()),
            ('poly', PolyFeat(degree=d, include_bias=False)),
            ('regressor', regressor)
        ]
        pipe = Pipeline(steps)
        cv_results = cross_validate(
            pipe, X_train, y_train, 
            cv=cv_splitter, 
            scoring=('r2', 'neg_mean_squared_error')
        )
        
        avg_r2 = np.mean(cv_results['test_r2'])
        avg_mse = -np.mean(cv_results['test_neg_mean_squared_error'])
        
        print(f"Eval Degree {d}: R2 = {avg_r2:.4f} | MSE = {avg_mse:.4f}")
        
        if avg_r2 > highest_r2:
            highest_r2 = avg_r2
            lowest_mse = avg_mse
            best_deg = d
            
    print(f"\n=> Best {model_type} Model for var{task_number}: Degree {best_deg} (R2: {highest_r2:.4f}, MSE: {lowest_mse:.4f})\n")
    
    final_regressor = LassoCV(cv=3, random_state=42, max_iter=2500) if use_lasso else RidgeCV(alphas=alpha_range)
    # Rebuild the pipeline using the absolute best degree we found
    final_pipe = Pipeline([
        ('scaler', StdScaler()),
        ('poly', PolyFeat(degree=best_deg, include_bias=False)),
        ('regressor', final_regressor)
    ])
    
    final_pipe.fit(X_train, y_train)
    # generate our final predictions for the unseen test data
    final_preds = final_pipe.predict(X_test)
    
    output_dataframe = pd.DataFrame({'y': final_preds})
    output_dataframe.to_csv(out_path, index=False)
    print(f"=> Successfully generated {out_path}\n")

if __name__ == "__main__":
    execute_hybrid_pipeline(1, 10, use_lasso=True)
    execute_hybrid_pipeline(2, 20, use_lasso=False)