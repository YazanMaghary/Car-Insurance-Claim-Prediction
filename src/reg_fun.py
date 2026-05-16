from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score , root_mean_squared_error
import pandas as pd

def reg_metrics (y_true , y_pred , label = '' , verbose = True , output_dict = False):
    mae  = mean_absolute_error(y_true , y_pred)
    mse  = mean_squared_error(y_true , y_pred)
    rmse = root_mean_squared_error(y_true , y_pred)
    r2 = r2_score(y_true ,y_pred)
    if verbose : 
        print('-'*60 , f"Regression Metrics : {label}" , '-'*60 ,sep='\n')
        print(f"MAE : {mae : ,.3f}")
        print(f"MSE : {mse : ,.3f}")
        print(f"RMSE : {rmse : ,.3f}")
        print(f"R2 : {r2 : ,.3f}")
    if output_dict:
        metrics = {'Label' : label , 'MAE' : mae , "MSE" : mse , 'RMSE' : rmse , 'R2' : r2}
        return metrics

def evaluate_regression(reg , X_train , y_train ,X_test ,y_test , verbose = True , output_frame = False) :
    y_train_pred = reg.predict(X_train)
    results_train = reg_metrics(y_train , y_train_pred , label = 'Train Metrics' , output_dict=output_frame , verbose=verbose)
    y_test_pred = reg.predict(X_test)
    results_test = reg_metrics(y_test , y_test_pred , label = 'Test Metrics' , output_dict=output_frame , verbose=verbose)
    if output_frame:
        result_df = pd.DataFrame([results_train , results_test])
        result_df = result_df.set_index('Label')
        result_df.index.name = None
        return result_df.round(3)