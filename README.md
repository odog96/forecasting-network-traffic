### Project overview


### Folder structure 

**data-clean** 
    - fills missing values
    - cleans host names (trims down)
    - removes zero only variables
    - removes single values 
    - remove error / packets fields

**eda file** 
    - looks at time series charts for all site / host combinations (at 5 min intervals)
    - looks at histograms/box plots/ correlation charts
    - removes initial set of site / host based on sparsity of spikes
    - aggregates to hour data & looks as same charts as above
    - removes a 2nd set of site/host combination based on lack of distribution

**model_lstm** 
    - includes feature identification
    - scaling / normalization
    - parameter set up
    - build 
     
**serve.py**
    - file used by CML model for model deploy
    - '5-min' forecasts with model based on 5 minute intevals
    - 'hourly' forecasts with model based on hourly data
    
**train_models_exp.py**
    - ML flow file with experiments

**retrain.py**
    - productionized model training
    - no ML flow included
    
**check_model.py**
    -
    
**get_predictions.py**
    -
    
**helper_functions.py**
    - houses tranformation functions for other scripts including model training. 

**train.py**
    - includes MLFLOW experiments integrated into training
    - v2 moves all functions to helper_functions.py 
    - v2 also adds api functionality for model registry



### existing issues

1. tracked aggregated  metrics needs update. Currently createing new recored
2. tracked delayed metrics is not loading ground truth for each record