import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from feature_eng import (get_intial_analysis, 
                         get_power_stats)


data_inv = pd.read_csv('anonymized_inventory_data.csv')
data_order = pd.read_csv('anonymized_order_data.csv')

data_inv['inv_date_col'] = pd.to_datetime(data_inv['inv_date_col'])
data_order['order_date_col'] = pd.to_datetime(data_order['order_date_col'])

complete_order_data = pd.merge(
    data_order, 
    data_inv, 
    left_on = ['join_col'],      
    right_on = ['join_col'],     
    how = 'left')

################################################################################################


#### Figure B ####
cat_feat_1_crit = complete_order_data['order categorical feature 1'] == 'order categorical feature 1 #1'   

complete_order_data[cat_feat_1_crit].groupby(
    [pd.Grouper(key = 'order_date_col', freq = 'w'),            
    'inventory categorical feature 1'])['inventory categorical feature 2'].count().unstack().iloc[:-2].plot()   
plt.title("inventory categorical feature 1 by Week")                  
plt.ylabel("Count")                                         
plt.xlabel('Week')                                         


#### Figure C ####

# Incomplete data?
complete_order_data.groupby(
    [pd.Grouper(key = 'order_date_col', freq = 'w'),    
    'order categorical feature 1'])['inventory categorical feature 2'].count().unstack().plot()     
plt.title("order categorical feature 1 by Week")                  
plt.ylabel("Count")                                         
plt.xlabel('Week')   



################################################################################################
# Exploratory Analysis: spikes in stolen items (see writeup)

cat_feat_1_crit = complete_order_data['order categorical feature 1'] == 'order categorical feature 1 #8'     

complete_order_data[cat_feat_1_crit].groupby(
    [pd.Grouper(key = 'order_date_col',                         
    freq = 'w')])['order categorical feature 2'].count().iloc[:-1].plot()  
plt.title("Number of Cat. Feat. #2 by Week")                       
plt.ylabel("Count")                                        
plt.xlabel('Week')                                          
plt.legend([])

complete_order_data['inventory categorical feature 6'] = complete_order_data['inventory categorical feature 6'].astype(      
    str).apply(lambda x: x.lower())

complete_order_data[cat_feat_1_crit].groupby(
    [pd.Grouper(key = 'order_date_col',                         
    freq = 'w'),
    'inventory categorical feature 6'])['order categorical feature 2'].count().unstack().plot()   
plt.legend([])

################################################################################################
# Exploratory Analysis: number of returns completed is unusual (see writeup)

return_crit = complete_order_data['order categorical feature 1'] == 'order categorical feature 1 #4'   

returns_by_code = complete_order_data[return_crit].groupby(
    'join_col')['inventory categorical feature 2'].count().reset_index()                 

returns_by_code = returns_by_code.rename(
                            columns = {'inventory categorical feature 2':'num_of_cat_feat_2'})                           

################################################################################################
# Beginning of Main Project

data = pd.merge(data_inv, returns_by_code,                              
                left_on = ['join_col'],                           
                right_on = ['join_col'], how = 'left')            

# Potentially a target variable for a regression
data['change_in_value'] = data['inventory quantitative feature 3'] - data['inventory quantitative feature 1']    

# Target variable
data['increase_in_value'] = (data['change_in_value'] > 0).astype(int)   

# Descriptive stuff         
#                           

target_var = 'increase_in_value'   

# Potential features
data['day_of_week_bought'] = pd.to_datetime(data['inv_date_col']).dt.strftime('%A')
dow_df = get_intial_analysis(
    data, 
    target_var = target_var, 
    potential_feature = 'day_of_week_bought')       

# Consider looking into the weekend issue           
part_of_week_func = lambda x: 'weekend' if x in ['Saturday', 'Sunday'] else 'weekday'

data['part_of_week'] = data['day_of_week_bought'].apply(part_of_week_func)      

weekend_df = get_intial_analysis(
    data, 
    target_var = target_var, 
    potential_feature = 'part_of_week')         

init_potential_categ_vars = ['part_of_week']    

# These features not especially interesting/significant
hours_type_func = lambda x: 'nonbusiness_hours' if x else 'business_hours'
data['hours_type'] = ((data['inv_date_col'].dt.hour >= 18) & (data['part_of_week'] != 'weekend')).apply(hours_type_func)
nonbusi_df = get_intial_analysis(data, 
                             target_var = target_var, 
                             potential_feature = 'hours_type')
init_potential_categ_vars += ['hours_type']

# Further notes
""" num_of_cat_feat_2 -> Meaningless
    inventory categorical feature 1 -> great
    inventory categorical feature 6 -> Really good
    inventory categorical feature 4 -> pretty good
"""
# Data cleaning

data['num_of_cat_feat_2'] = data['num_of_cat_feat_2'].fillna(0)                  
data['inventory categorical feature 6'] = data['inventory categorical feature 6'].astype(str).apply(lambda x: x.lower())  

# Further work: Could do fuzzymatch here or some other text cleaning
other_potential_categ_vars = ['num_of_cat_feat_2', 'inventory categorical feature 1', 'inventory categorical feature 6', 'inventory categorical feature 4']
for other_potential_categ_var in other_potential_categ_vars:
    # other_potential_categ_var = 'inventory categorical feature 6'
    init_analysis_df = get_intial_analysis(
        data, 
        target_var = target_var, 
        potential_feature = other_potential_categ_var)
    print('\t\t\t', other_potential_categ_var.upper(), '\n')
    print(init_analysis_df, '\n\n')

# Does not seem to matter
other_potential_categ_vars.remove('num_of_cat_feat_2')
y = data[target_var]
X = pd.DataFrame()

# Feature selection using statistical power
remaining_categorical_vars = init_potential_categ_vars + other_potential_categ_vars
power_thresh = 0.80
for remaining_categorical_var in remaining_categorical_vars:
    # Test group is one of the 'x' categories, control group is the other 'x - 1' categories
    # How much of the test group's curve is past the 2.5th / 97.5th percentile of the control
    dir_power_stats = get_power_stats(data, 
                                      feature = remaining_categorical_var, 
                                      target_var = target_var)
    total_num_of_categories = len(dir_power_stats)
    
    # To avoid dummy variable trap
    if total_num_of_categories <= 2:
        max_num_of_categories = total_num_of_categories - 1
        sub_dir_power_stats = dir_power_stats.tail(max_num_of_categories)    
    
    else:
        dir_power_stat_name = f'dir_power_stat_{remaining_categorical_var}'
        power_thresh_crit = power_thresh <= dir_power_stats[dir_power_stat_name].apply(abs)
        sub_dir_power_stats = dir_power_stats[power_thresh_crit]

        # To avoid dummy variable trap
        if total_num_of_categories == len(sub_dir_power_stats): 
            # Dropping the category with the least amount of statistical power
            weakest_power_category = sub_dir_power_stats[dir_power_stat_name].apply(abs).min()
            dummy_var_trap_crit = sub_dir_power_stats[dir_power_stat_name] != weakest_power_category
            sub_dir_power_stats = sub_dir_power_stats[dummy_var_trap_crit]

    selected_categories = list(sub_dir_power_stats[remaining_categorical_var])
        
    final_categ_df = pd.get_dummies(data[remaining_categorical_var])[selected_categories]

    X = pd.concat([X, final_categ_df], axis = 1) 

# Adding in price paid as a feature         
X['inventory quantitative feature 1'] = data['inventory quantitative feature 1']        

# Used later in post-mortem                 
mod_df = pd.concat([X, y], axis = 1)


# Note: ideally split train / test W/R/T time 
# Note: ideally test different hyperparameters, perform grid search cross validation





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

result_df = y_test.to_frame()
# Not great
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
result_df['pred'] = log_reg.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, result_df['pred'])

result_df.groupby(target_var)['pred'].hist(alpha = 0.7)
plt.title("Logistic Regression Prediction, Broken Down by Actual Outcome")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.legend([f'did_not_{target_var}', target_var])
plt.show()

# Takes into account interactions (i.e., creates its own interaction terms (e.g., Yeezy AND Goat))
dec_tree = DecisionTreeClassifier(max_depth = 10)
dec_tree.fit(X_train, y_train)
result_df['pred'] = dec_tree.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, result_df['pred'])


result_df.groupby(target_var)['pred'].hist(alpha = 0.7)
plt.title("Decision Tree Prediction, Broken Down by Actual Outcome")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.legend([f'did_not_{target_var}', target_var])
plt.show()

# Introduces some bias and generalizes better b/c it's taking subsample of features 
# and observations
rf_clf = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state=0)
rf_clf.fit(X_train, y_train)

result_df['pred'] = rf_clf.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, result_df['pred'])

result_df.groupby(target_var)['pred'].hist(alpha = 0.7)
plt.title("Random Forest Prediction, Broken Down by Actual Outcome")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.legend([f'did_not_{target_var}', target_var])
plt.show()



feature_importance = pd.DataFrame(list(zip(X.columns, rf_clf.feature_importances_)),
                                  columns = ['Feature', 'Importance']
                                  ).sort_values('Importance', ascending = False
                                  ).set_index('Feature')

##### Figure E #####
feature_importance

##### Figure A #####

target_name = ' '.join(elem.capitalize() for elem in target_var.split('_'))
data.groupby(target_var)['inventory quantitative feature 1'].hist(alpha = 0.7)
plt.legend([f'Did Not {target_name}', target_name])
plt.axvline(-0.9,linestyle= '--', color = 'black', alpha = 0.7)
plt.axvline(0.15,linestyle= '--', color = 'black', alpha = 0.7)
plt.ylabel('Count')
plt.xlabel('inventory quantitative feature 1')
plt.title("Looking into Inventory Quantitative Feature 1")

##### Figure D #####
data.groupby(target_var)[['inventory quantitative feature 1']].describe()
# Looking into the most important features
data.groupby(target_var)['inventory quantitative feature 1'].hist(alpha = 0.7)
plt.legend([f'Did Not {target_name}', target_name])
plt.axvline(-0.9,linestyle= '--', color = 'black', alpha = 0.7)
plt.axvline(0.15,linestyle= '--', color = 'black', alpha = 0.7)
plt.ylabel('Count')
plt.xlabel('inventory quantitative feature 1')
plt.title("Inventory Quantitative Feature 1 Bivariate Exploratory Analysis")

get_intial_analysis(data, 
                    target_var = target_var, 
                    potential_feature = 'inventory categorical feature 1').loc[["inventory categorical feature 1 #10"]]

get_intial_analysis(data, 
                    target_var = target_var, 
                    potential_feature = 'inventory categorical feature 6').loc[['inventory categorical feature 6 #30']]

#get_intial_analysis(data, 
#                    target_var = target_var, 
#                    potential_feature = 'inventory categorical feature 6').loc[['inventory categorical feature 6 #28']]