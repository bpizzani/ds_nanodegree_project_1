import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#defining some utils functions

def total_count(df,col1,col2,look_for):
    from collections import defaultdict
    new_df = defaultdict(int)
    for val in look_for:
        for idx in range(df.shape[0]):
            if val in df[col1][idx]:
                new_df[val] += df[col2][idx]
                
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1,col2]    
    new_df.sort_values("count",ascending = False, inplace=True)
    return new_df


def clean_and_plot(df,column, possible_vals, title='Method of Educating Suggested', plot=True):
    '''
    INPUT 
        df - a dataframe holding the CousinEducation column
        title - string the title of your plot
        axis - axis object
        plot - bool providing whether or not you want a plot back
        
    OUTPUT
        study_df - a dataframe with the count of how many individuals
        Displays a plot of pretty things related to the CousinEducation column.
    '''
    study = df[column].value_counts().reset_index()
    study.rename(columns={'index': 'method', column: 'count'}, inplace=True)
    study_df = total_count(study, 'method', 'count', possible_vals) #t.total_count(study, 'method', 'count', possible_vals) 

    study_df.set_index('method', inplace=True)
    if plot:
        (study_df/study_df.sum()).plot(kind='bar', legend=None);
        plt.title(title);
        plt.show()
    props_study_df = study_df/study_df.sum()
    return props_study_df


def higher_ed(formal_ed_str):
    '''
    INPUT
        formal_ed_str - a string of one of the values from the Formal Education column
    
    OUTPUT
        return 1 if the string is  in ("Master's degree", "Professional degree","Doctoral degree")
        return 0 otherwise
    
    '''
    if formal_ed_str in ("Master's degree","Professional degree","Doctoral degree","Bachelor's degree"):
        return 1
    else:
        return 0
    
def gender_split(gender):
    '''
    INPUT
        gender - a string of one of the values from the Gender column
    
    OUTPUT
        return Male or Female if the string is  in just one Gender option
        return Other otherwise
    
    '''    
    if gender == "Male":
        return "Male"
    elif gender == "Female":
        return "Female"
    elif ( (pd.Series(gender).str.contains("Male"))[0] and ((pd.Series(gender).str.contains("Other"))[0]  or (pd.Series(gender).str.contains("Transgender"))[0] or (pd.Series(gender).str.contains("non-conforming"))[0])) :
        return "Other"
    elif   pd.Series(gender).str.contains("Female")[0] and (pd.Series(gender).str.contains("Other")[0]  or  pd.Series(gender).str.contains("Transgender")[0] or pd.Series(gender).str.contains("non-conforming")[0]):
        return "Other"
    else:
        return "Other"
    

def race_split(x):
    if x == "White or of European descent":
        return "White or of European descent"
    elif x == "I prefer not to say":
        return "I prefer not to say"
    elif x == "South Asian":
        return "South Asian"
    elif x == "Hispanic or Latino/Latina":
        return "Hispanic or Latino/Latina"
    elif x == "East Asian":
        return "East Asian"
    elif x == "Middle Eastern":
        return "Middle Eastern"
    elif x == "Black or of African descent":
        return "Black or of African descent"
    elif x != "White or of European descent" and x != "I prefer not to say":
        return "Other"
    
    
def white_race_bool(x):
    if x == "White or of European descent":
        return 1
    else:
        return 0
    
    
def prepare_x_y(df,label):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    

    # Drop rows with missing salary values
    df = df.dropna(subset=["Salary","Gender","Race"],axis=0)
    y = df[label]

    #Drop respondent and expected salary columns
    #df = df.drop(["'Respondent', 'ExpectedSalary', 'Salary'"], axis=1)
    df['race_clean'] = df["Race"].apply(race_split)
    df['gender_clean'] = df["Gender"].apply(gender_split)
    df["white_race"] = df["race_clean"].apply(white_race_bool)
    

    df = df[["white_race","FormalEducation","gender_clean","MajorUndergrad","JobSatisfaction","HoursPerWeek"]]
    
    num_variables = df.columns[df.dtypes != 'object']
    cat_variables = df.columns[df.dtypes == 'object']
    
    df[num_variables] = df[num_variables].apply(lambda col: col.fillna(col.mean()),axis=0)
    
    for cat in cat_variables:
        df = pd.concat([df.drop(cat, axis=1), pd.get_dummies(df[cat], prefix=cat,prefix_sep="_", drop_first=True)], axis=1)
    
    X = df
    return X, y


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df
