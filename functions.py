#!/usr/bin/env python
# coding: utf-8

# In[14]:


#general imports that we will need will almost always use - it is a good practice to import all libraries at the beginning of the notebook or script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import lasso_path, SGDRegressor
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# ignore warnings
import warnings
warnings.filterwarnings('ignore')





#set random seed for reproducibility
RSEED = 42
np.random.seed(RSEED)


# Transforms all data to fit the same criteria making it easier to work on.

# In[15]:


def normalize_data(x):
    """ 
    Normalizes string data by converting to lowercase and removing underscores, hyphens, and spaces. 
    Converts 'nan' strings to actual NaN values.
    Args:
        x: The input string to normalize.
    """
    x = str(x)
    x = x.lower()
    x = x.replace("_", "")
    x = x.replace("-", "")
    x = x.replace(" ", "")
    if x == "nan":
        return np.nan
    return x


"Function that automatically looks for the closest match on the valid list, therefore correcting the visible typos."

# In[16]:


def correct_missing_letters(value, valid_list, max_missing=2):
    """
    Corrects values with missing letters based on valid_list
    Args:
        value: The input string to correct.
        valid_list: The list of valid strings to match against.
        max_missing: The maximum number of missing letters allowed.
    """
    best_match = value
    smallest_diff = 999
    if pd.isna(value):  # ignores NaN
        return np.nan
    for ref in valid_list:
        # absolute length difference
        len_diff = abs(len(ref) - len(value))
        if len_diff == 0 or len_diff > max_missing:
            continue  # ignores if equal or difference > limit

        # check if the value is a subsequence of the correct name (maintaining order)
        it = iter(ref)
        is_subseq = all(ch in it for ch in value)

        if is_subseq and len_diff < smallest_diff:
            smallest_diff = len_diff
            best_match = ref

    return best_match

valid_list = []

"""
The remove_outliers function handles abnormal or extreme values in car data to prepare training, validation, and test sets for a machine learning model. It does this without removing any rows from the data, only changing the values within the columns where they are problematic. The treatment follows two main logics. The first is replacement with NaN for impossible or illogical values, for example, years of manufacture prior to 1990, engines with a displacement greater than 6 litres, and also strange combinations such as very new cars with excessively high mileage or large engines with unrealistically low fuel consumption. The problematic value in that cell is replaced with NaN, indicating that it is missing and will be handled later. The second logic is capping or limitation, which is applied to columns such as mileage, mpg consumption, and tax. In this case, instead of replacing values that are above a certain high percentile, such as 98% or 99%, with NaN calculated in the training set, they are cut off and replaced by this upper limit. For mpg, a cut-off is also made at the lower limit to avoid values close to zero or negative. This approach of cutting or replacing with NaN instead of removing the entire row ensures that the size of your training, validation, and test datasets remains the same, but with much cleaner and more consistent data, which is crucial for building a quality model.
"""
# In[ ]:


def remove_outliers(X_train, X_val, X_test, y_train, y_val):
    """
    Deals with outliers:
    -Train/Val/Test: Substitutes impossible values for NaN (does not remove rows)
    -Train/Val/Test: Caps extreme values
    -year < 1990 → NaN 
    -mileage > P95 → capped
    -mpg < P0.5 or > P95 → capped
    -tax > P95 → capped
    -engineSize < 0.5 or > 6.0 → NaN
    -Logical validations:
        -new cars (year >= 2022) with mileage > 100,000 → NaN
        -large engines (> 4.0L) with mpg > 60 → NaN
    Args:
        X_train: Training feature set.
        X_val: Validation feature set.
        X_test: Test feature set.
        y_train: Training target variable.
        y_val: Validation target variable.
    """
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_tst = X_test.copy()
    y_tr = y_train.copy()
    y_v = y_val.copy()
    
    
    # ========== YEAR < 1990 ==========
    """if 'year' in X_tr.columns:
        mask_tr = X_tr['year'] < 1990
        mask_v = X_v['year'] < 1990
        mask_tst = X_tst['year'] < 1990
        
        removed_tr = mask_tr.sum()
        removed_v = mask_v.sum()
        removed_tst = mask_tst.sum()
        
        X_tr.loc[mask_tr, 'year'] = np.nan
        X_v.loc[mask_v, 'year'] = np.nan
        X_tst.loc[mask_tst, 'year'] = np.nan
        
        if removed_tr > 0 or removed_v > 0 or removed_tst > 0:
            print(f"\n[YEAR < 1990]")
            print(f" {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")"""
    

    # ========== MILEAGE (Capping) ==========
    if 'mileage' in X_tr.columns:
        print(f"\n[MILEAGE]")
        
        upper_mileage = X_train['mileage'].quantile(0.99)
        train_above = (X_tr['mileage'] > upper_mileage).sum()
        val_above = (X_v['mileage'] > upper_mileage).sum()
        test_above = (X_tst['mileage'] > upper_mileage).sum()
        
        print(f" P99 = {upper_mileage:,.0f} milhas")
        print(f" Capped: {train_above} train, {val_above} val, {test_above} test")
        
        X_tr['mileage'] = np.clip(X_tr['mileage'], 0, upper_mileage)
        X_v['mileage'] = np.clip(X_v['mileage'], 0, upper_mileage)
        X_tst['mileage'] = np.clip(X_tst['mileage'], 0, upper_mileage)
    

    # ========== MPG (Capping) ==========
    if 'mpg' in X_tr.columns:
        print(f"\n[MPG]")
        q_low = X_tr['mpg'].quantile(0.005)
        q_high = X_tr['mpg'].quantile(0.95)
        print(f" [{q_low:.1f}, {q_high:.1f}] MPG (0.5%–98%)")
        
        train_affected = ((X_tr['mpg'] < q_low) | (X_tr['mpg'] > q_high)).sum()
        val_affected = ((X_v['mpg'] < q_low) | (X_v['mpg'] > q_high)).sum()
        test_affected = ((X_tst['mpg'] < q_low) | (X_tst['mpg'] > q_high)).sum()
        
        print(f"  {train_affected} train, {val_affected} val, {test_affected} test")
        
        X_tr['mpg'] = np.clip(X_tr['mpg'], q_low, q_high)
        X_v['mpg'] = np.clip(X_v['mpg'], q_low, q_high)
        X_tst['mpg'] = np.clip(X_tst['mpg'], q_low, q_high)
    

    # ========== TAX (Capping) ==========
    if 'tax' in X_tr.columns:
        print(f"\n[TAX]")
        upper_tax = X_train['tax'].quantile(0.98)
        train_above = (X_tr['tax'] > upper_tax).sum()
        val_above = (X_v['tax'] > upper_tax).sum()
        test_above = (X_tst['tax'] > upper_tax).sum()
        
        print(f"  P98 = £{upper_tax:.0f}")
        print(f"  Capped: {train_above} train, {val_above} val, {test_above} test")
        
        X_tr['tax'] = np.clip(X_tr['tax'], 0, upper_tax)
        X_v['tax'] = np.clip(X_v['tax'], 0, upper_tax)
        X_tst['tax'] = np.clip(X_tst['tax'], 0, upper_tax)
    

    # ========== ENGINE SIZE ==========
    if 'engineSize' in X_tr.columns:
        print(f"\n[ENGINE SIZE]")
        
        mask_tr = ((X_tr['engineSize'] < 0.5))
        mask_v  = ((X_v['engineSize'] < 0.5))
        mask_tst = ((X_tst['engineSize'] < 0.5))
        
        removed_tr = mask_tr.sum()
        removed_v = mask_v.sum()
        removed_tst = mask_tst.sum()
        
        X_tr.loc[mask_tr, 'engineSize'] = np.nan
        X_v.loc[mask_v, 'engineSize'] = np.nan
        X_tst.loc[mask_tst, 'engineSize'] = np.nan
        
        if removed_tr > 0 or removed_v > 0 or removed_tst > 0:
            print(f" Engine > 6.0L: {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    

    # ========== LOGIC VALIDATION ==========
    print(f"\n[Logic Validation]")
    
    # new cars with high mileage (physically impossible)
    if 'year' in X_tr.columns and 'mileage' in X_tr.columns:
        current_year = 2025
        
        mask_tr = (current_year - X_tr['year'] <= 3) & (X_tr['mileage'] > 100000)
        mask_v = (current_year - X_v['year'] <= 3) & (X_v['mileage'] > 100000)
        mask_tst = (current_year - X_tst['year'] <= 3) & (X_tst['mileage'] > 100000)
        
        removed_tr = mask_tr.sum()
        removed_v = mask_v.sum()
        removed_tst = mask_tst.sum()
        
        X_tr.loc[mask_tr, 'year'] = np.nan
        X_v.loc[mask_v, 'year'] = np.nan
        X_tst.loc[mask_tst, 'year'] = np.nan
        
        if removed_tr > 0 or removed_v > 0 or removed_tst > 0:
            print(f" New cars + high km: {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    
    # large engine with high MPG (physically improbable)
    if 'mpg' in X_tr.columns and 'engineSize' in X_tr.columns:
        mask_tr = (X_tr['engineSize'] > 4.0) & (X_tr['mpg'] > 60)
        mask_v = (X_v['engineSize'] > 4.0) & (X_v['mpg'] > 60)
        mask_tst = (X_tst['engineSize'] > 4.0) & (X_tst['mpg'] > 60)
        
        removed_tr = mask_tr.sum()
        removed_v = mask_v.sum()
        removed_tst = mask_tst.sum()
        
        X_tr.loc[mask_tr, 'mpg'] = np.nan
        X_v.loc[mask_v, 'mpg'] = np.nan
        X_tst.loc[mask_tst, 'mpg'] = np.nan
        
        if removed_tr > 0 or removed_v > 0 or removed_tst > 0:
            print(f" large engine + high mpg: {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("="*60)
    print(f"Mantidos: {len(X_tr)} train (100.0%), "
          f"{len(X_v)} val (100.0%), "
          f"{len(X_tst)} test (100.0%)")
    print(f"Nenhuma linha removida - valores impossíveis substituídos por NaN")
    print("="*60 + "\n")
    
    return X_tr, X_v, X_tst, y_tr, y_v


"""
solution for filling in the missing values based on their statistical peers.
Explanation: 
The pipeline begins with high-cardinality categorical variables. In the case of `model`, imputation is performed using global mode, since the categories are numerous and there is insufficient information for reliable conditional imputation. For `Brand`, the function takes a smarter approach: it first learns a model→brand mapping from the available data, and only then uses external reference lists to identify the correct brand when the model does not exist in the training sample. This step avoids systematic errors, such as associating an ‘Astra’ with BMW, and produces semantically consistent values. When no rule can resolve the missing value, the global mode is applied as a last resort.
This is followed by the imputation of conditional categorical variables, namely `fuelType` and `transmission`. Here, the function calculates the mode by logically relevant groups: for `fuelType` it uses the `brand`, and for `transmission` it uses the combination (`Brand`, `fuelType`), then resorts to the fallback for the mode by brand and finally to the global mode. This strategy respects actual patterns in the automotive market — for example, the fact that certain brands and fuel types tend to have consistent transmissions — avoiding random or structurally inconsistent imputations.
Binary variables, such as `has_reported_damage`, are imputed by mode, which is appropriate for attributes with only two possible states.
Before proceeding to numerical imputation, the function ensures that validation and testing do not introduce new categories that do not exist in the training data. Unknown categorical values are replaced by the mode of the respective column.
The most sophisticated step in the pipeline handles numeric variables with missing values using Iterative Imputer (MICE) with a RandomForestRegressor as the estimator. This method models each numeric variable with missing values as a function of the others, iteratively, capturing non-linear relationships and preserving important correlations between attributes — such as the relationship between vehicle age, mileage, engine size, or fuel consumption. In order for MICE to integrate categorical variables, these are temporarily converted into numerical codes consistent with the training values, ensuring consistency. After imputation, only the values of the numerical columns are replaced, keeping the original categories intact. 
Finally, the function applies plausibility validations (‘clipping’). These limits ensure that the imputed values remain within the physically possible or commercially realistic range — for example, years between 1990 and 2025, engine capacity between 0.5L and 10L, mileage not negative and below the expected upper limit. This prevents subsequent models from dealing with absurd values or artefacts produced by iterative imputation.
"""
# In[ ]:





def impute_missing_values_hybrid(X_train, X_val, X_test):
    '''
    Hybrid intelligent imputation:
    1. Simple categorical: model, Brand (rules + mode)
    2. Conditional categorical: fuelType, transmission (mode by group)
    3. Binary flags: has_damage, has_reported_damage (mode)
    4. Correlated numerical: IterativeImputer (MICE)
    5. Optional flags to indicate imputed values
    6. Plausibility clipping
    Args:
        X_train: Training feature set.
        X_val: Validation feature set.
        X_test: Test feature set.
    '''
    
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test.copy()
    
    print("="*80)
    print("HYBRID IMPUTATION PIPELINE")
    print("="*80)
    
    
# =========================================================================
    # STEP 1: MODEL (brand mode if Brand known, else global mode)
    # =========================================================================
    '''
    print("\n[1/6] MODEL - brand-aware mode + global fallback")

    
    global_mode_model = X_tr["model"].mode()[0] if len(X_tr["model"].mode()) > 0 else "unknown"

   
    n_missing_train = X_tr["model"].isna().sum()

    
    brand_to_model_mode = (
        X_tr.dropna(subset=["Brand", "model"])
            .groupby("Brand")["model"]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
            .to_dict()
    )

    def fill_model(df):
        """Fill missing model values.
        Args:
            df: DataFrame to process.
        """
        miss_model = df["model"].isna()

        
        has_brand = df["Brand"].notna()
        idx_brand = df.index[miss_model & has_brand]
        df.loc[idx_brand, "model"] = df.loc[idx_brand, "Brand"].map(brand_to_model_mode)

      
        df["model"] = df["model"].fillna(global_mode_model)

    
    fill_model(X_tr)
    fill_model(X_v)
    fill_model(X_te)

    print(f"  Global mode: '{global_mode_model}'")
    print(f"  Imputed - Train: {n_missing_train}, Val: {X_val['model'].isna().sum()}, "
        f"Test: {X_test['model'].isna().sum()}")
    # =========================================================================
    # STEP 2: BRAND (inferred from model, then mode)
    # =========================================================================
    print("\n[2/6] BRAND - inferred from model + learned mapping")
    
    # Create model->Brand dictionary from known data
    model_to_brand_map = (
        X_tr.dropna(subset=['Brand', 'model'])
        .groupby('model')['Brand']
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
        .to_dict()
    )
    
    # Fallback: hardcoded lists for cases not in data
    toyota = ["yaris", "aygo", "corolla", "chr", "avensis", "prius", "rav4", "hilux", 
              "verso", "supra", "landcruiser", "camry", "proaceverso", "urbancruiser", 
              "auris", "gt86"]
    ford = ["focus", "fiesta", "mondeo", "kuga", "galaxy", "smax", "bmax", "ecosport", 
            "puma", "tourneocustom", "tourneoconnect", "grandtourneoconnect", "cmax", 
            "grandcmax", "edge", "mustang", "fusion", "streetka", "ranger", "escort", 
            "ka", "ka+"]
    opel = ["corsa", "mokkax", "astra", "insignia", "mokka", "zafira", "viva", "meriva", 
            "adam", "combolife", "crosslandx", "grandlandx", "gtc", "antara", "vivaro", 
            "vectra", "agila", "tigra", "cascada", "ampera"]
    vw = ["golf", "golfsv", "polo", "passat", "tiguan", "tiguanallspace", "touran", 
          "touareg", "troc", "tcross", "arteon", "sharan", "jetta", "cc", "caravelle", 
          "california", "caddy", "caddymaxi", "beetle", "scirocco", "up", "amarok", "eos", "fox"]
    audi = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "q2", "q3", "q5", "q7", 
            "q8", "s3", "s4", "s5", "s8", "rs3", "rs4", "rs5", "rs6", "sq5", "sq7", "tt", "r8"]
    mercedes = ["aclass", "bclass", "cclass", "eclass", "sclass", "claclass", "clsclass", 
                "glaclass", "glbclass", "glcclass", "gleclass", "glsclass", "glclass", 
                "gclass", "vclass", "xclass", "slclass", "slkclass", "mclass", "slc", 
                "clk", "clclass", "clcclass", "mercedes200", "mercedes220", "mercedes230"]
    skoda = ["fabia", "octavia", "superb", "karoq", "kodiaq", "kamiq", "yeti", 
             "yetioutdoor", "scala", "rapid", "citigo", "roomster"]
    hyundai = ["i10", "i20", "i30", "i40", "i800", "ioniq", "kona", "tucson", "santafe", 
               "getz", "ix20", "ix35", "veloster", "accent", "terracan"]
    bmw_models = ["series1", "series2", "series3", "series4", "series5", "series6", 
                  "series7", "series8", "x1", "x2", "x3", "x4", "x5", "x6", "x7", 
                  "z3", "z4", "m2", "m3", "m4", "m5", "m6", "iq"]
    seat_models = ["leon", "ateca", "toledo", "arona", "ibiza", "alhambra"]
    
    def infer_brand_smart(model_val):
        """Infer Brand from model using learned mapping and hardcoded lists.
        Args:
            model_val: The model value to infer the brand for.
        """
        if pd.isna(model_val):
            return None
        
        # First try learned mapping
        if model_val in model_to_brand_map:
            return model_to_brand_map[model_val]
        
        # Fallback to hardcoded lists
        m = str(model_val).lower()
        if m in toyota: return "toyota"
        if m in ford: return "ford"
        if m in opel: return "opel"
        if m in vw: return "vw"
        if m in audi: return "audi"
        if m in bmw_models: return "bmw"
        if m in mercedes: return "mercedes"
        if m in skoda: return "skoda"
        if m in hyundai: return "hyundai"
        if m in seat_models: return "seat"
        if m == "kadjar": return "renault"
        if m == "shuttle": return "honda"
        return None
    
    # Apply inference
    n_missing_brand = X_tr["Brand"].isna().sum()
    for df in [X_tr, X_v, X_te]:
        mask_nan = df["Brand"].isna()
        df.loc[mask_nan, "Brand"] = df.loc[mask_nan, "model"].apply(infer_brand_smart)
    
    # Global mode for remaining
    global_mode_brand = X_tr["Brand"].mode()[0] if len(X_tr["Brand"].mode()) > 0 else "ford"
    X_tr["Brand"].fillna(global_mode_brand, inplace=True)
    X_v["Brand"].fillna(global_mode_brand, inplace=True)
    X_te["Brand"].fillna(global_mode_brand, inplace=True)
    
    print(f"  Learned mapping: {len(model_to_brand_map)} models")
    print(f"  Imputed - Train: {n_missing_brand}, Val: {X_val['Brand'].isna().sum()}, "
          f"Test: {X_test['Brand'].isna().sum()}")
    
    # =========================================================================
    # STEP 3: CONDITIONAL CATEGORICAL (fuelType, transmission)
    # =========================================================================
    print("\n[3/6] FUELTYPE & TRANSMISSION - mode by group")
    
    # fuelType by Brand
    mode_fueltype_brand = (
        X_tr.groupby("Brand")["fuelType"]
        .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    )
    global_mode_fueltype = X_tr["fuelType"].mode()[0] if len(X_tr["fuelType"].mode()) > 0 else "Petrol"
    
    def fill_fueltype(row):
        """Fill missing fuelType based on Brand mode, else global mode.
        Args:
            row: DataFrame row to process.
        """
        if pd.notna(row["fuelType"]):
            return row["fuelType"]
        val = mode_fueltype_brand.get(row["Brand"], global_mode_fueltype)
        return val if pd.notna(val) else global_mode_fueltype
    
    n_missing_fuel = X_tr["fuelType"].isna().sum()
    X_tr["fuelType"] = X_tr.apply(fill_fueltype, axis=1)
    X_v["fuelType"] = X_v.apply(fill_fueltype, axis=1)
    X_te["fuelType"] = X_te.apply(fill_fueltype, axis=1)
    
    # transmission by Brand + fuelType
    mode_transmission_brandfuel = (
        X_tr.groupby(["Brand", "fuelType"])["transmission"]
        .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    )
    mode_transmission_brand = (
        X_tr.groupby("Brand")["transmission"]
        .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    )
    global_mode_transmission = X_tr["transmission"].mode()[0] if len(X_tr["transmission"].mode()) > 0 else "Manual"
    
    def fill_transmission(row):
        """Fill missing transmission based on (Brand, fuelType) mode,
        then Brand mode, else global mode.
        Args:
            row: DataFrame row to process.
        """
        if pd.notna(row["transmission"]):
            return row["transmission"]
        val = mode_transmission_brandfuel.get((row["Brand"], row["fuelType"]))
        if pd.isna(val):
            val = mode_transmission_brand.get(row["Brand"], global_mode_transmission)
        return val if pd.notna(val) else global_mode_transmission
    
    n_missing_trans = X_tr["transmission"].isna().sum()
    X_tr["transmission"] = X_tr.apply(fill_transmission, axis=1)
    X_v["transmission"] = X_v.apply(fill_transmission, axis=1)
    X_te["transmission"] = X_te.apply(fill_transmission, axis=1)
    
    print(f"  fuelType imputed - Train: {n_missing_fuel}")
    print(f"  transmission imputed - Train: {n_missing_trans}")
    
    # =========================================================================
    # STEP 3.5: BINARY FLAGS (has_damage, has_reported_damage)
    # =========================================================================
    print("\n[3.5/6] BINARY FLAGS - has_reported_damage")
    
    for col in ['has_reported_damage']:
        if col in X_tr.columns:
            mode_val = X_tr[col].mode()[0] if len(X_tr[col].mode()) > 0 else 0
            n_missing_train = X_tr[col].isna().sum()
            n_missing_val = X_v[col].isna().sum()
            n_missing_test = X_te[col].isna().sum()
            
            X_tr[col].fillna(mode_val, inplace=True)
            X_v[col].fillna(mode_val, inplace=True)
            X_te[col].fillna(mode_val, inplace=True)
            
            if n_missing_train > 0 or n_missing_val > 0 or n_missing_test > 0:
                print(f"  {col} - mode: {mode_val}, imputed Train: {n_missing_train}, "
                      f"Val: {n_missing_val}, Test: {n_missing_test}")
    
    # =========================================================================
    # STEP 4: ENSURE KNOWN CATEGORICAL VALUES (before MICE)
    # =========================================================================
    print("\n[4/6] SYNCHRONIZATION - force known categorical values")
    
    cat_cols_to_sync = ['Brand', 'model', 'fuelType', 'transmission']
    
    for col in cat_cols_to_sync:
        if col in X_tr.columns:
            # Get known values (excluding NaN)
            known_values = set(X_tr[col].dropna().unique())
            mode_val = X_tr[col].mode()[0]
            
            # Val: replace unknown with mode (only non-null values)
            mask_unknown_val = X_v[col].notna() & (~X_v[col].isin(known_values))
            n_unknown_val = mask_unknown_val.sum()
            if n_unknown_val > 0:
                X_v.loc[mask_unknown_val, col] = mode_val
                print(f"  {col} - Val: {n_unknown_val} unknown values -> '{mode_val}'")
            
            # Test: same
            mask_unknown_test = X_te[col].notna() & (~X_te[col].isin(known_values))
            n_unknown_test = mask_unknown_test.sum()
            if n_unknown_test > 0:
                X_te.loc[mask_unknown_test, col] = mode_val
                print(f"  {col} - Test: {n_unknown_test} unknown values -> '{mode_val}'")
    '''
    # =========================================================================
    # STEP 5: CORRELATED NUMERICAL - IterativeImputer (MICE)
    # =========================================================================
    print("\n[5/6] NUMERICAL - IterativeImputer (MICE)")
    
    numeric_cols = ['year', 'engineSize', 'mileage', 'mpg', 'tax', 'previousOwners']
    
    # Check which have missing
    numeric_cols_with_missing = [col for col in numeric_cols 
                                  if X_tr[col].isna().sum() > 0]
    
    if numeric_cols_with_missing:
        print(f"  Columns to impute: {numeric_cols_with_missing}")
        
        # Prepare data for imputer
        # Convert categorical to numeric codes temporarily
        cat_cols = ['Brand', 'model', 'fuelType', 'transmission']
        
        # Create temporary copies
        X_tr_temp = X_tr.copy()
        X_v_temp = X_v.copy()
        X_te_temp = X_te.copy()
        
        # Temporary label encoding
        label_mappings = {}
        for col in cat_cols:
            if col in X_tr_temp.columns:
                # Create mapping from train (excluding NaN)
                unique_vals = X_tr_temp[col].dropna().unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                label_mappings[col] = mapping
                
                # Apply mapping (unknown values remain as NaN)
                X_tr_temp[col] = X_tr_temp[col].map(mapping)
                X_v_temp[col] = X_v_temp[col].map(mapping)
                X_te_temp[col] = X_te_temp[col].map(mapping)
        
        # Select features for imputer
        features_for_imputation = cat_cols + numeric_cols
        features_for_imputation = [f for f in features_for_imputation if f in X_tr_temp.columns]
        
        # Configure and train imputer
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42),
            max_iter=10,
            random_state=42,
            verbose=0
        )
        
        # Fit on train
        X_tr_imputed = imputer.fit_transform(X_tr_temp[features_for_imputation])
        X_v_imputed = imputer.transform(X_v_temp[features_for_imputation])
        X_te_imputed = imputer.transform(X_te_temp[features_for_imputation])
        
        # Replace only imputed numerical columns
        for i, col in enumerate(numeric_cols):
            if col in features_for_imputation:
                idx = features_for_imputation.index(col)
                X_tr[col] = X_tr_imputed[:, idx]
                X_v[col] = X_v_imputed[:, idx]
                X_te[col] = X_te_imputed[:, idx]
        
        print(f"  IterativeImputer applied successfully")
    else:
        print(f"  No numerical columns with missing values")
    
    # =========================================================================
    # STEP 6: VALIDATION AND CORRECTIONS
    # =========================================================================
    print("\n[6/6] VALIDATION - checking logical limits")
    
    # Sanity corrections
    if 'year' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['year'] = df['year'].clip(upper=2025)
    
    if 'engineSize' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['engineSize'] = df['engineSize'].clip(lower=0.5)
    
    if 'mileage' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['mileage'] = df['mileage'].clip(lower=0, upper=500000)
    
    if 'mpg' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['mpg'] = df['mpg'].clip(lower=10, upper=200)
    
    if 'tax' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['tax'] = df['tax'].clip(lower=0, upper=1000)
    
    if 'previousOwners' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['previousOwners'] = df['previousOwners'].clip(lower=0, upper=10).round()
    
    print(f"  Limits applied")
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "="*80)
    print("IMPUTATION COMPLETED")
    print("="*80)
    
    print("\nFinal missing values:")
    print(f"  Train: {X_tr.isna().sum().sum()}")
    print(f"  Val:   {X_v.isna().sum().sum()}")
    print(f"  Test:  {X_te.isna().sum().sum()}")
    
    if X_tr.isna().sum().sum() > 0:
        print("\nColumns with remaining NaNs in Train:")
        print(X_tr.isna().sum()[X_tr.isna().sum() > 0])
    
    if X_v.isna().sum().sum() > 0:
        print("\nColumns with remaining NaNs in Val:")
        print(X_v.isna().sum()[X_v.isna().sum() > 0])
    
    if X_te.isna().sum().sum() > 0:
        print("\nColumns with remaining NaNs in Test:")
        print(X_te.isna().sum()[X_te.isna().sum() > 0])
    
    return X_tr, X_v, X_te


def impute_knn_categorical(X_train, X_val, X_test):
    # =========================================================================
    # STEP 3.25: kNN CATEGORICAL (leakage-safe)
    # =========================================================================
    print("\n[3.25/6] kNN CATEGORICAL IMPUTATION (train-only)")

    

    # Categóricas a refinar com kNN (opcional)
    knn_cat_cols = ['fuelType', 'transmission', 'Brand', 'model']
    knn_num_cols = ['year', 'engineSize', 'mileage', 'mpg', 'tax']

    knn_cat_cols = [c for c in knn_cat_cols if c in X_train.columns]
    knn_num_cols = [c for c in knn_num_cols if c in X_train.columns]

    if knn_cat_cols and knn_num_cols:

        # Escalar numéricas (fit só no treino)
        scaler_knn = StandardScaler()
        X_tr_num_scaled = scaler_knn.fit_transform(X_train[knn_num_cols])
        X_v_num_scaled  = scaler_knn.transform(X_val[knn_num_cols])
        X_te_num_scaled = scaler_knn.transform(X_test[knn_num_cols])

        # Base kNN: apenas linhas completas do treino
        mask_complete = X_train[knn_cat_cols + knn_num_cols].notna().all(axis=1)

        X_knn_base = X_train.loc[mask_complete, knn_cat_cols + knn_num_cols].copy()
        X_knn_num_base = scaler_knn.transform(X_knn_base[knn_num_cols])

        # Codificar categorias como códigos (apenas treino)
        cat_maps = {}
        for col in knn_cat_cols:
            X_knn_base[col] = X_knn_base[col].astype('category')
            cat_maps[col] = X_knn_base[col].cat.categories
            X_knn_base[col] = X_knn_base[col].cat.codes

        X_knn_matrix = np.hstack([X_knn_num_base, X_knn_base[knn_cat_cols].values])

        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(X_knn_matrix)

        def knn_impute(df, df_num_scaled, name):
            n_imputed = 0

            for col in knn_cat_cols:
                for idx in df[df[col].isna()].index:

                    row_num = df_num_scaled[df.index.get_loc(idx)].reshape(1, -1)

                    dummy_cat = np.zeros((1, len(knn_cat_cols)))
                    row_vec = np.hstack([row_num, dummy_cat])

                    _, neighbors = knn.kneighbors(row_vec)
                    neigh_vals = X_knn_base.iloc[neighbors[0]][col]
                    neigh_vals = neigh_vals[neigh_vals >= 0]

                    if len(neigh_vals) > 0:
                        code = neigh_vals.mode()[0]
                        df.at[idx, col] = cat_maps[col][code]
                    else:
                        df.at[idx, col] = X_train[col].mode()[0]

                    n_imputed += 1

            print(f"  {name}: {n_imputed} values imputed via kNN")

        knn_impute(X_train, X_tr_num_scaled, "Train")
        knn_impute(X_val,  X_v_num_scaled,  "Val")
        knn_impute(X_test, X_te_num_scaled, "Test")

    else:
        print("  Skipped (missing required columns)")

    return X_train, X_val, X_test


# Chi2 test for feature importance in categorical variables.

# In[19]:


def TestIndependence(X,y,var,alpha=0.05): 
    """
    Perform Chi-squared test of independence between a categorical feature and the target.
    Args:
        X: Feature DataFrame.
        y: Target Series.
        var: Name of the categorical variable to test.
        alpha: Significance level for the test."""       
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:#if p<alpha we reject the null and there is a relationship so the var is important for prediction
        result="{0} is IMPORTANT for Prediction".format(var)#
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)#independent H0
    print(result)


# Spearman correlation map function.

# In[20]:


def cor_heatmap(cor):
    """
    Plot a heatmap of the correlation matrix.
    Args:
        cor: Correlation matrix (DataFrame).
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(data = cor, annot = True, cmap = plt.cm.Purples, fmt='.1')
    plt.show()


# Lasso importance grid

# In[22]:


def plot_importance(coef,name):
    """
    Plot feature importance from model coefficients.
    Args:
        coef: Series of model coefficients.
        name: Name of the model (for title).
    """
    imp_coef = coef.sort_values()
    plt.figure(figsize=(6,8))
    imp_coef.plot(kind = "barh", color='purple')
    plt.title("Feature importance using " + name + " Model")
    plt.show()


# Model evaluation functions

# In[ ]:


def compute_metrics(model, X, y, split):
    """
    Compute evaluation metrics for a regression model.
    Args:
        model: Fitted regression model.
        X: Features for prediction.
        y: True target values.
        split: Data split identifier (e.g., 'train', 'val', 'test').
    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X)
    return {
        "split": split,
        "MAE": mean_absolute_error(y, y_pred),
        "MedAE": median_absolute_error(y, y_pred),
        "RMSE": root_mean_squared_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }

def compute_metrics_log(model, X, y, split):
    """
    Compute evaluation metrics for a regression model with log-transformed target.
    Args:
        model: Fitted regression model.
        X: Features for prediction.
        y: True target values (original scale).
        split: Data split identifier (e.g., 'train', 'val', 'test').
    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log)
    return {
        "split": split,
        "MAE": mean_absolute_error(y, y_pred),
        "MedAE": median_absolute_error(y, y_pred),
        "RMSE": root_mean_squared_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }


# In[24]:


def run_model(X, y, scaler=None, model=None, fill_method=None):
    """
    Train a model with optional preprocessing.
    
    Parameters:
    - X: Features (will be copied to avoid modifying original)
    - y: Target
    - scaler: Scaler instance (e.g., StandardScaler()) or None for no scaling
    - model: Model instance or None for LogisticRegression default
    - fill_method: 'median', 'mean', or None for no filling
    
    Returns:
    - model: Fitted model
    - scaler: Fitted scaler (or None)
    - fill_values: Dictionary of fill values (or None)
    """
    # Copy to avoid modifying original data
    X_processed = X.copy()
    
    # Fill missing values - this function uses simple statistics from the training set but you can modify it to use more complex strategies
    fill_values = None
    if fill_method is not None:
        if fill_method == 'function':
            fill_values = impute_missing_values_hybrid(X_processed)
        elif fill_method == 'mean':
            fill_values = X_processed.mean()
        X_processed = X_processed.fillna(fill_values)
    
    # Scale features
    if scaler is not None:
        X_processed = scaler.fit_transform(X_processed)
    
    # Use provided model or create default
    if model is None:
        model = RandomForestRegressor()
    
    # Fit the model
    model.fit(X_processed, y)
    
    return model, scaler, fill_values


# In[25]:


def evaluate_model_rf_mae(X, y, model=None, scaler=None, fill_method=None):
    """
    Avalia um modelo RandomForestRegressor usando o Mean Absolute Error (MAE).
    
    Esta versão ASSUME que X e y JÁ ESTÃO PRÉ-PROCESSADOS (X_val_final, y_val_final).
    Os parâmetros 'scaler' e 'fill_values' são mantidos na assinatura para 
    compatibilidade, mas são ignorados no processamento interno.

    Parameters:
    - X: Features (Dados de validação já processados, e.g., X_val_final)
    - y: Target (e.g., y_val)
    - model: Modelo ajustado (Fitted RandomForestRegressor)
    - scaler: Ignorado.
    - fill_values: Ignorado.
    
    Returns:
    - mae: Mean Absolute Error (Erro Absoluto Médio)
    """
    # 1. Copia dos dados (mantido por segurança, embora não haja modificação)
    X_processed = X.copy()
    
    # 2. Imputação e Scaling SÃO IGNORADOS, pois os dados já estão processados
    # if fill_values is not None: ...
    # if scaler is not None: ...
    
    # 3. Fazer as previsões
    y_pred = model.predict(X_processed)
    
    # 4. Calcular o MAE
    # Nota: Assumindo que 'y' é o target na escala que se pretende (ex: log)
    mae = mean_absolute_error(y, y_pred)
    
    return mae

