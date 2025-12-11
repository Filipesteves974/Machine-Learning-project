#!/usr/bin/env python
# coding: utf-8

# In[23]:


#general imports that we will need will almost always use - it is a good practice to import all libraries at the beginning of the notebook or script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# data partition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#filter methods
# spearman 
# chi-square
import scipy.stats as stats
from scipy.stats import chi2_contingency

#wrapper methods
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import lasso_path, SGDRegressor


# embedded methods
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet
from sklearn.compose import TransformedTargetRegressor


#set random seed for reproducibility
RSEED = 42
np.random.seed(RSEED)


# Transforms all data to fit the same criteria making it easier to work on.

# In[24]:


def normalize_data(x):
    x = str(x)
    x = x.lower()
    x = x.replace("_", "")
    x = x.replace("-", "")
    x = x.replace(" ", "")
    if x == "nan":
        return np.nan
    return x


# Function that automatically looks for the closest match on the valid list, therefore correcting the visible typos.

# In[25]:


def correct_missing_letters(value, valid_list, max_missing=2):
    """
    corrects values with missing letters based on valid_list
    """
    best_match = value
    smallest_diff = 999
    if pd.isna(value):  # <- ignores NaN
        return np.nan
    for ref in valid_list:
        # absolute length difference
        len_diff = abs(len(ref) - len(value))
        if len_diff == 0 or len_diff > max_missing:
            continue  # ignora se igual ou diferença > limite

        # verificar se o valor é subsequência do nome correto (mantendo ordem)
        it = iter(ref)
        is_subseq = all(ch in it for ch in value)

        if is_subseq and len_diff < smallest_diff:
            smallest_diff = len_diff
            best_match = ref

    return best_match

valid_list = []


# Individually treats the outliers detected for each feature. Methods applied were Winsorization and removing by NaN placement

# In[26]:


def remove_outliers_smart_v3(X_train, X_val, X_test, y_train, y_val):
    """
    Trata outliers:
    - Train/Val/Test: Substitui valores impossíveis por NaN (não remove linhas)
    - Train/Val/Test: Faz capping em valores extremos
    """
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_tst = X_test.copy()
    y_tr = y_train.copy()
    y_v = y_val.copy()
    
    
    # ========== YEAR < 1990 ==========
    if 'year' in X_tr.columns:
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
            print(f" {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    

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
        q_high = X_tr['mpg'].quantile(0.98)
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
    

    # ========== ENGINE SIZE > 6.0L ==========
    if 'engineSize' in X_tr.columns:
        print(f"\n[ENGINE SIZE]")
        
        mask_tr = X_tr['engineSize'] > 6.0
        mask_v = X_v['engineSize'] > 6.0
        mask_tst = X_tst['engineSize'] > 6.0
        
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
    
    # Carros novos com quilometragem absurda
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
            print(f" Carros novos c/ alta quilometragem: {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    
    # Motores grandes com MPG alto (fisicamente impossível)
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
            print(f" Motor grande c/ MPG alto: {removed_tr} train, {removed_v} val, {removed_tst} test (→ NaN)")
    

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("="*60)
    print(f"Mantidos: {len(X_tr)} train (100.0%), "
          f"{len(X_v)} val (100.0%), "
          f"{len(X_tst)} test (100.0%)")
    print(f"Nenhuma linha removida - valores impossíveis substituídos por NaN")
    print("="*60 + "\n")
    
    return X_tr, X_v, X_tst, y_tr, y_v


# Hybrid solution for filling in the missing values based on their statistical peers.

# In[27]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def impute_missing_values_hybrid(X_train, X_val, X_test, create_flags=True):
    """
    Hybrid intelligent imputation:
    1. Simple categorical: model, Brand (rules + mode)
    2. Conditional categorical: fuelType, transmission (mode by group)
    3. Binary flags: has_damage, has_reported_damage (mode)
    4. Correlated numerical: IterativeImputer (MICE)
    5. Optional flags to indicate imputed values
    """
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test.copy()
    
    print("="*80)
    print("HYBRID IMPUTATION PIPELINE")
    print("="*80)
    
    # =========================================================================
    # STEP 0: Create missing flags (BEFORE imputation)
    # =========================================================================
    if create_flags:
        print("\n[CREATING MISSING FLAGS]")
        cols_to_flag = ['model', 'Brand', 'year', 'engineSize', 'mileage', 
                        'fuelType', 'transmission', 'mpg', 'tax', 'previousOwners',
                        'has_damage', 'has_reported_damage']
        
        if 'paintQuality%' in X_tr.columns:
            cols_to_flag.append('paintQuality%')
        
        for col in cols_to_flag:
            if col in X_tr.columns:
                X_tr[f'{col}_was_missing'] = X_tr[col].isna().astype(int)
                X_v[f'{col}_was_missing'] = X_v[col].isna().astype(int)
                X_te[f'{col}_was_missing'] = X_te[col].isna().astype(int)
        
        print(f"  Created {len([c for c in cols_to_flag if c in X_tr.columns])} missing flags")
    
    # =========================================================================
    # STEP 1: MODEL (global mode)
    # =========================================================================
    print("\n[1/6] MODEL - global mode")
    
    global_mode_model = X_tr["model"].mode()[0] if len(X_tr["model"].mode()) > 0 else "unknown"
    
    n_missing_train = X_tr["model"].isna().sum()
    X_tr["model"].fillna(global_mode_model, inplace=True)
    X_v["model"].fillna(global_mode_model, inplace=True)
    X_te["model"].fillna(global_mode_model, inplace=True)
    
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
    print("\n[3.5/6] BINARY FLAGS - has_damage, has_reported_damage")
    
    for col in ['has_damage', 'has_reported_damage']:
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
    
    # =========================================================================
    # STEP 5: CORRELATED NUMERICAL - IterativeImputer (MICE)
    # =========================================================================
    print("\n[5/6] NUMERICAL - IterativeImputer (MICE)")
    
    numeric_cols = ['year', 'engineSize', 'mileage', 'mpg', 'tax', 'previousOwners']
    if 'paintQuality%' in X_tr.columns:
        numeric_cols.append('paintQuality%')
    
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
            df['year'] = df['year'].clip(lower=1990, upper=2025)
    
    if 'engineSize' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['engineSize'] = df['engineSize'].clip(lower=0.5, upper=10.0)
    
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
    
    if 'paintQuality%' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['paintQuality%'] = df['paintQuality%'].clip(lower=0, upper=100)
    
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
    
    if create_flags:
        flag_cols = [col for col in X_tr.columns if col.endswith('_was_missing')]
        print(f"\nFlags created: {len(flag_cols)} columns")
        if flag_cols:
            print(f"  Example: {flag_cols[:3]}")
    
    return X_tr, X_v, X_te



# Chi2 test for feature importance in categorical variables.

# In[28]:


def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:#if p<alpha we reject the null and there is a relationship so the var is important for prediction
        result="{0} is IMPORTANT for Prediction".format(var)#
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)#independent H0
    print(result)


# Spearman correlation map function.

# In[29]:


def cor_heatmap(cor):
    plt.figure(figsize=(12,10))
    sns.heatmap(data = cor, annot = True, cmap = plt.cm.Purples, fmt='.1')
    plt.show()


# RFE

# In[30]:


def optimal_rfe(X, y, scoring='r2', cv=5, verbose=True):

    model = LinearRegression()
    n_features = X.shape[1]
    scores = []

    if verbose:
        print("Trying features")

    for n in range(1, n_features + 1):
        rfe = RFE(model, n_features_to_select=n)
        X_rfe = rfe.fit_transform(X, y)
        score = np.mean(cross_val_score(model, X_rfe, y, scoring=scoring, cv=cv))
        scores.append(score)

        if verbose:
            print(f"{n:2d} features -> {scoring}: {score:.4f}")

    best_n = np.argmax(scores) + 1
    best_score = scores[best_n - 1]

    best_rfe = RFE(model, n_features_to_select=best_n)
    best_rfe.fit(X, y)

    feature_ranking = (
        {feature: rank for feature, rank in zip(X.columns, best_rfe.ranking_)}
        if hasattr(X, "columns")
        else None
    )

    if verbose:
        print("\nBest number of features:", best_n)
        print("Best average score:", round(best_score, 4))
        if feature_ranking:
            print("Selected features:", X.columns[best_rfe.support_].tolist())

    return best_rfe, best_n, best_score, feature_ranking


# Lasso importance grid

# In[31]:


def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(6,8))
    imp_coef.plot(kind = "barh", color='purple')
    plt.title("Feature importance using " + name + " Model")
    plt.show()


# Model evaluation functions

# In[32]:


def evaluate_model(model, X_train, X_val, y_train, y_val):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    print("Validation set:")
    print("R²:", r2_score(y_val, y_pred_val))
    print("MAE:", mean_absolute_error(y_val, y_pred_val))
    print("RMSE:", root_mean_squared_error(y_val, y_pred_val))
    print("\nTraining set:")
    print("R²:", r2_score(y_train, y_pred_train))
    print("MAE:", mean_absolute_error(y_train, y_pred_train))
    print("RMSE:", root_mean_squared_error(y_train, y_pred_train))
    print("-" * 60)

def evaluate_model_original_scale(model, X_train, X_val, y_train, y_val):
    # predições em log
    y_pred_train_log = model.predict(X_train)
    y_pred_val_log = model.predict(X_val)

    # voltar para a escala original
    y_pred_train = np.exp(y_pred_train_log)
    y_pred_val = np.exp(y_pred_val_log)

    print("Validation set:")
    print("R²:", r2_score(y_val, y_pred_val))
    print("MAE:", mean_absolute_error(y_val, y_pred_val))
    print("RMSE:", root_mean_squared_error(y_val, y_pred_val))
    print("\nTraining set:")
    print("R²:", r2_score(y_train, y_pred_train))
    print("MAE:", mean_absolute_error(y_train, y_pred_train))
    print("RMSE:", root_mean_squared_error(y_train, y_pred_train))
    print("-" * 60)


# In[ ]:


# !jupyter nbconvert --to python functions.ipynb

