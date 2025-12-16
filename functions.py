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
from difflib import SequenceMatcher, get_close_matches
from collections import Counter
# ignore warnings
import warnings
warnings.filterwarnings('ignore')





#set random seed for reproducibility
RSEED = 42
np.random.seed(RSEED)



# In[15]:


def similarity_ratio(a, b):
    """
    Computes string similarity between two values using difflib.
    Returns a value between 0 and 1.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_categorical_column(
    df,
    column_name,
    case_threshold=0.95,
    truncation_threshold=0.85,
    min_length=2,
    aggressive_short=True,
    invalid_values=None,
    show_changes=True
):
    """
    Cleans a categorical column by fixing:
    - Case and whitespace inconsistencies
    - Truncated values
    - Very short or corrupted strings
    - Known invalid values

    The goal is to reduce category noise without manually hardcoding everything.
    """

    # Keep a copy of the original column to show what changed later
    original_col = df[column_name].copy()

    
    # Basic normalization: trim spaces and convert empty / fake NaNs
    df[column_name] = df[column_name].astype(str).str.strip()
    df[column_name] = df[column_name].replace('', np.nan)
    df[column_name] = df[column_name].replace('nan', np.nan)

    
    # Convert known invalid values (e.g. "unknown", "Other") to NaN
    if invalid_values:
        for inv in invalid_values:
            mask = df[column_name].str.lower() == inv.lower()
            df[column_name] = df[column_name].mask(mask, np.nan)

    values = df[column_name].dropna()
    unique_vals = values.unique()
    value_counts = values.value_counts()

    print(f"\nCleaning column: {column_name}")
    print(f"Initial unique values: {len(unique_vals)}")


    # PHASE 1 — Fix case differences and extra spaces
    case_mapping = {}
    lower_groups = {}

    # Group values that are identical once lowercased and space-normalized
    for val in unique_vals:
        normalized = ' '.join(val.lower().split())
        lower_groups.setdefault(normalized, []).append(val)

    # Pick a canonical version for each group
    for normalized, variants in lower_groups.items():
        if len(variants) > 1:
            canonical = max(
                variants,
                key=lambda x: (
                    len(x.split()) == len(x.strip().split()),  # no extra spaces
                    x[0].isupper() and not x.isupper(),         # Title Case
                    value_counts.get(x, 0),                     # most frequent
                    len(x)                                      # longest
                )
            )
            for v in variants:
                if v != canonical:
                    case_mapping[v] = canonical

    df[column_name] = df[column_name].replace(case_mapping)
    print(f"Phase 1 corrections (case/spacing): {len(case_mapping)}")


    # PHASE 2 — Fix obvious truncations (e.g. 'Volks' → 'Volkswagen')
    values = df[column_name].dropna().astype(str)
    unique_vals = values.unique()
    value_counts = values.value_counts()

    truncation_mapping = {}
    processed = set()

    # Process shorter strings first
    sorted_vals = sorted(unique_vals, key=lambda x: (len(x), x))

    for short_val in sorted_vals:
        if short_val in processed or len(short_val) < min_length:
            continue

        candidates = [
            v for v in unique_vals
            if len(v) > len(short_val) and v not in processed
        ]

        matches = get_close_matches(
            short_val,
            candidates,
            n=3,
            cutoff=truncation_threshold
        )

        if matches:
            best_match = matches[0]
            sim = similarity_ratio(short_val, best_match)

            min_prefix = min(len(short_val), max(3, int(len(best_match) * 0.6)))
            is_prefix = best_match.lower().startswith(
                short_val[:min_prefix].lower()
            )

            # Accept if similarity is high and structure makes sense
            if (sim >= truncation_threshold and is_prefix) or sim >= 0.95:
                if (
                    value_counts.get(best_match, 0)
                    >= value_counts.get(short_val, 0)
                ):
                    truncation_mapping[short_val] = best_match
                    processed.add(short_val)

    df[column_name] = df[column_name].replace(truncation_mapping)
    print(f"Phase 2 corrections (truncations): {len(truncation_mapping)}")


    # PHASE 3 — Aggressive fixes for very short / corrupted values
    if aggressive_short:
        values = df[column_name].dropna().astype(str)
        unique_vals = values.unique()
        value_counts = values.value_counts()

        short_mapping = {}

        # Hand-written rules for known common issues
        specific_rules = {
            'W': 'VW', 'V': 'VW',
            'MW': 'BMW', 'BM': 'BMW',
            'ercedes': 'Mercedes',
            'oyota': 'Toyota',
            'koda': 'Skoda',
            'yundai': 'Hyundai'
        }

        if column_name.lower() in ['transmission', 'fueltype', 'fuel']:
            specific_rules.update({
                'anual': 'Manual',
                'utomatic': 'Automatic',
                'emi-Auto': 'Semi-Auto',
                'etrol': 'Petrol',
                'iesel': 'Diesel',
                'ybrid': 'Hybrid',
                'ther': 'Other'
            })

        for short, full in specific_rules.items():
            if short in unique_vals and full in unique_vals:
                short_mapping[short] = full

        # Generic logic for very short strings (<= 3 chars)
        very_short = [v for v in unique_vals if len(v) <= 3]
        normal_vals = [v for v in unique_vals if len(v) > 3]

        for short in very_short:
            candidates = [
                v for v in normal_vals
                if v.lower().startswith(short.lower())
                or short.lower() in v.lower()
            ]
            if candidates:
                best = max(candidates, key=lambda x: value_counts.get(x, 0))
                if similarity_ratio(short, best) >= 0.5:
                    short_mapping[short] = best

        df[column_name] = df[column_name].replace(short_mapping)
        print(f"Phase 3 corrections (short values): {len(short_mapping)}")

    
    # Summary
    final_unique = df[column_name].nunique()
    total_corrections = (
        len(case_mapping)
        + len(truncation_mapping)
        + (len(short_mapping) if aggressive_short else 0)
    )

    print(f"Final unique values: {final_unique}")
    print(f"Total corrections applied: {total_corrections}")

    if show_changes and total_corrections > 0:
        print("\nExamples of changes:")
        mappings = {**case_mapping, **truncation_mapping}
        if aggressive_short:
            mappings.update(short_mapping)

        for old, new in list(mappings.items())[:10]:
            count = (original_col == old).sum()
            print(f"{old} → {new} ({count} rows)")

    return df



def find_potential_duplicates(df, column_name, threshold=0.85):
    """
    Finds pairs of category values that are very similar and
    may represent the same category.
    """
    values = df[column_name].dropna().unique()
    potential_dupes = []

    for i, v1 in enumerate(values):
        matches = get_close_matches(
            v1,
            values[i + 1:],
            n=5,
            cutoff=threshold
        )
        for v2 in matches:
            sim = similarity_ratio(v1, v2)
            c1 = (df[column_name] == v1).sum()
            c2 = (df[column_name] == v2).sum()
            potential_dupes.append((v1, v2, sim, c1, c2))

    return potential_dupes

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
This function performs a hybrid numerical imputation strategy focused on preserving data integrity 
and preventing information leakage. It identifies numerical variables with missing values 
(such as year, engine size, mileage, fuel consumption, taxes, and number of previous owners) 
and imputes them using MICE (Multiple Imputation by Chained Equations) via IterativeImputer. 
To improve the quality of the imputation, the model leverages correlations not only between 
numerical variables but also with categorical features (make, model, fuel type, and transmission), 
which are temporarily coded with labels using only the training data. The imputer is tuned 
exclusively on the training set and then applied consistently to the validation and test sets. 
After imputation, the function enforces logical and domain-specific bounds 
(e.g., non-negative mileage, realistic engine sizes, valid year ranges) to ensure plausibility. 
The result is a set of training, validation, and test datasets with numerically 
consistent and statistically informed imputations, ready for downstream modelling.

"""
# In[ ]:

def impute_numeric_features(
    X_train,
    X_val,
    X_test
):
    """
    Hybrid numerical imputation using MICE (IterativeImputer).

    What this function does:
    ------------------------
    - Focuses ONLY on numerical variables that have missing values
    - Uses correlated information from:
        • numerical variables
        • temporarily encoded categorical variables
    - Fits the imputer ONLY on the training set
    - Applies the same transformation to validation and test sets
    - Enforces logical bounds after imputation (sanity checks)

    Inputs:
    -------
    X_train : pandas.DataFrame
        Training feature set
    X_val : pandas.DataFrame
        Validation feature set
    X_test : pandas.DataFrame
        Test feature set

    Returns:
    --------
    X_train_imputed, X_val_imputed, X_test_imputed
    """

    # Work on copies to avoid modifying original datasets
    X_tr = X_train.copy()
    X_v  = X_val.copy()
    X_te = X_test.copy()

    print("=" * 80)
    print("NUMERICAL IMPUTATION PIPELINE (MICE)")
    print("=" * 80)


    # STEP 1 — Identify numerical columns with missing values
    print("\n[1/3] Detect numerical columns with missing values")

    numeric_cols = [
        'year',
        'engineSize',
        'mileage',
        'mpg',
        'tax',
        'previousOwners'
    ]

    numeric_cols_with_missing = [
        col for col in numeric_cols
        if col in X_tr.columns and X_tr[col].isna().sum() > 0
    ]

    if not numeric_cols_with_missing:
        print("  No numerical columns require imputation")
        return X_tr, X_v, X_te

    print(f"  Columns to impute: {numeric_cols_with_missing}")


    # STEP 2 — Prepare data for IterativeImputer (MICE)
    print("\n[2/3] Preparing data for MICE")

    # Categorical columns are used ONLY as auxiliary predictors
    # They are temporarily encoded as numeric labels
    cat_cols = ['Brand', 'model', 'fuelType', 'transmission']

    # Temporary copies for encoding
    X_tr_temp = X_tr.copy()
    X_v_temp  = X_v.copy()
    X_te_temp = X_te.copy()

    label_mappings = {}

    for col in cat_cols:
        if col in X_tr_temp.columns:
            # Build mapping ONLY from training data (no leakage)
            unique_vals = X_tr_temp[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            label_mappings[col] = mapping

            # Apply mapping; unseen values become NaN
            X_tr_temp[col] = X_tr_temp[col].map(mapping)
            X_v_temp[col]  = X_v_temp[col].map(mapping)
            X_te_temp[col] = X_te_temp[col].map(mapping)

    # Final feature set used by the imputer
    features_for_imputation = [
        f for f in (cat_cols + numeric_cols)
        if f in X_tr_temp.columns
    ]

    # Configure IterativeImputer with RandomForest as estimator
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=42
        ),
        max_iter=10,
        random_state=42,
        verbose=0
    )

    # Fit ONLY on training data
    X_tr_imp = imputer.fit_transform(X_tr_temp[features_for_imputation])
    X_v_imp  = imputer.transform(X_v_temp[features_for_imputation])
    X_te_imp = imputer.transform(X_te_temp[features_for_imputation])

    # Replace ONLY numerical columns in the original datasets
    for col in numeric_cols:
        if col in features_for_imputation:
            idx = features_for_imputation.index(col)
            X_tr[col] = X_tr_imp[:, idx]
            X_v[col]  = X_v_imp[:, idx]
            X_te[col] = X_te_imp[:, idx]

    print("  IterativeImputer applied successfully")


    # Post-imputation sanity checks (logical bounds)
    print("\n[3/3] Applying logical bounds")

    if 'year' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['year'] = df['year'].clip(upper=2025)

    if 'engineSize' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['engineSize'] = df['engineSize'].clip(lower=0.5)

    if 'mileage' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['mileage'] = df['mileage'].clip(lower=0, upper=500_000)

    if 'mpg' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['mpg'] = df['mpg'].clip(lower=10, upper=200)

    if 'tax' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['tax'] = df['tax'].clip(lower=0, upper=1000)

    if 'previousOwners' in X_tr.columns:
        for df in [X_tr, X_v, X_te]:
            df['previousOwners'] = (
                df['previousOwners']
                .clip(lower=0, upper=10)
                .round()
            )

    # FINAL REPORT
    print("\n" + "=" * 80)
    print("IMPUTATION COMPLETED")
    print("=" * 80)

    print("\nRemaining missing values:")
    print(f"  Train: {X_tr.isna().sum().sum()}")
    print(f"  Val:   {X_v.isna().sum().sum()}")
    print(f"  Test:  {X_te.isna().sum().sum()}")

    return X_tr, X_v, X_te




def impute_knn_categorical(X_train, X_val, X_test):
    """
    kNN-based categorical imputation (leakage-safe).

    What this does:
    - Uses ONLY the training set to build a kNN “reference space”.
    - Finds nearest neighbors using scaled numerical features + encoded categorical features.
    - For each missing categorical value in Train/Val/Test, imputes it with the most common
      category among the nearest neighbors (mode).
    - Falls back to the global training mode if neighbors are not usable.

    Leakage-safe:
    - The scaler is fit only on X_train.
    - The kNN model is fit only on complete rows from X_train.
    - Val/Test are only transformed and queried; they never influence training mappings.
    """

    # =========================================================================
    # STEP 3.25: kNN CATEGORICAL IMPUTATION (train-only)
    # =========================================================================
    print("\n[3.25/6] kNN CATEGORICAL IMPUTATION (train-only)")

    # Categorical columns we want to impute/refine using kNN
    # (only if they exist in the dataset)
    knn_cat_cols = ['fuelType', 'transmission', 'Brand', 'model']

    # Numerical columns used to define similarity between rows
    # (kNN distance is computed mainly based on these; they are scaled)
    knn_num_cols = ['year', 'engineSize', 'mileage', 'mpg', 'tax']

    # Keep only columns that actually exist (robust to different schemas)
    knn_cat_cols = [c for c in knn_cat_cols if c in X_train.columns]
    knn_num_cols = [c for c in knn_num_cols if c in X_train.columns]

    # We need at least one categorical and one numerical column to run kNN
    if knn_cat_cols and knn_num_cols:

       
        # 1) Scale numeric features (fit ONLY on training to avoid leakage)
        scaler_knn = StandardScaler()

        # Fit on training numerical data; transform val/test with the same scaler
        X_tr_num_scaled = scaler_knn.fit_transform(X_train[knn_num_cols])
        X_v_num_scaled  = scaler_knn.transform(X_val[knn_num_cols])
        X_te_num_scaled = scaler_knn.transform(X_test[knn_num_cols])


        # 2) Build the kNN reference base from training rows that are complete
        #    (no missing in the features used to compute distances)
        mask_complete = X_train[knn_cat_cols + knn_num_cols].notna().all(axis=1)

        # Reference dataset: only complete rows (so neighbors have valid info)
        X_knn_base = X_train.loc[mask_complete, knn_cat_cols + knn_num_cols].copy()

        # Use the same scaler to get scaled numerical part for the base
        X_knn_num_base = scaler_knn.transform(X_knn_base[knn_num_cols])


        # 3) Encode categorical variables into numeric codes (train-only)
        #    This is temporary encoding just for kNN distance / neighbor lookup.
        cat_maps = {}  # stores category order so we can map codes back to labels

        for col in knn_cat_cols:
            # Convert to pandas categorical dtype
            X_knn_base[col] = X_knn_base[col].astype('category')

            # Save the categories to decode later (code -> original label)
            cat_maps[col] = X_knn_base[col].cat.categories

            # Replace strings with integer codes (0..K-1), missing would be -1
            # (but base rows are complete, so we expect no -1 here)
            X_knn_base[col] = X_knn_base[col].cat.codes


        # 4) Create the matrix used for neighbor search:
        #    [scaled numerics | categorical codes]
        X_knn_matrix = np.hstack([
            X_knn_num_base,
            X_knn_base[knn_cat_cols].values
        ])

        # Train the kNN model on the training base only
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(X_knn_matrix)


        # 5) Helper: impute missing categories in a given dataframe (train/val/test)
        def knn_impute(df, df_num_scaled, name):
            """
            For each missing value in the selected categorical columns:
            - Build a query vector composed of:
              • scaled numerical features for that row
              • dummy zeros for categorical part (since they're missing/unknown)
            - Get nearest neighbors from the train base
            - Impute using the neighbor mode (most frequent category code)
            - Decode that code back into the original category label
            - If neighbors are unusable, fallback to training global mode
            """
            n_imputed = 0

            # Loop through each categorical column we want to fill
            for col in knn_cat_cols:

                # Find row indices where this column is missing
                for idx in df[df[col].isna()].index:

                    # Grab the scaled numeric vector for that specific row
                    # df.index.get_loc(idx) gives the integer position of idx in df
                    row_num = df_num_scaled[df.index.get_loc(idx)].reshape(1, -1)

                    # Build a placeholder categorical vector of zeros.
                    # This keeps dimensions consistent with X_knn_matrix.
                    # (We are essentially using numerics to define proximity,
                    #  and category codes in the base to vote on the target column.)
                    dummy_cat = np.zeros((1, len(knn_cat_cols)))

                    # Full query vector: [scaled numerics | dummy categorical block]
                    row_vec = np.hstack([row_num, dummy_cat])

                    # Query neighbors from the train base
                    _, neighbors = knn.kneighbors(row_vec)

                    # Extract the neighbor values (codes) for the target column
                    neigh_vals = X_knn_base.iloc[neighbors[0]][col]

                    # Safety: remove invalid codes (pandas uses -1 for missing category codes)
                    neigh_vals = neigh_vals[neigh_vals >= 0]

                    if len(neigh_vals) > 0:
                        # Impute with the most common neighbor code (mode)
                        code = neigh_vals.mode()[0]

                        # Decode numeric code back to the original category label
                        df.at[idx, col] = cat_maps[col][code]
                    else:
                        # Fallback: if neighbors are unusable, use the global train mode
                        df.at[idx, col] = X_train[col].mode()[0]

                    n_imputed += 1

            print(f"  {name}: {n_imputed} values imputed via kNN")

        # Apply the same imputation logic to each split
        knn_impute(X_train, X_tr_num_scaled, "Train")
        knn_impute(X_val,  X_v_num_scaled,  "Val")
        knn_impute(X_test, X_te_num_scaled, "Test")

    else:
        # If required columns are not present, skip safely
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

