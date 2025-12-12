#!/usr/bin/env python
"""
End-to-end sklearn pipeline that mirrors the cleaning + feature engineering +
imputation steps used throughout the notebooks, followed by a RandomForestRegressor.

Usage:
    from full_pipeline import build_pipeline
    pipe = build_pipeline()
    pipe.fit(X_train, y_train_log)
    preds = pipe.predict(X_val)

This pipeline expects as input a pandas.DataFrame with the original raw columns
(Brand, model, year, mileage, etc).  All preprocessing logic is encapsulated
inside the pipeline so it can be safely used with cross-validation without data
leakage.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder

from functions import correct_missing_letters, impute_missing_values_hybrid, normalize_data


def _standardize_categorical(series: pd.Series, valid_list: List[str]) -> pd.Series:
    """Lowercases, strips extra characters and fixes typos."""
    cleaned = series.map(lambda val: normalize_data(val))
    return cleaned.map(lambda val: correct_missing_letters(val, valid_list))


class CategoricalNormalizer(BaseEstimator, TransformerMixin):
    """Normalises Brand/fuel/model/transmission values using the helper functions."""

    def __init__(self) -> None:
        self.valid_brands_ = [
            "bmw",
            "mercedes",
            "ford",
            "hyundai",
            "audi",
            "toyota",
            "opel",
            "skoda",
            "vw",
        ]
        self.valid_fuel_ = ["diesel", "petrol", "hybrid", "electric", "other"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalNormalizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Brand" in X.columns:
            X["Brand"] = _standardize_categorical(X["Brand"], self.valid_brands_)
        if "fuelType" in X.columns:
            X["fuelType"] = _standardize_categorical(X["fuelType"], self.valid_fuel_)
        if "model" in X.columns:
            to_correct = [
                "golf",
                "polo",
                "passat",
                "tiguan",
                "a3",
                "a4",
                "a5",
                "a6",
                "yaris",
                "fiesta",
                "focus",
                "insignia",
            ]
            X["model"] = _standardize_categorical(X["model"], to_correct)
        if "transmission" in X.columns:
            valid_trans = ["automatic", "manual", "semiauto", "unknown"]
            X["transmission"] = _standardize_categorical(X["transmission"], valid_trans)
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Replicates the remove_outliers_smart_v3 logic using stats learned on X_train."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierClipper":
        self.upper_mileage_ = X["mileage"].quantile(0.99) if "mileage" in X.columns else None
        if "mpg" in X.columns:
            self.mpg_low_, self.mpg_high_ = X["mpg"].quantile([0.005, 0.98])
        else:
            self.mpg_low_, self.mpg_high_ = None, None
        self.upper_tax_ = X["tax"].quantile(0.98) if "tax" in X.columns else None
        self.paint_min_, self.paint_max_ = 0, 100
        self.year_limits_ = (1990, 2025)
        self.engine_limits_ = (0.5, 6.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # YEAR
        if "year" in X.columns:
            lower, upper = self.year_limits_
            X.loc[X["year"] < lower, "year"] = np.nan
            X["year"] = X["year"].clip(lower=lower, upper=upper)
        # MILEAGE
        if self.upper_mileage_ is not None:
            X["mileage"] = np.clip(X["mileage"], 0, self.upper_mileage_)
        # MPG
        if self.mpg_low_ is not None and self.mpg_high_ is not None:
            X["mpg"] = np.clip(X["mpg"], self.mpg_low_, self.mpg_high_)
        # TAX
        if self.upper_tax_ is not None:
            X["tax"] = np.clip(X["tax"], 0, self.upper_tax_)
        # ENGINE SIZE
        if "engineSize" in X.columns:
            lower, upper = self.engine_limits_
            X["engineSize"] = X["engineSize"].clip(lower=lower, upper=upper)
        # Logical rules
        if "year" in X.columns and "mileage" in X.columns:
            current_year = 2025
            mask = (current_year - X["year"] <= 3) & (X["mileage"] > 100000)
            X.loc[mask, "year"] = np.nan
        if "engineSize" in X.columns and "mpg" in X.columns:
            mask = (X["engineSize"] > 4.0) & (X["mpg"] > 60)
            X.loc[mask, "mpg"] = np.nan
        if "previousOwners" in X.columns:
            X["previousOwners"] = X["previousOwners"].clip(lower=0, upper=10).round()
        if "paintQuality%" in X.columns:
            X["paintQuality%"] = X["paintQuality%"].clip(lower=self.paint_min_, upper=self.paint_max_)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features (car_age, mileage bins, etc.)."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "year" in X.columns:
            X["car_age"] = 2025 - X["year"]
            X["age_squared"] = X["car_age"] ** 2
        if "mileage" in X.columns:
            X["mileage_per_year"] = X["mileage"] / (X["car_age"] + 1)
            mileage_bins = [0, 10000, 50000, 100000, 150000, np.inf]
            labels = ["0-10k", "10-50k", "50-100k", "100-150k", "150k+"]
            X["mileage_bin"] = pd.cut(
                X["mileage"],
                bins=mileage_bins,
                labels=labels,
                include_lowest=True,
            )
        if all(col in X.columns for col in ["tax", "engineSize"]):
            X["tax_to_engine_ratio"] = X["tax"] / (X["engineSize"] + 1e-6)
        if "fuelType" in X.columns:
            eco = ["hybrid", "electric"]
            X["is_eco"] = X["fuelType"].isin(eco).astype(int)
        if "Brand" in X.columns:
            luxury = ["bmw", "audi", "mercedes"]
            X["is_luxury"] = X["Brand"].isin(luxury).astype(int)
        return X


class HybridImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps the impute_missing_values_hybrid helper so it can be plugged into an SK pipeline.
    """

    def __init__(self, create_flags: bool = True) -> None:
        self.create_flags = create_flags

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HybridImputerTransformer":
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_clean, _, _ = impute_missing_values_hybrid(X, X.copy(), X.copy(), create_flags=self.create_flags)
        # Ensure original column ordering is kept when possible
        return X_clean


@dataclass
class PipelineConfig:
    create_missing_flags: bool = True
    random_state: int = 42
    n_jobs: int = -1
    n_estimators: int = 300
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 3


def build_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Builds the full preprocessing + RF pipeline."""
    if config is None:
        config = PipelineConfig()

    numerical_features = [
        "mileage",
        "mpg",
        "tax_to_engine_ratio",
        "car_age",
        "year",
        "engineSize",
        "mileage_per_year",
    ]
    low_card = ["transmission", "fuelType", "mileage_bin"]
    high_card = ["model", "Brand"]
    passthrough_cols = ["is_luxury", "is_eco"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", RobustScaler())]), numerical_features),
            ("low_card", OneHotEncoder(handle_unknown="ignore"), low_card),
            ("high_card", TargetEncoder(smooth=12), high_card),  # target encoding for Brand/model
            ("pass", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    pipeline = Pipeline(steps=[
        ("categorical_normalizer", CategoricalNormalizer()),
        ("outlier_clipper", OutlierClipper()),
        ("feature_engineer", FeatureEngineer()),
        ("hybrid_imputer", HybridImputerTransformer(create_flags=config.create_missing_flags)),
        ("preprocessor", preprocessor),
        ("regressor", rf),
    ])
    return pipeline


__all__ = ["build_pipeline", "PipelineConfig"]
