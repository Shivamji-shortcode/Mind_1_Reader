import polars as pl
import re

class CleaningActions:
    @staticmethod
    def strip_currency(df, column_name):
        """Action 1: Universal symbol stripper. Handles $, €, £, and commas."""
        return df.with_columns(
            pl.col(column_name)
            .str.replace_all(r"[^0-9.]", "") # Removes everything except digits and dots
            .cast(pl.Float64, strict=False)
        )

    @staticmethod
    def clean_percentage(df, column_name):
        """Action 2: Converts percentage strings (50%) to floats (0.5)."""
        return df.with_columns(
            (pl.col(column_name)
            .str.replace("%", "")
            .cast(pl.Float64, strict=False) / 100)
        )

    @staticmethod
    def unify_date(df, column_name):
        """Action 3: Standardizes dates to YYYY-MM-DD."""
        return df.with_columns(
            pl.coalesce([
                pl.col(column_name).str.to_date("%Y-%m-%d", strict=False),
                pl.col(column_name).str.to_date("%d/%m/%Y", strict=False),
                pl.col(column_name).str.to_date("%b %d, %Y", strict=False),
            ]).alias(column_name)
        )

    @staticmethod
    def smart_impute(df, column_name):
        """Action 4: Advanced Imputation (Mean/Median for Numbers, Mode for Text)."""
        col = df[column_name]
        
        # Check if column is numerical
        if col.dtype in [pl.Float64, pl.Int64]:
            skewness = col.skew()
            # If data is skewed (outliers present), Median is safer than Mean
            if skewness is not None and abs(skewness) > 1:
                return df.with_columns(pl.col(column_name).fill_null(col.median()))
            else:
                return df.with_columns(pl.col(column_name).fill_null(col.mean()))
        else:
            # For categorical data (Color, Category), use the most frequent value
            mode_val = col.mode()
            if len(mode_val) > 0:
                return df.with_columns(pl.col(column_name).fill_null(mode_val[0]))
            return df

    @staticmethod
    def get_profit_insight(df, price_col="price", discount_col="discount"):
        """Action 5: Executive Insight - Calculates Net Revenue/Profit per row."""
        # This handles the 'Attention' part of your requirement
        return df.with_columns(
            (pl.col(price_col).fill_null(0) * (1 - pl.col(discount_col).fill_null(0)))
            .alias("net_revenue")
        )