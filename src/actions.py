import polars as pl
import numpy as np

class CleaningActions:
    @staticmethod
    @staticmethod
    @staticmethod
    def strip_currency(df, column_name):
        """Action 0: Safe symbol stripper."""
        # SHIELD: If it is already a Number or a Date, skip string operations
        if df[column_name].dtype != pl.Utf8:
            return df
            
        return df.with_columns(
            pl.col(column_name)
            .str.replace_all(r"[^0-9.]", "")
            .cast(pl.Float64, strict=False)
        )

    @staticmethod
    def clean_percentage(df, column_name):
        """Action 1: Safe percentage converter."""
        # SHIELD: If it's not text, we can't search for '%'
        if df[column_name].dtype != pl.Utf8:
            return df
            
        return df.with_columns(
            (pl.col(column_name)
            .str.replace("%", "")
            .cast(pl.Float64, strict=False) / 100)
        )
    
    @staticmethod
    def unify_date(df, column_name):
        """Action 2: Standardizes dates, but ONLY if the column is currently text."""
        # GUARD: If the brain tries to 'unify_date' on a column that is already 
        # a Number (Float) or a Date, just return the dataframe.
        if df[column_name].dtype != pl.Utf8:
            return df

        return df.with_columns(
            pl.coalesce([
                pl.col(column_name).str.to_date("%Y-%m-%d", strict=False),
                pl.col(column_name).str.to_date("%d/%m/%Y", strict=False),
                pl.col(column_name).str.to_date("%b %d, %Y", strict=False),
            ]).alias(column_name)
        )
    

    @staticmethod
    def smart_impute(df, column_name):
        """Action 3: Fills nulls based on type, with a safety check for empty columns."""
        col_data = df[column_name]
        
        # If the whole column is null, we can't calculate mean/mode. Just return.
        if col_data.null_count() == len(df):
            return df
            
        if col_data.dtype in [pl.Float64, pl.Int64]:
            fill_value = col_data.mean()
        else:
            # Get the mode safely
            mode_series = col_data.mode()
            fill_value = mode_series[0] if len(mode_series) > 0 else None

        # Only apply if we actually found a value to fill with
        if fill_value is not None:
            return df.with_columns(pl.col(column_name).fill_null(fill_value))
        
        return df
    @staticmethod
    def remove_outliers(df, column_name):
        """Action 4: Removes numerical outliers safely."""
        # 1. Guardrail: Only run on numeric columns
        if df[column_name].dtype not in [pl.Float64, pl.Int64]:
            return df
            
        # 2. Guardrail: Ensure there are actual values to calculate quantiles
        if df[column_name].null_count() == len(df):
            return df

        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)

        # 3. Check if quantiles were successfully calculated
        if q1 is None or q3 is None:
            return df

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return df.filter((pl.col(column_name) >= lower) & (pl.col(column_name) <= upper))
    

    @staticmethod
    def construct_profit_report(df):
        """Action 5: Indestructible Profit Constructor."""
        cols = df.columns
        # 1. Identity the Pillars
        price_col = next((c for c in cols if any(k in c.lower() for k in ['price', 'mrp', 'cost'])), None)
        discount_col = next((c for c in cols if any(k in c.lower() for k in ['discount', 'off', 'pct'])), None)

        if not price_col:
            return df

        try:
            # 2. Prepare Price (Clean strings on the fly if necessary)
            if df[price_col].dtype == pl.Utf8:
                price_vals = df[price_col].str.replace_all(r"[^0-9.]", "").cast(pl.Float64, strict=False).fill_null(0)
            else:
                price_vals = df[price_col].cast(pl.Float64, strict=False).fill_null(0)

            # 3. Prepare Discount
            if discount_col:
                if df[discount_col].dtype == pl.Utf8:
                    disc_vals = df[discount_col].str.replace_all(r"[^0-9.]", "").cast(pl.Float64, strict=False).fill_null(0)
                else:
                    disc_vals = df[discount_col].cast(pl.Float64, strict=False).fill_null(0)
                
                # If it's a percentage (e.g. 50.0), convert to decimal (0.5)
                if disc_vals.mean() > 1:
                    disc_vals = disc_vals / 100
            else:
                disc_vals = 0

            # 4. Final Calculation
            return df.with_columns((price_vals * (1 - disc_vals)).alias("net_revenue"))
            
        except Exception:
            # If something goes wrong, just return the original DF so the loop doesn't crash
            return df
    

    @staticmethod
    def normalize_columns(df, column_name):
        """Action 6: Only trim/lowercase if it is text."""
        if df[column_name].dtype == pl.Utf8:
            return df.with_columns(pl.col(column_name).str.strip_chars().str.to_lowercase())
        return df