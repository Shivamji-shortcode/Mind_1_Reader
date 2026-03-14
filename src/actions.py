import polars as pl
import re

class CleaningAtions:
    @staticmethod
    def strip_currency(df,column_name):
        """Action 1: Remove symbols like $ and € and converts to float."""
        return df.with_columns(
            pl.col(column_name)
            .str.replace_all(r"[^0-9.]","") # removes everyhting except digits and dots
            .cast(pl.Float64, strict=False)
        )
    @staticmethod
    def clean_percentage(df, column_name):
        """Action 2: converts '-27% string to the -0.27 floates."""
        return df.with_columns(
            (pl.col(column_name)
            .str.replace("%", "")
            .cast(pl.Float64, strict=False) / 100)
        )
    @staticmethod
    def unify_date(df, column_name):
        """Action 3 : Attempts to parse multiple dates formats in to YYYY-MM-DD."""
        return df.with_columns(
            pl.col(column_name).str.to_date(
                formats = ["%Y-%m-%d", "%d/%m/%y", "%b %d, %y"],
                strict=False
            )
        )
