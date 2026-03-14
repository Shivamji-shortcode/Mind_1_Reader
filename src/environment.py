import polars as pl
import numpy as np
from actions import CleaningActions
from brain import extract_global_features

class DataCleaningEnv:
    def __init__(self, file_path):
        # 1. Load data - can be expanded later for MongoDB/SQL
        self.df = pl.read_csv(file_path)
        self.columns = self.df.columns
        self.current_col_idx = 0
        self.total_cols = len(self.columns)

    def get_state(self):
        """
        Returns the GLOBAL state of the current column.
        Mind 1 looks at the whole column to decide the best normalization strategy.
        """
        if self.current_col_idx >= self.total_cols:
            return None
        
        col_name = self.columns[self.current_col_idx]
        # This calls the statistical 'eyes' we built in the brain
        return extract_global_features(self.df, col_name)

    def step(self, action_id):
        """
        The agent takes an action on the entire column based on its features.
        Action Map:
        0: Strip Currency | 1: Clean Percent | 2: Unify Date 
        3: Smart Impute (Mean/Median/Mode) | 4: Profit Calculation
        """
        col_name = self.columns[self.current_col_idx]
        
        # Execute Action
        if action_id == 0:
            self.df = CleaningActions.strip_currency(self.df, col_name)
        elif action_id == 1:
            self.df = CleaningActions.clean_percentage(self.df, col_name)
        elif action_id == 2:
            self.df = CleaningActions.unify_date(self.df, col_name)
        elif action_id == 3:
            self.df = CleaningActions.smart_impute(self.df, col_name)
        elif action_id == 4:
            # If the agent identifies a financial column, it runs profit logic
            self.df = CleaningActions.get_profit_insight(self.df)

        # Reward Logic: Did we reduce Nulls or fix the Type?
        null_count_after = self.df[col_name].null_count()
        if null_count_after == 0:
            reward = 20  # Perfect clean/normalization
        else:
            reward = -5  # Still messy

        # Move to the next column
        self.current_col_idx += 1
        done = self.current_col_idx >= self.total_cols
        
        return reward, done

    def reset(self):
        """Resets the environment to start analyzing from the first column."""
        self.current_col_idx = 0
        return self.get_state()

    def get_executive_summary(self):
        """
        Fulfills your 'Data Outline' requirement.
        Prints profit/loss and attention areas.
        """
        print("--- Mind 1 Executive Report ---")
        if "net_revenue" in self.df.columns:
            total_profit = self.df["net_revenue"].sum()
            print(f"Total Projected Profit: ${total_profit:,.2f}")
        
        for col in self.df.columns:
            nulls = self.df[col].null_count()
            if nulls > 0:
                print(f"ATTENTION REQUIRED: Column '{col}' still has {nulls} missing values.")