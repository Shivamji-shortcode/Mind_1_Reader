import polars as pl
import numpy as np
from actions import CleaningActions
from brain import extract_global_features

class DataCleaningEnv:
    def __init__(self, file_path):
        # 1. Load data
        self.df = pl.read_csv(file_path)
        self.columns = self.df.columns
        self.current_col_idx = 0
        self.total_cols = len(self.columns)

    def get_state(self):
        """Returns the Semantic Profile of the current column."""
        if self.current_col_idx >= self.total_cols:
            return None
        
        col_name = self.columns[self.current_col_idx]
        # Calls the brain's role-aware feature extractor
        return extract_global_features(self.df, col_name)

    def step(self, action_id):
        """
        Mind 1 Environment Step Function
        Action Map:
        0: Strip Currency | 1: Clean Percent | 2: Unify Date 
        3: Smart Impute   | 4: Remove Outliers | 5: Profit Report
        6: Normalize
        """
        col_name = self.columns[self.current_col_idx]
        col_lower = col_name.lower()
        
        # --- 1. CAPTURE STATE BEFORE ACTION ---
        old_nulls = self.df[col_name].null_count()
        old_dtype = self.df[col_name].dtype

        # --- 2. EXECUTE ACTION ---
        if action_id == 0:
            self.df = CleaningActions.strip_currency(self.df, col_name)
        elif action_id == 1:
            self.df = CleaningActions.clean_percentage(self.df, col_name)
        elif action_id == 2:
            self.df = CleaningActions.unify_date(self.df, col_name)
        elif action_id == 3:
            self.df = CleaningActions.smart_impute(self.df, col_name)
        elif action_id == 4:
            self.df = CleaningActions.remove_outliers(self.df, col_name)
        elif action_id == 5:
            self.df = CleaningActions.construct_profit_report(self.df)
        elif action_id == 6:
            self.df = CleaningActions.normalize_columns(self.df, col_name)

        # --- 3. REWARD CALCULATION ---
        new_dtype = self.df[col_name].dtype
        new_nulls = self.df[col_name].null_count()
        reward = 0

        # A. Type Integrity (High Stakes Encouragement)
        if "price" in col_lower or "mrp" in col_lower:
            if new_dtype in [pl.Float64, pl.Int64]:
                reward += 250  # Increased Carrot
            elif new_dtype == pl.Date:
                reward -= 100  # Smaller Stick
        
        elif "date" in col_lower:
            if new_dtype == pl.Date:
                reward += 250  # Increased Carrot
            elif new_dtype in [pl.Float64, pl.Int64]:
                reward -= 100  # Smaller Stick

        # B. Null Reduction Reward
        if new_nulls < old_nulls:
            reward += 50
        if new_nulls == 0:
            reward += 30

        # C. Profit Report Jackpot
        if action_id == 5 and "net_revenue" in self.df.columns:
            mean_rev = self.df["net_revenue"].mean()
            if mean_rev is not None and mean_rev > 0:
                reward += 800  # Big win
            else:
                reward -= 50   # Small penalty for fail
        
        # Additional Global Reward check for Action 5 (from your previous logic)
        if action_id == 5 and "net_revenue" in self.df.columns:
             if self.df["net_revenue"].mean() is not None and self.df["net_revenue"].mean() > 0:
                reward += 100 # Jackpot for generating the report!

        # D. Action-Type Mismatch Penalty
        if action_id == 0 and "price" not in col_lower and "mrp" not in col_lower:
            reward -= 20

        # --- 4. ADVANCE STATE & TERMINAL REWARD ---
        self.current_col_idx += 1
        done = self.current_col_idx >= self.total_cols
        
        # THE "FINISH THE JOB" LOGIC (Encouragement version)
        if done:
            clean_cols_count = sum([1 for col in self.columns if self.df[col].null_count() == 0])
            
            # Threshold lowered to 40% to help the agent find the win
            if clean_cols_count < (len(self.columns) * 0.4):
                reward -= 100  # "Lazy Agent" Penalty (Lowered)
            else:
                reward += 1000 # "Professional Cleaner" Bonus (Increased)
        
        return reward, done
        
    def reset(self):
        """Resets the cycle for a new training episode."""
        self.current_col_idx = 0
        return self.get_state()

    def get_executive_summary(self):
        """The Output Outline: Shows Profit and Attention areas."""
        print("\n" + "="*30)
        print("MIND 1 EXECUTIVE SUMMARY")
        print("="*30)
        
        if "net_revenue" in self.df.columns:
            total_rev = self.df["net_revenue"].sum()
            print(f"TOTAL PROJECTED REVENUE: ${total_rev:,.2f}")
        
        print("\n--- COLUMN ATTENTION REPORT ---")
        for col in self.df.columns:
            nulls = self.df[col].null_count()
            status = "CLEAN" if nulls == 0 else f"DIRTY ({nulls} nulls)"
            print(f"Column: {col:.<20} {status}")
        print("="*30 + "\n")