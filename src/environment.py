import polars as pl
import numpy as np
from actions import CleaningActions


class DataCleaninEnv:
    def __init__(self, file_path):
        # load our Chaos_master
        self.df = pl.read_csv(file_path)
        self.current_row = 0
        self.total_rows = self.df.height

    def get_state(self):
        """Returns the current row on ehich agent is looking at ..."""
        if self.current_row >= self.total_rows:
            return None
        # Convert row to a dictionary so that an agent can read it 
        return self.df.row(self.current_row, named=True)
    
    def step(self, action_id, column_name):
        """
        The agent takes an action on a specific column.
        Action-id. 0 : Strip currency 
        Action_id . 1 : Clean Percent
        Action_id 2 . Unifiy the date"""
        # 1. Capture values before action to compare 
        # We use a slice to keep it fast 
        old_df = self.df.clone()

        # 2. Execute the choosen Action 
        if action_id == 0:
            self.df = CleaningActions.strip_currency(self.df, column_name)
        elif action_id == 1:
            self.df = CleaningActions.clean_percentage(self.df, column_name)
        elif action_id == 2:

            self.df = CleaningActions.unify_date(self.df, column_name)
        
        # Calculate the reward (Briann Logic)
        new_val = self.df.select(pl.col(column_name)).row(self.current_row)[0]

        if new_val is not None:
            reward = 10     # Success Data is now a valid number/ date
        else:
            reward = -20    # Failure the action resulted in the null values (data loss)
        
        # Move to the next rows 
        self.current_row += 1
        done = self.current_row >= self. total_rows
        return reward, done
    
    def reset(self):
        """Resets the Gym so Mind_1 can new trainign round ."""
        self.current_row = 0
        return self.get_state()