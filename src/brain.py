import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import re

def extract_global_features(df, column_name):
    """
    Conceptual Vision: Identifies columns like MRP, Price, or Cost 
    even if the names are different.
    """
    col = df[column_name]
    features = []
    col_name_lower = column_name.lower()
    
    # --- 1. STATISTICAL DNA ---
    features.append(1.0 if col.dtype in [pl.Float64, pl.Int64] else 0.0) # Is Numeric
    features.append(col.null_count() / len(df)) # Null Density

    # --- 2. FUZZY SEMANTIC MAPPING (Handles 'mrp', 'price', etc.) ---
    # Concept: Price Pillar
    price_synonyms = ['price', 'mrp', 'cost', 'val', 'amt', 'retail', 'list']
    is_price_concept = any(s in col_name_lower for s in price_synonyms)
    
    # Concept: Discount Pillar
    discount_synonyms = ['discount', 'off', 'pct', 'promo', 'less']
    is_discount_concept = any(s in col_name_lower for s in discount_synonyms)

    # --- 3. DATA CONTENT CHECK (The 'DNA' Check) ---
    # We look at the first 10 rows to see what the data actually looks like
    sample_data = str(col.slice(0, 10).to_list())
    has_currency_symbol = any(s in sample_data for s in ['$', '€', '£', '₹'])
    has_percent_symbol = '%' in sample_data

    # Combine Name + Content to create final Brain Signals
    # This ensures that even if name is 'Column_A', if it has '$', it's a Price.
    features.append(1.0 if is_price_concept or has_currency_symbol else 0.0)    # FINAL PRICE TAG
    features.append(1.0 if is_discount_concept or has_percent_symbol else 0.0) # FINAL DISCOUNT TAG

    # --- 4. CATEGORICAL TAG (For the 'Attention' logic) ---
    if col.dtype == pl.Utf8:
        unique_ratio = col.n_unique() / max(len(df), 1)
        features.append(1.0 if unique_ratio < 0.2 else 0.0) # Likely a category like "Color"
    else:
        features.append(0.0)

    # Total features: 5
    return features

class Mind1Net(nn.Module):
    def __init__(self, input_size=5, output_size=8):
        super(Mind1Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

class Mind1Agent:
    def __init__(self, input_size=5, output_size=8):
        self.model = Mind1Net(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()