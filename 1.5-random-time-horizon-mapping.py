import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import random

df = pd.read_parquet("df-a255fe46-cf75-4d24-accb-21fe548c43a1.parquet.gzip").drop(
    columns="churn"
)

# Assuming your dataframe is called df
# First, find contact_rk_gap's that have any membership_year > 70
contact_rk_gap_to_remove = df.loc[df["membership_year"] > 70, "contact_rk_gap"].unique()

# Then, filter the dataframe to exclude these contact_rk_gap's
df = df[~df["contact_rk_gap"].isin(contact_rk_gap_to_remove)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Preprocessing
# Encode categorical variables
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
# Convert booleans to int
for col in ["is_household", "gave_feedback"]:
    df[col] = df[col].astype(int)

# Feature columns
feature_cols = [
    "year",
    "membership_year",
    # "price",
    "is_household",
    "vehicle_count",
    "vehicle_age_mean",
    "beløp",
    "gave_feedback",
    "mean_feedback",
    "age",
    "gender",
    "sentralitetsindex",
]

# Compute per-customer total membership (churn horizon)
churn_horizon = df.groupby("contact_rk_gap")["membership_year"].max()

# Fit scaler on all feature data
df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

# Build examples list of (customer, seq_tensor, length, label)
examples = []
for customer, group in tqdm(df.groupby("contact_rk_gap"), desc="Building examples"):
    group = group.sort_values("membership_year")

    max_year = churn_horizon.loc[customer]
    if max_year == 1:
        selected_horizon = 1
    elif max_year == 2:
        selected_horizon = 2
    else:  # max_year >= 3
        selected_horizon = random.randrange(1, 4)

    # Remove the last `selected_horizon` rows
    group = group.iloc[: -(selected_horizon - 1)] if selected_horizon != 1 else group

    # Now continue as normal
    seq = group[feature_cols].values
    seq_tensor = torch.tensor(seq, dtype=torch.float)
    length = seq_tensor.size(0)

    label = selected_horizon - 1  # map 1→0, 2→1, 3→2
    examples.append((customer, seq_tensor, length, label))

# 3. Split by customer into train/val
customers = [ex[0] for ex in examples]
train_cust, val_cust = train_test_split(customers, test_size=0.2, random_state=42)

train_examples = []
val_examples = []

for ex in tqdm(examples, desc="Splitting examples"):
    if ex[0] in train_cust:
        train_examples.append(ex[1:])
    else:
        val_examples.append(ex[1:])

# Save the examples
torch.save(train_examples, "all/train_examples.pt")
torch.save(val_examples, "all/val_examples.pt")
