import json

ipynb_path = r'd:\Distributed-Biometric-Risk-Categorization\RFC_NN\model.ipynb'

with open(ipynb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# The cell to modify is the one with id "35b22490" or containing "Global Preprocessors"
new_preprocess_source = [
    "# Global Preprocessors\n",
    "scaler = StandardScaler()\n",
    "le_gender = LabelEncoder()\n",
    "global_fit_df = pd.concat([pd.read_csv('ward_a_vitals.csv'), pd.read_csv('ward_b_vitals.csv'), pd.read_csv('ward_c_vitals.csv')])\n",
    "numerical_cols = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Derived_BMI', 'Derived_HRV']\n",
    "scaler.fit(global_fit_df[numerical_cols])\n",
    "le_gender.fit(global_fit_df['Gender'])\n",
    "\n",
    "def preprocess_node_data(df):\n",
    "    df = df.copy()\n",
    "    df[numerical_cols] = scaler.transform(df[numerical_cols])\n",
    "    df['Gender'] = le_gender.transform(df['Gender'])\n",
    "    df['Risk Category'] = df['Risk Category'].map({'Low Risk': 0, 'High Risk': 1})\n",
    "    X = df.drop(columns=['Risk Category']).values\n",
    "    y = df['Risk Category'].values\n",
    "    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "def split_node_data(X, y, test_size=0.2):\n",
    "    return train_test_split(X, y, test_size=test_size, random_state=42)\n",
    "\n",
    "def augment_with_safety_cases(X_tensor, y_tensor, count=100):\n",
    "    \"\"\"\n",
    "    Injects synthetic critical cases into the training set so the NN learns \n",
    "    that extreme vitals = High Risk, even if other vitals are normal.\n",
    "    \"\"\"\n",
    "    # Define extreme cases (unscaled)\n",
    "    # Each row: HR, RR, Temp, SpO2, SysBP, DiaBP, Age, Gender(0/1), BMI, HRV\n",
    "    safety_cases = [\n",
    "        [0, 10, 37.0, 98, 100, 70, 50, 0, 25, 0.05],    # Cardiac Arrest (HR=0)\n",
    "        [140, 35, 41.5, 99, 90, 60, 40, 1, 24, 0.03],   # Extreme Fever (Temp=41.5)\n",
    "        [80, 25, 36.5, 99, 65, 112, 68, 1, 26.8, 0.2],   # User's Case (Crisis BP + Normal SpO2)\n",
    "        [80, 20, 36.5, 82, 110, 75, 30, 0, 22, 0.08]     # Severe Hypoxia (SpO2=82)\n",
    "    ]\n",
    "    \n",
    "    # Convert to df to use scaler\n",
    "    cols = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Gender', 'Derived_BMI', 'Derived_HRV']\n",
    "    df_safety = pd.DataFrame(safety_cases, columns=cols)\n",
    "    \n",
    "    # Preprocess (Scale everything except Gender which we manually set as 0/1)\n",
    "    df_safety[numerical_cols] = scaler.transform(df_safety[numerical_cols])\n",
    "    \n",
    "    X_safety = torch.tensor(df_safety.values, dtype=torch.float32)\n",
    "    y_safety = torch.ones(len(safety_cases), dtype=torch.long) # All are High Risk\n",
    "    \n",
    "    # Repeat to give enough weight in the mini-batch\n",
    "    X_safety_repeated = X_safety.repeat(count, 1)\n",
    "    y_safety_repeated = y_safety.repeat(count)\n",
    "    \n",
    "    # Concatenate to original data\n",
    "    return torch.cat([X_tensor, X_safety_repeated]), torch.cat([y_tensor, y_safety_repeated])\n",
    "\n",
    "X_a_full, y_a_full = preprocess_node_data(pd.read_csv('ward_a_vitals.csv'))\n",
    "X_b_full, y_b_full = preprocess_node_data(pd.read_csv('ward_b_vitals.csv'))\n",
    "X_c_full, y_c_full = preprocess_node_data(pd.read_csv('ward_c_vitals.csv'))\n",
    "\n",
    "X_a_train_orig, X_a_val, y_a_train_orig, y_a_val = split_node_data(X_a_full, y_a_full)\n",
    "X_b_train_orig, X_b_val, y_b_train_orig, y_b_val = split_node_data(X_b_full, y_b_full)\n",
    "X_c_train_orig, X_c_val, y_c_train_orig, y_c_val = split_node_data(X_c_full, y_c_full)\n",
    "\n",
    "# Augment ONLY the training sets\n",
    "X_a, y_a = augment_with_safety_cases(X_a_train_orig, y_a_train_orig)\n",
    "X_b, y_b = augment_with_safety_cases(X_b_train_orig, y_b_train_orig)\n",
    "X_c, y_c = augment_with_safety_cases(X_c_train_orig, y_c_train_orig)\n",
    "\n",
    "print(f\"Data Augmentation Complete. New Ward A training size: {len(X_a)}\")\n"
]

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if 'Global Preprocessors' in src:
            cell['source'] = new_preprocess_source
            cell['outputs'] = []
            cell['execution_count'] = None

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("model.ipynb: Successfully injected Data Augmentation logic.")
