import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load Data 
df = pd.read_csv('users.csv')

# ------------ Basic Exploration -------------
print("----- Shape ----")
print("Rows, Columns:", df.shape)

print("/n---- Head ----")
print(df.head())

print("/n---- Tail ----")
print(df.tail())

print("/n---- Random Sample ----")
print(df.sample(5))

print("/n----- Columns -----")
print(df.columns.tolist())

print("/n----- Data Types -----")
print(df.dtypes)    

print("/n----- Info -----")
print(df.info())

print("/n----- Summary Statistics Describe -----")
print(df.describe())

print("\n---- Summary Statistics (All Columns) ----")
print(df.describe(include='all'))

print("/n----- Missing Values -----")
print(df.isnull().sum())

print("/n----- Duplicates -----")
print(df.duplicated().sum())

print("/n----- Unique Values -----")
print(df.nunique())

# ----- Value Counts for Important Categorical Columns -----
categorical_columns = ['gender', 'bloodGroup', 'eyeColor', 'role']
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())
    print(f"\nNormalized (percent) for {col}:")
    print(df[col].value_counts(normalize=True))

# ----- Correlation (Numeric Columns) -----
print("\n----- Correlation Matrix -----")
numeric_cols = ['age', 'height', 'weight']
print(df[numeric_cols].corr())


def parse_address(addr_str):
    try:
        addr_dict = json.loads(addr_str.replace("'", "\""))  # fix single quotes to double
        return addr_dict.get('city', 'Unknown') 
    except:
        return 'Unknown'
df['city'] = df['address'].apply(parse_address)

# ------ Address ------
df['address'] = df['address'].apply(lambda x: json.loads(x.replace("'", '"')))
df['street'] = df['address'].apply(lambda x: x['address'])
df['city'] = df['address'].apply(lambda x: x['city'])
df['state'] = df['address'].apply(lambda x: x['state'])

# ------ Bank ------
df['bank'] = df['bank'].apply(lambda x: json.loads(x.replace("'", '"')))
df['bank_cardNumber'] = df['bank'].apply(lambda x: x['cardNumber'])
df['bank_cardType'] = df['bank'].apply(lambda x: x['cardType'])
df['bank_currency'] = df['bank'].apply(lambda x: x['currency'])

# ------ Crypto ------
df['crypto'] = df['crypto'].apply(lambda x: json.loads(x.replace("'", '"')))
df['crypto_coin'] = df['crypto'].apply(lambda x: x['coin'])
df['crypto_wallet'] = df['crypto'].apply(lambda x: x['wallet'])

# ------ Hair ------
df['hair'] = df['hair'].apply(lambda x: json.loads(x.replace("'", '"')))
df['hair_color'] = df['hair'].apply(lambda x: x['color'])
df['hair_type'] = df['hair'].apply(lambda x: x['type'])

# ------ Company ------
df['company'] = df['company'].apply(lambda x: json.loads(x.replace("'", '"')))
df['company_name'] = df['company'].apply(lambda x: x.get('name','Unknown'))
df['company_department'] = df['company'].apply(lambda x: x.get('department','Unknown'))
df['company_title'] = df['company'].apply(lambda x: x.get('title','Unknown'))

# ----- Drop original complex columns -----
df.drop(columns=['address','company','bank','crypto','hair'], inplace=True)

print("\n----- Cleaned DataFrame -----")
print(df.head())

# ----- Missing Values (Data Cleaning / Preparation) -----
print("\n----- Missing Values After Parsing -----")
for col in ['age', 'height', 'weight']:
    df[col].fillna(df[col].median(), inplace=True)

df['maidenName'].fillna('Unknown', inplace=True)
df['company_department'].fillna('Unknown', inplace=True)
df['company_title'].fillna('Unknown', inplace=True)
print(df.isnull().sum())

# ----- Feature Engineering -----

#full name
df['full_name'] = df['firstName'] + " " + df['lastName']

# BMI
df['BMI'] = df['weight'] / (df['height']/100)**2  # height 
df['BMI_category'] = pd.cut(df['BMI'], bins=[0,18.5,25,30,100], 
                            labels=['Underweight','Normal','Overweight','Obese'])

# Age Group
bins = [0,25,30,35,40,100]
labels = ['20-25','26-30','31-35','36-40','41+']
df['Age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

print(df[['BMI','BMI_category','Age_group']].head(10))
print(df['BMI'].describe())
print(df['BMI_category'].value_counts())
print(df['Age_group'].value_counts())
print(df['BMI_category'].value_counts(normalize=True))
print(df['Age_group'].value_counts(normalize=True))

# ----- Analysis -----

# Set general style
# ----- Analysis -----
sns.set_style("whitegrid")
palette_burgundy = ['#800020', '#B22222', "#FFAFAF", "#B7688D", "#8B0048"]

# Average age
print(f"Average age of users: {df['age'].mean():.2f}")

# Average age by gender
print("\nAverage age by gender:")
print(df.groupby('gender')['age'].mean())

# Users per gender
print("\nNumber of users per gender:")
print(df['gender'].value_counts())

# Top 10 cities
print("\nTop 10 cities with most users:")
print(df['city'].value_counts().head(10))

# Average height & weight
print(f"\nAverage height: {df['height'].mean():.2f} cm")
print(f"Average weight: {df['weight'].mean():.2f} kg")

# ==== Scatter plots: Age vs Height/Weight ====
plt.figure(figsize=(12,5))
sns.scatterplot(data=df, x='age', y='height', hue='gender', palette=["#FFAFAF",'#800020'])
plt.title('Age vs Height')
plt.show()

plt.figure(figsize=(12,5))
sns.scatterplot(data=df, x='age', y='weight', hue='gender', palette=["#FFAFAF", '#800020'])
plt.title('Age vs Weight')
plt.show()

# ==== Histograms / Countplots ====
plt.figure(figsize=(10,6))
sns.histplot(df['age'], bins=10, kde=True, color='#B22222')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['height'], bins=10, kde=True, color="#B7688D")
plt.title('Height Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['weight'], bins=10, kde=True, color='#8B0000')
plt.title('Weight Distribution')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='gender', palette=["#FFAFAF", '#800020'])
plt.title('Number of Users by Gender')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='BMI_category', palette=palette_burgundy[:len(df['BMI_category'].unique())])
plt.title('Number of Users by BMI Category')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='bloodGroup', palette=palette_burgundy[:len(df['bloodGroup'].unique())])
plt.title('Users by Blood Group')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Age_group', y='BMI', palette=palette_burgundy[:len(df['Age_group'].unique())])
plt.title('Average BMI by Age Group')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='eyeColor', palette=palette_burgundy[:len(df['eyeColor'].unique())])
plt.title('Users by Eye Color')
plt.xticks(rotation=45)
plt.show()

# Role Pie Chart
plt.figure(figsize=(6,6))
df['role'].value_counts().plot.pie(autopct='%1.1f%%', colors=palette_burgundy[:len(df['role'].unique())])
plt.title('Role Proportion')
plt.ylabel('')
plt.show()