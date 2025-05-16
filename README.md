# python8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set visualization styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
%matplotlib inline

try:
    df = pd.read_csv('owid-covid-data.csv', parse_dates=['date'])
    print("✅ Dataset loaded successfully!")
    print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: File not found. Please download the dataset from Our World in Data")
    print("Download link: https://ourworldindata.org/covid-cases")

    # Display column information
print("\n📋 Column Information:")
print(df.info())

# Show first 5 rows
print("\n👀 First 5 Rows:")
display(df.head())

# Check missing values
print("\n🔍 Missing Values Summary:")
missing_data = df.isnull().sum().sort_values(ascending=False)
display(missing_data[missing_data > 0])

countries = ["United States", "India", "Brazil", "Germany", "Kenya", "China"]
df = df[df['location'].isin(countries)].copy()

# Verify country selection
print("\n🌍 Selected Countries:")
print(df['location'].unique())

# Plot total cases over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='total_cases', hue='location')
plt.title('Total COVID-19 Cases Over Time', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Total Cases (millions)')
plt.legend(title='Country')
plt.show()

# Plot death rates comparison
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='death_rate', hue='location')
plt.title('COVID-19 Death Rate Over Time', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Death Rate (Deaths/Cases)')
plt.legend(title='Country')
plt.show()

# Get latest data for each country
latest = df.sort_values('date').groupby('location').last().reset_index()

# Bar plot of total cases
plt.figure(figsize=(12, 6))
sns.barplot(data=latest, x='location', y='total_cases')
plt.title('Total COVID-19 Cases by Country')
plt.ylabel('Total Cases (millions)')
plt.xticks(rotation=45)
plt.show()

# Vaccination comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=latest, x='location', y='vaccination_rate')
plt.title('Vaccination Rates by Country')
plt.ylabel('Percentage Vaccinated')
plt.xticks(rotation=45)
plt.show()

try:
    # Prepare data for mapping
    map_df = df.dropna(subset=['iso_code']).sort_values('date')
    map_df = map_df.groupby('iso_code').last().reset_index()
    
    # Create interactive map
    fig = px.choropleth(map_df, 
                       locations="iso_code",
                       color="total_cases",
                       hover_name="location",
                       projection="natural earth",
                       title="Global COVID-19 Cases",
                       color_continuous_scale=px.colors.sequential.Plasma)
    fig.show()
except ImportError:
    print("Plotly not available. Install with: pip install plotly")

    # Select numerical columns for correlation
corr_cols = ['total_cases', 'total_deaths', 'total_vaccinations', 
             'population', 'population_density']
corr_df = df[corr_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of COVID-19 Metrics')
plt.show()

### Key Findings and Insights

1. **Case Trends**:  
   - The United States had the highest total cases, followed by India and Brazil
   - Germany maintained relatively lower case numbers compared to other major economies

2. **Vaccination Progress**:  
   - [Country] had the fastest vaccination rollout reaching [X]% of population by [date]
   - Developing nations showed slower vaccination rates likely due to supply constraints

3. **Death Rates**:  
   - Brazil showed the highest death rate at [X]%, possibly due to [reason]
   - Countries with robust healthcare systems maintained lower death rates

4. **Anomalies**:  
   - Noticeable spikes in cases correlate with new variant emergences
   - Vaccination rates plateaued in [country] around [date]

5. **Regional Comparisons**:  
   - North America and Europe had faster vaccine distribution
   - Africa showed delayed vaccination campaigns but steady case growth
  
   - # Save cleaned data
df.to_csv('cleaned_covid_data.csv', index=False)
print("📁 Cleaned dataset saved as 'cleaned_covid_data.csv'")

# Export notebook to HTML
!jupyter nbconvert --to html COVID19_Analysis.ipynb
print("📄 Notebook exported to HTML format")
