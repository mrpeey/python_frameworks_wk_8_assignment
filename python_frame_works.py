#Part 1: Data Loading and Basic Exploration


#1. Download and load the data

# Load the metadata.csv file (assumed to be downloaded locally)
from matplotlib import image
import pandas



filepath = 'C:\\Users\\poulo\\Downloads\\metadata.csv (5)\\metadata.csv'

df = pandas.read_csv(filepath)
import pandas as pd

# Examine first few rows and structure
print(df.head())
print(df.info())


#2. Basic data exploration

# Count missing values per column
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

# Example: Drop rows where 'title' or 'publish_time' is missing (important columns)
df_cleaned = df.dropna(subset=['title', 'publish_time'])

# Convert publish_time to datetime and extract year
df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
df_cleaned['year'] = df_cleaned['publish_time'].dt.year

# Handle any remaining missing publish_time after conversion by dropping
df_cleaned = df_cleaned.dropna(subset=['publish_time'])

# Create a new column for abstract word count
df_cleaned['abstract_word_count'] = df_cleaned['abstract'].fillna('').apply(lambda x: len(x.split()))
print(df_cleaned[['title', 'publish_time', 'year', 'abstract_word_count']].head())


#3. Data cleaning

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Count papers by publication year
year_counts = df_cleaned['year'].value_counts().sort_index()
plt.figure(figsize=(10,5))
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title('Number of Publications Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Top journals by count
top_journals = df_cleaned['journal'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_journals.values, y=top_journals.index)
plt.title('Top Journals Publishing COVID-19 Research')
plt.xlabel('Number of Papers')
plt.show()

# Word frequency in titles
from collections import Counter
import re

title_words = ' '.join(df_cleaned['title'].dropna()).lower()
words = re.findall(r'\b\w{3,}\b', title_words)  # Words with length >= 3
word_counts = Counter(words)
most_common_words = dict(word_counts.most_common(100))

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(most_common_words)
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Distribution of paper counts by source (if source column exists)
if 'source_x' in df_cleaned.columns:
    source_counts = df_cleaned['source_x'].value_counts()
    plt.figure(figsize=(10,5))
    sns.barplot(x=source_counts.values, y=source_counts.index)
    plt.title('Paper Counts by Source')
    plt.show()
df = df_cleaned.copy()

#4. Data transformation

import streamlit as st

st.title("COVID-19 Research Papers Analysis")
st.write("Basic exploration and visualizations of the CORD-19 dataset metadata.")

# Interactive widget: select year range
min_year = int(df_cleaned['year'].min())
max_year = int(df_cleaned['year'].max())

selected_years = st.slider('Select publication year range:', min_year, max_year, (min_year, max_year))

filtered_df = df_cleaned[(df_cleaned['year'] >= selected_years[0]) & (df_cleaned['year'] <= selected_years[1])]

st.write(f"Showing data for years between {selected_years[0]} and {selected_years[1]}.")

# Show filtered data sample
st.dataframe(filtered_df[['title', 'year', 'journal']].head(10))

# Plot number of publications over time in selected range
year_counts_filtered = filtered_df['year'].value_counts().sort_index()
st.line_chart(year_counts_filtered)

# Dropdown for selecting top journal to display paper counts
top_journal_list = top_journals.index.tolist()
selected_journal = st.selectbox('Select Journal', top_journal_list)

journal_count = filtered_df[filtered_df['journal'] == selected_journal].shape[0]
st.write(f"Number of papers in {selected_journal}: {journal_count}")

# Display word cloud as an image in Streamlit
import numpy as np
st.pyplot(wordcloud.to_image())


# Part 3: Data Visualization

#5. Data visualization

# Calculate average title length by year
df['title_length'] = df['title'].fillna('').apply(len)
df['publish_year'] = df['publish_time'].dt.year
avg_title_length_by_year = df.groupby('publish_year')['title_length'].mean().reset_index()

# Plot average title length by year
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_title_length_by_year, x='publish_year', y='title_length')
plt.title('Average Title Length by Year')
plt.xlabel('Year')
plt.ylabel('Average Title Length')
plt.show()

# Bar plot of top 10 journals by article count
plt.figure(figsize=(12, 6))
sns.barplot(data=top_journals, x='journal', y='article_count')
plt.title('Top 10 Journals by Article Count')
plt.xlabel('Journal')
plt.ylabel('Article Count')
plt.xticks(rotation=45)
plt.show()

# Histogram of title lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['title_length'], bins=30, kde=True)
plt.title('Distribution of Title Lengths')
plt.xlabel('Title Length')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of title length vs. publish year
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='publish_year', y='title_length', alpha=0.5)
plt.title('Title Length vs. Publish Year')
plt.xlabel('Publish Year')
plt.ylabel('Title Length')
plt.show()

# Part 4: Advanced Analysis

#6. Advanced analysis

# Correlation between title length and publish year
correlation = df['title_length'].corr(df['publish_year'])
print(f"Correlation between title length and publish year: {correlation}")

# Trend analysis: Has title length increased over the years?
trend = avg_title_length_by_year['title_length'].pct_change().mean()
print(f"Average yearly percentage change in title length: {trend * 100:.2f}%")

# Identify journals with consistently high article counts over the years
consistent_journals = df.groupby(['journal', 'publish_year']).size().unstack(fill_value=0)
consistent_journals = consistent_journals[consistent_journals.sum(axis=1) > 50]  # Filter journals with more than 50 articles total
print(consistent_journals.head())

# Plot heatmap of article counts for consistent journals over the years
plt.figure(figsize=(12, 8))
sns.heatmap(consistent_journals, cmap='YlGnBu', cbar_kws={'label': 'Article Count'})
plt.title('Article Counts for Consistent Journals Over the Years')
plt.xlabel('Publish Year')
plt.ylabel('Journal')
plt.show()

# Save cleaned DataFrame to a new CSV file
df.to_csv('cleaned_metadata.csv', index=False)
print("Cleaned data saved to 'cleaned_metadata.csv'")

# Summary of findings
print("Summary of Findings:")
print(f"1. The dataset contains {df.shape[0]} articles after cleaning.")
print(f"2. The average title length is {df['title_length'].mean():.2f} characters.")
print(f"3. The correlation between title length and publish year is {correlation:.2f}.")
print(f"4. The average yearly percentage change in title length is {trend * 100:.2f}%.")
print("5. Top journals by article count have been visualized.")



#Part 4: Streamlit Application 

#7.Build a simple Streamlit app

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

wordcloud = WordCloud().generate("some text here")

fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis('off')
st.pyplot(fig)  
fig.savefig(wordcloud.__str__())

st.title("COVID-19 Research Papers Analysis")
st.write("Basic exploration and visualizations of the CORD-19 dataset metadata.")

# Interactive widget: select year range
min_year = int(df_cleaned['year'].min())
max_year = int(df_cleaned['year'].max())

selected_years = st.slider('Select publication year range:', min_year, max_year, (min_year, max_year))

filtered_df = df_cleaned[(df_cleaned['year'] >= selected_years[0]) & (df_cleaned['year'] <= selected_years[1])]

st.write(f"Showing data for years between {selected_years[0]} and {selected_years[1]}.")

# Show filtered data sample
st.dataframe(filtered_df[['title', 'year', 'journal']].head(10))

# Plot number of publications over time in selected range
year_counts_filtered = filtered_df['year'].value_counts().sort_index()
st.line_chart(year_counts_filtered)

# Dropdown for selecting top journal to display paper counts
top_journal_list = top_journals.index.tolist()
selected_journal = st.selectbox('Select Journal', top_journal_list)

journal_count = filtered_df[filtered_df['journal'] == selected_journal].shape[0]
st.write(f"Number of papers in {selected_journal}: {journal_count}")

# Display word cloud as an image in Streamlit
import numpy as np
st.pyplot(wordcloud.to_image())
