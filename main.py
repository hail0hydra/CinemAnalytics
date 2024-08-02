import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

u_data_path = 'ml-100K/u.data'
u_item_path = 'ml-100K/u.item'
u_user_path = 'ml-100K/u.user'

u_data = pd.read_csv(u_data_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
u_item = pd.read_csv(u_item_path, sep='|', header=None, encoding='latin1', 
                     names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                            'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
u_user = pd.read_csv(u_user_path, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

# Univariate plot for 'rating'
plt.figure(figsize=(10, 5))
sns.countplot(x='rating', data=u_data)
plt.title('Distribution of Ratings')
plt.show()

# Univariate plot for 'age'
plt.figure(figsize=(10, 5))
sns.histplot(u_user['age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Univariate plot for 'release date'
u_item['release_year'] = pd.to_datetime(u_item['release_date'], errors='coerce').dt.year
plt.figure(figsize=(10, 5))
sns.histplot(u_item['release_year'].dropna(), bins=30, kde=True)
plt.title('Distribution of Release Years')
plt.show()

# Univariate plot for 'gender'
plt.figure(figsize=(10, 5))
sns.countplot(x='gender', data=u_user)
plt.title('Distribution of Gender')
plt.show()

# Univariate plot for 'occupation'
plt.figure(figsize=(10, 5))
sns.countplot(y='occupation', data=u_user, order=u_user['occupation'].value_counts().index)
plt.title('Distribution of Occupation')
plt.show()

genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
              'Thriller', 'War', 'Western']
u_item_melted = u_item.melt(id_vars=['item_id', 'title', 'release_year'], value_vars=genre_cols, 
                            var_name='genre', value_name='is_genre')

u_item_genres = u_item_melted[u_item_melted['is_genre'] == 1]

genre_popularity = u_item_genres.groupby(['release_year', 'genre']).size().reset_index(name='count')

# Plotting the genre popularity over the years
plt.figure(figsize=(15, 10))
sns.lineplot(data=genre_popularity, x='release_year', y='count', hue='genre')
plt.title('Popularity of Genres Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies Released')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Merge the ratings with the item information
merged_data = pd.merge(u_data, u_item, on='item_id')

# Calculate average ratings and rating counts per movie
movie_ratings = merged_data.groupby('title').agg({'rating': ['mean', 'count']})
movie_ratings.columns = ['average_rating', 'rating_count']

# Filter movies with at least 100 ratings
popular_movies = movie_ratings[movie_ratings['rating_count'] >= 100]

# Get the top 25 movies by average rating
top_25_movies = popular_movies.sort_values(by='average_rating', ascending=False).head(25)

top_25_movies.reset_index(inplace=True)

# Plotting the top 25 movies by average rating
plt.figure(figsize=(12, 10))
sns.barplot(y='title', x='average_rating', data=top_25_movies, palette='viridis')
plt.title('Top 25 Movies by Average Rating (at least 100 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.show()

# Verify the statements regarding gender differences in genre preferences

# Merge user and ratings data
user_ratings = pd.merge(u_data, u_user, on='user_id')

# Merge with item data to get genres
user_ratings = pd.merge(user_ratings, u_item, on='item_id')

# Define a function to count the number of ratings per genre for each gender
def genre_count_per_gender(df, genre):
    return df[df[genre] == 1].groupby('gender')['rating'].count()

# Count ratings for Drama
drama_count = genre_count_per_gender(user_ratings, 'Drama')
print(f"Drama ratings by gender:\n{drama_count}")

# Count ratings for Romance
romance_count = genre_count_per_gender(user_ratings, 'Romance')
print(f"Romance ratings by gender:\n{romance_count}")

# Count ratings for Sci-Fi
scifi_count = genre_count_per_gender(user_ratings, 'Sci-Fi')
print(f"Sci-Fi ratings by gender:\n{scifi_count}")

# Verification statements
if drama_count['M'] > drama_count['F']:
    print("Men watch more Drama than women: True")
else:
    print("Men watch more Drama than women: False")

if romance_count['M'] > romance_count['F']:
    print("Men watch more Romance than women: True")
else:
    print("Men watch more Romance than women: False")

if scifi_count['F'] > scifi_count['M']:
    print("Women watch more Sci-Fi than men: True")
else:
    print("Women watch more Sci-Fi than men: False")

