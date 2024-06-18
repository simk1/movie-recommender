from surprise import Dataset, KNNBasic, accuracy, SVD
from surprise.model_selection import train_test_split

# Load the MovieLens 100k dataset
movie_data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and testing sets
trainset, testset = train_test_split(movie_data, test_size=0.2, random_state=42)

# Initialize and fit the KNNBasic recommender system
movie_recommender = KNNBasic()
movie_recommender.fit(trainset)

# Initialize and fit the SVD recommender system
svd_recommender = SVD()
svd_recommender.fit(trainset)

# Generate predictions on the test set
predictions = movie_recommender.test(testset)
svd_predictions = svd_recommender.test(testset)


# Calculate and print RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
svd_rmse = accuracy.rmse(svd_predictions)

print(f"RMSE: {rmse}")
print(f"SVD RMSE: {svd_rmse}")


