
# 🎬 NLP & Generative AI-Based Movie Recommendation System

## Final Report

Jupyter Notebook: https://github.com/mskeyashah/UCB-Project-Final-Report/blob/main/NLP%20%26%20Generative%20AI-Based%20Movie%20Recommendation%20System.ipynb

### 1. Define the Problem Statement
The objective of this project is to develop a recommendation system that suggests 3 to 4 similar movies or TV shows based on a given title. The system aims to integrate Natural Language Processing (NLP) and Generative AI to analyze both user reviews and movie metadata in order to generate relevant, personalized recommendations.
The main challenges include working with unstructured and subjective user reviews, integrating multiple metadata fields (such as genres, cast, plot summaries), and developing a model that can identify thematic and semantic similarities across a wide range of content. Additionally, the system must be scalable, interpretable, and capable of handling large volumes of movie data.
The anticipated benefit of this system is to enhance the movie-watching experience by providing users with high-quality, contextually appropriate recommendations that go beyond simple genre matching. This can help users discover lesser-known yet similar titles, improving engagement and satisfaction.

### 2. Model Outcomes or Predictions
The core outcome of this project is a list of 3 to 4 movies or shows that are similar to a user-provided title. These recommendations are generated based on both structured metadata and unstructured textual data (reviews and descriptions).
This project primarily uses unsupervised learning techniques. Specifically:
* Clustering (via K-Means) is used to identify groups of movies that share common themes or characteristics.
* Semantic similarity search (using sentence embeddings and K-Nearest Neighbors) is used to retrieve titles with high textual and contextual similarity to the input.
* Enhanced KNN with Sentiment integrates sentiment scores from user reviews to prioritize positively reviewed movies within the recommendation process.
These models do not rely on labeled outputs; rather, they identify patterns and similarities within the data to make recommendations.

### 3. Data Acquisition
The data for this project is sourced from two main locations:
1. TMDb Movie Metadata: The core dataset is sourced from the TMDb Kaggle Dataset, which includes detailed metadata for over 5,000 movies. This dataset provides information such as movie genres, cast, crew, plot summaries (overview), keywords, taglines, and language.
2. User Reviews and Ratings: Additional user-generated content is retrieved from the TMDb API using the movie_id from the above dataset. These reviews add valuable insights into viewer opinions and sentiment, allowing the model to incorporate subjective data into its recommendation process.

To assess the structure of the data and its potential, the Elbow Method was applied to determine the optimal number of clusters for K-Means. The elbow in the plot suggested an optimal number of 13 clusters, which balances granularity and performance.

### 4. Data Preprocessing and Preparation
a. Cleaning and Handling Missing Values:
Several columns with high proportions of missing or irrelevant data (such as homepage, runtime, and production_companies) were removed. Rows missing key information, particularly overview or title, were excluded to maintain dataset integrity. For JSON-like fields such as genres and keywords, custom parsing functions were written to extract and normalize their contents into plain text.
b. Data Splitting:
Because this project relies on unsupervised learning, traditional training and test splits are not strictly necessary. Instead, a cleaned dataset was prepared and saved as a CSV file for repeated use in clustering and embedding steps.
c. Text Preprocessing and Feature Engineering:
All relevant textual features (title, genres, keywords, overview, tagline, and user reviews) were cleaned and tokenized. This included:
* Lowercasing all text
* Removing non-alphabetic characters
* Eliminating stopwords
* Tokenizing using NLTK
The cleaned text from all sources was then combined into a single combined_text column, which served as the input for both clustering and embedding models.

### 5. Modeling
Two main models were implemented in this project:
1. TF-IDF + K-Means Clustering: This model vectorizes the combined_text of each movie using the Term Frequency-Inverse Document Frequency (TF-IDF) technique. The resulting feature matrix is then clustered using the K-Means algorithm. The optimal number of clusters was determined to be 13, using the Elbow Method.
2. Sentence Transformers + K-Nearest Neighbors: In order to capture deeper semantic meaning, the all-MiniLM-L6-v2 sentence transformer model from the Hugging Face library was used to embed the combined_text into dense vector representations. These embeddings were indexed using a cosine distance-based KNN model. For any input movie title, the system retrieves the top 4 closest embeddings (i.e., most similar movies).
3. Sentiment-Aware Embeddings + Enhanced KNN: This enhanced version integrates sentiment analysis of user reviews using the Twitter RoBERTa-base sentiment classifier. Each review was assigned a sentiment label (positive, neutral, negative), which was converted into a sentiment score and normalized. These sentiment scores were then concatenated with the sentence embeddings to create sentiment-augmented embeddings. The KNN model was retrained on this enhanced feature space to recommend movies that are not only similar in content, but also positively received by audiences

### 6. Model Evaluation
Type of Models Considered:
Since the project is focused on identifying relationships and groupings in unlabeled data, only unsupervised learning models were considered. This includes clustering models and semantic similarity-based recommendation systems.
Evaluation Metrics and Process:
* For Clustering (K-Means): The performance of the clustering model was evaluated using inertia and visualized with the Elbow Method. This helped determine that 13 clusters offered a suitable balance between cohesion and separation.
* For Embedding-Based Recommendations (KNN + Sentence Transformers): Evaluation was qualitative, based on the relevance and coherence of recommended titles. For example:
    * For the input title "Avatar", the recommendations included "Aliens," "The Abyss," and "Alien", all thematically aligned in terms of genre, visual storytelling, and director.
    * For "The Dark Knight Rises", the system recommended "The Dark Knight," "Batman Begins," and "Batman", reflecting strong thematic and narrative continuity.
* Enhanced KNN with Sentiment:
    * Improved upon previous recommendations by favoring positively-reviewed content. This made the output more emotionally resonant and increased user trust in the suggestions.
    * For movies in the action or thriller genre with mixed reviews, the sentiment-enhanced model filtered out low-rated ones and elevated titles with strong audience appreciation.
Final Model Selection:
The sentence embedding + KNN model was chosen as the final recommendation system due to its superior ability to understand contextual and semantic meaning in both reviews and metadata. The clustering model was retained for auxiliary grouping and exploratory purposes.

### 7. Future Enhancements
* User Viewing History and Ratings Integration: To further personalize the system, incorporating individual user behavior—such as viewing history, watch frequency, genre preferences, and explicit ratings—would allow the recommendation engine to adapt to personal taste and patterns. This would enable a hybrid recommendation system that blends collaborative filtering with the current content-based approach.
* Dynamic Sentiment Modeling: Update sentiment models to adapt over time or based on trending social media reactions to make the engine responsive to current audience moods.
* Multilingual and Multimedia Support: Extend sentiment and semantic models to handle multilingual reviews, and integrate other media types such as trailers, poster embeddings, or audio clips for more robust similarity assessment.
* Real-Time Feedback Loop: Allow users to rate the relevance of recommendations to train a feedback loop, improving accuracy over time.
This hybrid and sentiment-aware architecture creates a foundation for intelligent, scalable, and user-centric content discovery, paving the way for richer engagement and smarter entertainment experiences.



