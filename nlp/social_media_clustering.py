from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Accept social media posts
posts = []

n = int(input("Enter number of posts: "))

for i in range(n):
    post = input(f"Enter post {i + 1}: ")
    posts.append(post)

# Number of clusters
k = int(input("Enter number of clusters: "))

# TF-IDF with unigrams and bigrams
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

# Apply K-Means
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_

# Display clustered posts
print("\n==============================")
print("CLUSTER RESULTS")
print("==============================\n")

for i in range(len(posts)):
    print("Post:", posts[i])
    print("Cluster:", labels[i])
    print()

# Get feature names
terms = vectorizer.get_feature_names_out()

# Display important keywords
print("==============================")
print("IMPORTANT KEYWORDS")
print("==============================\n")

for i in range(k):

    center = model.cluster_centers_[i]

    top = center.argsort()[-5:][::-1]

    print("Cluster", i)

    for j in top:
        print("-", terms[j])

    print()

# Marketing insights
print("==============================")
print("MARKETING INSIGHTS")
print("==============================")

print("Similar customer opinions are grouped together.")
print("Clusters help identify product trends and issues.")
print("Important keywords show the major topics discussed by customers.")