from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Accept customer reviews
reviews = []

n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input(f"Enter review {i + 1}: "))

# Convert reviews into numerical vectors
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(reviews)

# Apply LDA
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

# Display important keywords
words = vectorizer.get_feature_names_out()

print("\nTopics:")

for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)

    top_words = topic.argsort()[-5:][::-1]

    for j in top_words:
        print(words[j])

# Get topic distribution for each review
topic_distribution = lda.transform(X)

print("\nTopic Distribution:")

for i, distribution in enumerate(topic_distribution):
    print(f"Review {i + 1}: {distribution}")

# Apply t-SNE
# t-SNE needs at least 2 samples
if n >= 2:
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(5, n - 1)
    )

    X_tsne = tsne.fit_transform(topic_distribution)

    # Display t-SNE coordinates
    print("\nt-SNE Coordinates:")

    for i, point in enumerate(X_tsne):
        print(f"Review {i + 1} -> ({point[0]:.2f}, {point[1]:.2f})")

    # Visualize clusters
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        s=100
    )

    for i in range(n):
        plt.annotate(
            f"Review {i + 1}",
            (X_tsne[i, 0], X_tsne[i, 1])
        )

    plt.title("t-SNE Visualization of Customer Reviews")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

else:
    print("\nAt least 2 reviews are required for t-SNE.")