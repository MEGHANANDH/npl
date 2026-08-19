from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

docs = []
labels = []

# Accept legal documents
n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))
    labels.append(input(f"Enter category {i + 1}: "))

# -------------------------------
# Rule-Based Classification
# -------------------------------

rule_pred = []

for doc in docs:
    doc_lower = doc.lower()

    if "contract" in doc_lower:
        rule_pred.append("contract")

    elif "judgment" in doc_lower:
        rule_pred.append("judgment")

    else:
        rule_pred.append("agreement")

# Calculate Rule-Based accuracy
rule_acc = accuracy_score(labels, rule_pred)

# -------------------------------
# Maximum Entropy Classifier
# -------------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)

ml_pred = model.predict(X)

# Calculate MaxEnt accuracy
ml_acc = accuracy_score(labels, ml_pred)

# -------------------------------
# Display Results
# -------------------------------

print("\n--------------------------------")
print("Classification Results")
print("--------------------------------")

print("Actual Categories :", labels)
print("Rule Predictions  :", rule_pred)
print("MaxEnt Predictions:", list(ml_pred))

print("\nRule-Based Accuracy:", rule_acc)
print("Maximum Entropy Accuracy:", ml_acc)

print("\n--------------------------------")
print("Comparison")
print("--------------------------------")

if rule_acc > ml_acc:
    print("Rule-Based Classifier performed better.")

elif ml_acc > rule_acc:
    print("Maximum Entropy Classifier performed better.")

else:
    print("Both classifiers have the same accuracy.")