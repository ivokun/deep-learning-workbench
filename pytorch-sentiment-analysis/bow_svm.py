import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bow_vectorizer = CountVectorizer(binary=False)

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br>\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
  reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
  reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

  return reviews

reviews_train = []

for line in open('datasets/aclImdb/movie_data/full_train.txt', 'r', encoding="utf-8"):
  reviews_train.append(line.strip())

reviews_test = []

for line in open('datasets/aclImdb/movie_data/full_test.txt', 'r', encoding="utf-8"):
  reviews_test.append(line.strip())

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

bow_vectorizer.fit(reviews_train_clean)
X = bow_vectorizer.transform(reviews_train_clean)
X_test = bow_vectorizer.transform(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
  svm = LinearSVC(C=c)
  svm.fit(X_train, y_train)
  print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, svm.predict(X_val))))

final_model = LinearSVC(C=0.01)
final_model.fit(X, target)
print("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X_test)))
