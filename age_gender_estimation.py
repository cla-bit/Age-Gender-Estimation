import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


path = Path("dataset/UTKFace/")
filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

# Shuffle the filenames
filenames = shuffle(filenames, random_state=42)

# Define the age group
age_grouping = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
age_labels = [f"{i}-{i+4}" for i in range(0, 130, 5)]
print(age_labels)

# Extract age and gender from filenames
data = []
for filename in filenames:
    age, gender, image_path = filename.split('_')[:3]
    gender = int(gender)
    age = int(age)
    img_path = filename
    data.append({'image_path': img_path, 'age': age, 'gender': gender})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Create a new column 'age_group' in the DataFrame
df['age_group'] = pd.cut(df['age'], bins=age_grouping, labels=age_labels, right=False)

gender_dict = {0: "Male", 1: "Female"}

df = df.astype({'age': 'float32', 'gender': 'int32'})

img = Image.open("dataset/UTKFace/" + df.image_path[11])
plt.title("Age: " + str(df.age[11]) + ", Gender: " + gender_dict[df.gender[11]])
plt.imshow(img)

# Visualize age distribution
sns.displot(df["age"], kde=True, bins=20)
plt.title("Age Distribution")
plt.show()

# Visualize age group distribution
sns.countplot(x='age_group', data=df, order=sorted(df['age_group'].unique()))
plt.title("Age Group Distribution")
plt.show()


# Function to plot a grid of images
def plot_images(images, labels, rows_no, cols_no):
    fig, axes = plt.subplots(rows_no, cols_no, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        i_path = os.path.join(path, images[i])
        img_open = Image.open(i_path)
        ax.imshow(img_open)
        ax.axis('off')
        ax.set_title(f"Age: {labels['age_group'][i]}, Gender: {'Male' if labels['gender'][i] == 0 else 'Female'}")

    plt.show()


# Number of rows and columns in the grid
rows = 4
cols = 4

# Plot the images
plot_images(df['image_path'].values[:rows * cols], df[['age_group', 'gender']].head(rows * cols), rows, cols)


# Function to split the dataset into training and testing sets
def split_dataset(data_input, test_size=0.2, random_state=42):
    train_input, test_input = train_test_split(data_input, test_size=test_size, random_state=random_state)
    return train_input, test_input


# Split the dataset into training and testing sets
train, test = split_dataset(df)


# Function to convert images to NumPy arrays and flatten them
def images_to_arrays(image_paths, target_size=(64, 64)):
    X = []

    for image in image_paths:
        img_i = Image.open(image).convert("L")  # convert to grayscale
        img_i = img_i.resize(target_size, Image.LANCZOS)  # Resize with anti-aliasing
        img_array = np.array(img_i)
        flattened_array = img_array.flatten()
        X.append(flattened_array)

    return np.array(X)


# Gender recognition data
X_train_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.image_path])
X_test_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.image_path])

# Age estimation data
X_train_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.image_path])
X_test_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.image_path])

# Feature scaling for gender recognition
X_train_scaled_gender = StandardScaler().fit_transform(X_train_images_gender)
X_test_scaled_gender = StandardScaler().fit(X_train_images_gender).transform(X_test_images_gender)

# Feature scaling for age estimation
X_train_scaled_age = StandardScaler().fit_transform(X_train_images_age)
X_test_scaled_age = StandardScaler().fit(X_train_images_age).transform(X_test_images_age)


# KNN Model for gender recognition
knn_model_gender = KNeighborsClassifier(n_neighbors=5)
knn_model_gender.fit(X_train_scaled_gender, train['gender'])

# KNN Model for age estimation
knn_model_age = KNeighborsClassifier(n_neighbors=5)
knn_model_age.fit(X_train_scaled_age, train['age_group'])

# Random Forest Model for gender recognition
rf_model_gender = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_gender.fit(X_train_scaled_gender, train['gender'])

# Random Forest Model for age estimation
rf_model_age = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_age.fit(X_train_scaled_age, train['age_group'])


# Function to evaluate the model performance
def evaluate_model(model, x_test_sample, y_test_sample):
    # Feature scaling for test set
    X_test_scaled = StandardScaler().fit_transform(x_test_sample)

    # Predictions
    y_prediction = model.predict(X_test_scaled)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test_sample, y_prediction)
    print(f"Accuracy: {accuracy:.2f}")

    # Display classification report
    print("Classification Report:\n", classification_report(y_test_sample, y_prediction))
    print("Confusion Matrix:\n", confusion_matrix(y_test_sample, y_prediction))


# Evaluate models for gender recognition
print("\nKNN Model Gender Evaluation:")
evaluate_model(knn_model_gender, X_test_images_gender, test['gender'])

print("\nKNN Model Age Estimation Evaluation:")
evaluate_model(knn_model_age, X_test_images_age, test['age_group'])

print("\nRandom Forest Model Gender Evaluation:")
evaluate_model(rf_model_gender, X_test_images_gender, test['gender'])

print("\nRandom Forest Model Age Estimation Evaluation:")
evaluate_model(rf_model_age, X_test_images_age, test['age_group'])


