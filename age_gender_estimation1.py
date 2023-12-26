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


class FaceDataset(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = self._loadfile_names()
        self.shuffle_filenames = self._shuffle_filenames()
        self.load_data = self._load_data()
        self.df = self._create_dataframe()
        self.gender_dict = {0: "Male", 1: "Female"}

    def _loadfile_names(self):
        return list(map(lambda x: x.name, self.dataset_path.glob('*.jpg')))

    def _shuffle_filenames(self):
        return shuffle(self.filenames, random_state=42)

    def _load_data(self):
        data = []
        for filename in self.filenames:
            age, gender, image_path = filename.split('_')[:3]
            gender = int(gender)
            age = int(age)
            img_path = filename
            data.append({'IMAGES': img_path, 'AGE': age, 'GENDER': gender})
        return data

    def _create_dataframe(self):
        df = pd.DataFrame(self.load_data)
        age_grouping = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
        age_labels = [f"{i}-{i+4}" for i in range(0, 130, 5)]

        df['age_group'] = pd.cut(df['AGE'], bins=age_grouping, labels=age_labels, right=False)
        df.rename(columns={'age': 'AGE', 'gender': 'GENDER',
                           'age_group': 'AGE GROUP', 'image_path': 'IMAGES'}, inplace=True)

        df = df.astype({'AGE': 'float32', 'GENDER': 'int32'})
        return df

#     def visualize__an__image(self):
#         img = Image.open("/kaggle/input/utkface-new/UTKFace" + self.df.image_path[11])
#         plt.title("Age: " + str(self.df.age[11]) + ", Gender: " + self.gender_dict[self.df.gender[11]])
#         plt.imshow(img)

    def visualize_age_distribution(self):
        sns.displot(self.df["AGE"], kde=True, bins=20)
        plt.title("Age Distribution")
        plt.show()

    def visualize_age_group_distribution(self):
        sns.countplot(x='AGE GROUP', data=self.df, order=sorted(self.df['AGE GROUP'].unique()))
        plt.title("Age Group Distribution")
        plt.show()

    def plot_image_grid(self, row_no, cols_no):
        images = self.df['IMAGES'].values[:row_no * cols_no]
        labels = self.df[['AGE GROUP', 'GENDER']].head(row_no * cols_no)
        fig, axes = plt.subplots(row_no, cols_no, figsize=(20, 20))
        for i, ax in enumerate(axes.flat):
            i_path = os.path.join(self.dataset_path, images[i])
            img = Image.open(i_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Age Group: {labels['AGE GROUP'][i]}, GENDER: {'Male' if labels['GENDER'][i] == 0 else 'Female'}")
        plt.show()

    def split_data(self, test_size=0.2, random_state=42):
        train_input, test_input = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_input, test_input


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


class ModelEvaluator:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self):
        # Feature scaling for test set
        X_test_scaled = StandardScaler().fit_transform(self.x_test)

        # Predictions
        y_prediction = self.model.predict(X_test_scaled)

        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, y_prediction)
        print(f"Accuracy: {accuracy:.2f}")

        # Display classification report
        print("Classification Report:\n", classification_report(self.y_test, y_prediction))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_prediction))


def main_script():
    dataset = FaceDataset(Path('dataset/UTKFace/'))

    # Visualize age distribution
    dataset.visualize_age_distribution()

    # Visualize age group distribution
    dataset.visualize_age_group_distribution()

    # Plot image grid
    dataset.plot_image_grid(row_no=5, cols_no=5)

    # Split data
    train, test = dataset.split_data()

    # Gender recognition data
    X_train_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.IMAGES])
    X_test_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.IMAGES])

    # Age estimation data
    X_train_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.IMAGES])
    X_test_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.IMAGES])

    # Feature scaling for gender recognition
    X_train_scaled_gender = StandardScaler().fit_transform(X_train_images_gender)
    # X_test_scaled_gender = StandardScaler().fit(X_train_images_gender).transform(X_test_images_gender)

    # Feature scaling for age estimation
    X_train_scaled_age = StandardScaler().fit_transform(X_train_images_age)
    # X_test_scaled_age = StandardScaler().fit(X_train_images_age).transform(X_test_images_age)

    # KNN Model for gender recognition
    knn_model_gender = KNeighborsClassifier(n_neighbors=5)
    knn_model_gender.fit(X_train_scaled_gender, train['GENDER'])

    # KNN Model for age estimation
    knn_model_age = KNeighborsClassifier(n_neighbors=5)
    knn_model_age.fit(X_train_scaled_age, train['AGE GROUP'])

    # Random Forest Model for gender recognition
    rf_model_gender = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_gender.fit(X_train_scaled_gender, train['GENDER'])

    # Random Forest Model for age estimation
    rf_model_age = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_age.fit(X_train_scaled_age, train['AGE GROUP'])

    # Evaluate KNN Model for gender recognition
    knn_evaluator_gender = ModelEvaluator(knn_model_gender, X_test_images_gender, test['GENDER'])
    knn_evaluator_gender.evaluate_model()

    # Evaluate KNN Model for age estimation
    knn_evaluator_age = ModelEvaluator(knn_model_age, X_test_images_age, test['AGE GROUP'])
    knn_evaluator_age.evaluate_model()

    # Evaluate Random Forest Model for gender recognition
    rf_evaluator_gender = ModelEvaluator(rf_model_gender, X_test_images_gender, test['GENDER'])
    rf_evaluator_gender.evaluate_model()

    # Evaluate Random Forest Model for age estimation
    rf_evaluator_age = ModelEvaluator(rf_model_age, X_test_images_age, test['AGE GROUP'])
    rf_evaluator_age.evaluate_model()

    return dataset, knn_evaluator_gender, knn_evaluator_age, rf_evaluator_gender, rf_evaluator_age


if __name__ == '__main__':
    main_script()

