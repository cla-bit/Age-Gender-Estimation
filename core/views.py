from django.shortcuts import render
from django.views.generic import View
from django.conf import settings

from age_gender_estimation1 import FaceDataset, images_to_arrays, StandardScaler, KNeighborsClassifier, RandomForestClassifier, ModelEvaluator

# Create your views here.


class FaceDatasetView(View):
    def get(self, request, *args, **kwargs):
        dataset = FaceDataset(settings.UTKFACE_PATH)

        # Split data
        train, test = dataset.split_data()
        # Gender recognition data
        X_train_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.image_path])
        X_test_images_gender = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.image_path])

        # Age estimation data
        X_train_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in train.image_path])
        X_test_images_age = images_to_arrays(['dataset/UTKFace/' + filename for filename in test.image_path])

        # Feature scaling for gender recognition
        X_train_scaled_gender = StandardScaler().fit_transform(X_train_images_gender)
        # X_test_scaled_gender = StandardScaler().fit(X_train_images_gender).transform(X_test_images_gender)

        # Feature scaling for age estimation
        X_train_scaled_age = StandardScaler().fit_transform(X_train_images_age)
        # X_test_scaled_age = StandardScaler().fit(X_train_images_age).transform(X_test_images_age)

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

        # Evaluate KNN Model for gender recognition
        knn_evaluator_gender = ModelEvaluator(knn_model_gender, X_test_images_gender, test['gender'])
        knn_evaluator_gender.evaluate_model()

        # Evaluate KNN Model for age estimation
        knn_evaluator_age = ModelEvaluator(knn_model_age, X_test_images_age, test['age_group'])
        knn_evaluator_age.evaluate_model()

        # Evaluate Random Forest Model for gender recognition
        rf_evaluator_gender = ModelEvaluator(rf_model_gender, X_test_images_gender, test['gender'])
        rf_evaluator_gender.evaluate_model()

        # Evaluate Random Forest Model for age estimation
        rf_evaluator_age = ModelEvaluator(rf_model_age, X_test_images_age, test['age_group'])
        rf_evaluator_age.evaluate_model()

        context = {
            'age_distribution': dataset.visualize_age_distribution(),
            'age_group_distribution': dataset.visualize_age_group_distribution(),
            'image_grid': dataset.plot_image_grid(row_no=5, cols_no=5),
            'knn_evaluator_gender': knn_evaluator_gender.evaluate_model(),
            'knn_evaluator_age': knn_evaluator_age.evaluate_model(),
            'rf_evaluator_gender': rf_evaluator_gender.evaluate_model(),
            'rf_evaluator_age': rf_evaluator_age.evaluate_model()
        }
        return render(request, 'core/face_dataset.html', context)


