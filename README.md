# Age-Gender-Estimation

<p> The "Age-Gender-Estimation" Django App is a robust and sophisticated Django web application designed to leverage on the power of machine learning to distinctively classify genders and ages of humans facial features. This app leverages the power of KNN and RF techniques to compare the analysis of these models in all of its metrics .</p>

<h4> Key Features </h4>
<ul>
	<li>Age Group Classification: Human ages can be grouped into various groups depending on their features.</li>
	<li>Gender Recognition: The application integrates with various data sources. This data is used to assess the techniques to know how fit it is to classify humans based on gender.</li>
</ul>

<h3>How to run this app</h3>
This project is to be run using Docker or in a virtual environment or best on Jupyter IDE. In case Dokcer is not available, the project can be done using the following steps:

<ol>
	<li>Create and activate a virtual environment and install the requirements:
		<code>pip install requirements.txt</code>
	</li>
	<li>Create a <code>.env </code>file and insert the following or set up a <code>django secret key</code>
		<ul>
			<li>SECRET_KEY=default-django-secret-key</li>
			<li>DEBUG=True</li>
		</ul>
	</li>
	<li>Run the loan_prediction_model.py:
		<code>python manage.py shell < loan_prediction_model.py</code>
	</li>
	<li>Run the django runserver command:
		<code>python manage.py runserver</code>
	</li>
</0l>
