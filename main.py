from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = int(request.form.get('beds'))
        bathrooms = float(request.form.get('baths'))
        size = float(request.form.get('size'))
        zipcode = int(request.form.get('zip_code'))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                                  columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(list(unknown_categories), data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Predict the price
        prediction = pipe.predict(input_data)[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
