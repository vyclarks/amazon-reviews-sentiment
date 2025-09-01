from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from text_preprocessing import preprocess_reviews

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Use non-GUI backend for Matplotlib
import joblib
import base64


app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    app.logger.info("GET request received on /test")
    return jsonify({"message": "Flask is working!"}), 200

# Route for rendering the frontend
@app.route("/")
def home():
    return render_template("index.html")  # Render the index.html file from the templates directory


@app.route("/predict", methods=["POST"])
def predict():

    # Select the predictor to be loaded from Models folder
    trained_model = joblib.load(r"models/lightgbm_model.pkl")
  
    try:
        # The request contains a file
        if "file" in request.files:
            file = request.files["file"]
            app.logger.info("Processing file upload: ", file.filename, file.content_type)

            # Ensure the file has a valid extension (optional)
            if file.content_type != "text/csv":
                app.logger.info("Invalid file type:", file.content_type)
                return jsonify({"error": "Invalid file type. Only CSV files are supported."}), 400
            
            # Multiple predictions from csv file
            try:
                df = pd.read_csv(file)
                app.logger.info("File successfully loaded. Columns: %s", df.columns.tolist())

            except Exception as e:
                app.logger.info("Error reading CSV file:", str(e))
                return jsonify({"error": "Unable to process the uploaded file. Ensure it is a valid CSV."}), 400
 
            if 'Review' not in df.columns:
                app.logger.error("Column 'Review' not found in uploaded file.")
                return jsonify({"error": "Column 'Review' not found in file"}), 400
        
            predictions, graph = multiple_prediction(df, trained_model)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            app.logger.info("Predictions successfully generated for file.")
            return response

        # The request contains a single review
        elif "text" in request.json:
            print(f"Received text: {request.json['text']}")

            # Single review prediction
            review = request.json.get("text", "")
            app.logger.info("Processing single text prediction: %s", review)

            if not review:
                app.logger.error("No text provided in the request.")
                return jsonify({"error": "No text provided"}), 400
        
            predicted_sentiment = single_prediction(review, trained_model)
            app.logger.info("Prediction result: %s", predicted_sentiment)

            return jsonify({"prediction": predicted_sentiment}), 200
        
        else:
            app.logger.error("Invalid request. No file or text input detected.")
            return jsonify({"error": "Invalid request. Please provide 'text' or upload a 'file'."}), 400

    except Exception as e:
        app.logger.error("Error occurred: %s", str(e))
        return jsonify({"error": str(e)})


def single_prediction(review, predictor):
    """
    Perform prediction on a single review and return results as string label: "Positive", "Neutral", or "Negative"

    Parameters:
        df_data (pd.DataFrame): DataFrame containing the reviews to predict.
        predictor: The trained prediction model.

    Returns:
        label: prediction label
    """
    app.logger.info("Preprocessing single review: %s", review)

    df = pd.DataFrame({'Review': [review]})

    # Preprocess and transform the single review
    tfidf_transformed = preprocess_reviews(df)

    # Make prediction
    prediction = predictor.predict(tfidf_transformed)[0]  # Get the first (and only) prediction

    app.logger.info("Raw prediction value: %s", prediction)

    # Map prediction to label
    # label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    # label = label_mapping.get(int(prediction), "Unknown")
    # app.logger.info("Mapped prediction label: %s", label)

    return prediction


def multiple_prediction(df_data, predictor):
    """
    Perform predictions on multiple reviews and return results as an CSV file.

    Parameters:
        df_data (pd.DataFrame): DataFrame containing the reviews to predict.
        predictor: The trained prediction model.

    Returns:
        Flask Response: An CSV file containing the predictions.
    """
    
    app.logger.info("Preprocessing multiple reviews. Number of rows: %d", len(df_data))

    # Preprocess and transform the reviews
    tfidf_transformed= preprocess_reviews(df_data)

    # Predict with the model
    predictions = predictor.predict(tfidf_transformed)

    app.logger.info("Raw predictions: %s", predictions)
    df_data['Prediction'] = predictions

    app.logger.info("Mapped predictions successfully added to DataFrame.")

    # Save the DataFrame to a CSV in memory
    output = BytesIO()
    df_data.to_csv(output, index=False)
    output.seek(0)

    app.logger.info("CSV file successfully generated.")

    graph = get_distribution_graph(df_data)
    
    return output, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ["red", "yellow", "green"] 
    wp = {"linewidth": 1, "edgecolor": "black"}

    # Count unique predictions
    tags = data["Prediction"].value_counts()

    # Dynamically set explode to match the number of categories
    explode = [0.01] * len(tags)  # Small offset for each segment

    # Create the pie chart
    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors[:len(tags)],  # Match the number of segments
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    # Save the plot to a BytesIO object
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

if __name__ == "__main__":
    app.run(port=5000, debug=True)
