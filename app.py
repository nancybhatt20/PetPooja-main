from flask import Flask, render_template
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

app = Flask(__name__)

# Function to generate graph
def generate_graph():
    csv_filename = "human_detection_results.csv"
    
    # Load data
    df = pd.read_csv(csv_filename)

    # Filter valid data
    df = df[df["Detected People Count"].apply(lambda x: str(x).isdigit())]
    df["Detected People Count"] = df["Detected People Count"].astype(int)

    # Aggregate counts by weekday
    weekday_counts = df.groupby("Weekday")["Detected People Count"].mean()

    # Create the graph
    plt.figure(figsize=(10, 5))
    plt.bar(weekday_counts.index, weekday_counts.values, color='blue', alpha=0.6)
    plt.xlabel("Weekday")
    plt.ylabel("Average People Count")
    plt.title("People Count Per Weekday")

    # Save graph to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{graph_url}"

@app.route("/")
def home():
    graph_url = generate_graph()
    return render_template("index.html", graph_url=graph_url)

if __name__ == "__main__":
    app.run(debug=True)
