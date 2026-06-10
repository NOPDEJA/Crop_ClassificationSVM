import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import sys

tile = sys.argv[1]
year = sys.argv[2]
files = glob.glob(f"./merge_results/{year}/{tile}/classification_report_{tile}_*.csv")
data = []
accuracy_data = []

for file in files:
    # Extract year and month from filename
    match = re.search(r'classification_report_.*?_(\d{4})-(\d{2})\.csv', file)
    if match:
        year, month = match.groups()
        date = f"{year}-{month}"  # YYYY-MM format

        # Read CSV
        df = pd.read_csv(file, index_col=0)

        # Select only the relevant classes
        relevant_classes = ['1', '7']
        df_filtered = df.loc[df.index.isin(relevant_classes)]

        # Store data for each class
        for crop_class in relevant_classes:
            if crop_class in df_filtered.index:
                precision = df_filtered.loc[crop_class, 'precision']
                recall = df_filtered.loc[crop_class, 'recall']
                f1_score = df_filtered.loc[crop_class, 'f1-score']
                data.append([date, crop_class, precision, recall, f1_score])

        # Extract accuracy separately (it's a row in the CSV)
        if 'accuracy' in df.index:
            accuracy_value = df.loc['accuracy', 'precision']  # Accuracy is stored under "precision"
            accuracy_data.append([date, accuracy_value])

# Step 3: Create DataFrames and sort by date
df_metrics = pd.DataFrame(data, columns=['Date', 'Class', 'Precision', 'Recall', 'F1-Score'])
df_accuracy = pd.DataFrame(accuracy_data, columns=['Date', 'Accuracy'])

df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])
df_metrics.sort_values('Date', inplace=True)

df_accuracy['Date'] = pd.to_datetime(df_accuracy['Date'])
df_accuracy.sort_values('Date', inplace=True)

# Step 4: Plot Precision, Recall, and F1-Score for each class
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Classification Metrics Over Time')

metrics = ['Precision', 'Recall', 'F1-Score']
colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for each class
classes = df_metrics['Class'].unique()

for ax, metric in zip(axes, metrics):
    for i, crop_class in enumerate(classes):
        subset = df_metrics[df_metrics['Class'] == crop_class]
        ax.plot(subset['Date'], subset[metric], marker='o', label=f"Class {crop_class}", color=colors[i % len(colors)])

    ax.set_title(metric)
    ax.set_xlabel('Date')
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=-0, top=1)
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)  # Rotate labels
plt.savefig(f"./merge_results/{year}/{tile}/prf.png")

# Step 5: Plot Overall Accuracy Separately
plt.figure(figsize=(8, 4))
plt.plot(df_accuracy['Date'], df_accuracy['Accuracy'], 'k--', marker='o', label="Overall Accuracy")

plt.title("Overall Accuracy Over Time")
plt.xlabel("Date")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(bottom=-0, top=1)
plt.legend()
plt.grid(True)
plt.savefig(f"./merge_results/{year}/{tile}/acc.png")
plt.show()