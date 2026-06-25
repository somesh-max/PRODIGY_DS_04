import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv(r"C:\Users\somes\Documents\pridogy\task-4\social_media_sentiments.csv")

# Display first rows
print(df.head())

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(6, 5))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("sentiment_distribution.png")
plt.show()

# Generate word cloud
text = " ".join(df["Text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()