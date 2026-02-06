import pandas as pd

from preprocessing import clean_text
from sentiment import get_sentiment_score, score_to_label

from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/GrammarandProductReviews.csv"


# ---------------- DATA LOADING ----------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    print("\n✅ Dataset Loaded")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample Data:")
    print(df.head(3))

    return df


# ---------------- BASIC EDA ----------------

def basic_eda(df):

    print("\n📊 Missing Values:")
    print(df.isnull().sum())

    print("\n📊 Data Types:")
    print(df.dtypes)

    print("\n📊 Duplicate Rows:", df.duplicated().sum())


# ---------------- TEXT PREP ----------------

def prepare_text_data(df):

    print("\n🧹 Preparing Text Data...")

    df = df.dropna(subset=["reviews.text"])

    df["clean_review"] = df["reviews.text"].apply(clean_text)

    print("\n✅ Text Cleaning Done")
    print(df[["reviews.text", "clean_review"]].head(3))

    return df


# ---------------- SENTIMENT GENERATION ----------------

def generate_sentiment(df):

    print("\n🧠 Generating Sentiment Scores...")

    df["sentiment_score"] = df["clean_review"].apply(get_sentiment_score)
    df["predicted_sentiment"] = df["sentiment_score"].apply(score_to_label)

    print("\n✅ Sentiment Generation Done")
    print(df[["clean_review", "sentiment_score", "predicted_sentiment"]].head(5))

    return df


# ---------------- TRUE SENTIMENT FROM RATING ----------------

def rating_to_sentiment(rating):

    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"


def add_true_sentiment(df):

    df["true_sentiment"] = df["reviews.rating"].apply(rating_to_sentiment)

    print("\n⭐ True Sentiment Added")
    print(df[["reviews.rating", "true_sentiment"]].head(5))

    return df


# ---------------- MODEL EVALUATION ----------------

def evaluate_sentiment_model(df):

    print("\n📊 Evaluating Sentiment Model...")

    y_true = df["true_sentiment"]
    y_pred = df["predicted_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n✅ Accuracy: {accuracy:.4f}")

    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred))


# ---------------- BRAND AFFINITY ----------------

def calculate_brand_affinity(df):

    print("\n🏷 Calculating Brand Affinity...")

    brand_stats = df.groupby("brand").agg({
        "sentiment_score": "mean",
        "clean_review": "count"
    }).reset_index()

    brand_stats.rename(columns={
        "clean_review": "review_count"
    }, inplace=True)

    brand_stats["review_weight"] = (
        brand_stats["review_count"] / brand_stats["review_count"].max()
    )

    brand_stats["affinity_score"] = (
        0.7 * brand_stats["sentiment_score"] +
        0.3 * brand_stats["review_weight"]
    )

    brand_stats = brand_stats.sort_values(
        by="affinity_score",
        ascending=False
    )

    print("\n✅ Top 10 Brands by Affinity:")
    print(brand_stats.head(10))

    return brand_stats


# ---------------- RECOMMENDATION ENGINE ----------------

def recommend_brands(category_name, brand_affinity_df, df):

    print(f"\n🎯 Recommendations for Category: {category_name}")

    category_brands = df[
        df["categories"].str.contains(category_name, case=False, na=False)
    ]["brand"].unique()

    recommendations = brand_affinity_df[
        brand_affinity_df["brand"].isin(category_brands)
    ].head(5)

    print("\n✅ Recommended Brands:")
    print(recommendations[["brand", "affinity_score"]])

    return recommendations


# ---------------- SAVE OUTPUTS ----------------

def save_outputs(df, brand_affinity_df):

    print("\n💾 Saving Outputs...")

    df.to_csv("data/processed_reviews.csv", index=False)
    brand_affinity_df.to_csv("data/brand_affinity_scores.csv", index=False)

    print("✅ Files Saved")


# ---------------- MAIN PIPELINE ----------------

if __name__ == "__main__":

    df = load_data()
    basic_eda(df)

    df = prepare_text_data(df)
    df = generate_sentiment(df)
    df = add_true_sentiment(df)

    evaluate_sentiment_model(df)

    brand_affinity_df = calculate_brand_affinity(df)

    recommend_brands(
        category_name="Food",
        brand_affinity_df=brand_affinity_df,
        df=df
    )

    save_outputs(df, brand_affinity_df)

    print("\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
