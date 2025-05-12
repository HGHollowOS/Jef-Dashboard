try:
    from textblob import TextBlob
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    TextBlob = None
    WordCloud = None
    plt = None
    pd = None

def check_dependencies():
    if TextBlob is None:
        raise ImportError("NLP features require textblob and wordcloud. Please install them first.")

def sentiment_score(text):
    if not isinstance(text, str) or text == "NaN":
        return 0
    return TextBlob(text).sentiment.polarity

def wordcloud_for_high_pain(df: pd.DataFrame):
    texts = df.loc[df['Vinger_pijn_stijfheid'] > 3, 'Subjectieve_notes']
    texts = texts[texts.apply(lambda x: isinstance(x, str) and x != "NaN")]
    text = ' '.join(texts)
    if not text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
