# Summarizing-artical
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(article_text, max_length=130, min_length=30):
    """
    Summarizes the given article using a transformer model.
    """
    summary = summarizer(article_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Sample long article (replace this with any lengthy text)
article = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more. The developments in AI are happening at a rapid pace, and this technology is being integrated into numerous fields such as healthcare, finance, education, and transportation. There is a growing debate around the ethical implications of AI, especially concerning data privacy, decision transparency, and the future of human employment.
"""

# Generate and print the summary
summary = summarize_article(article)
print("Original Article:\n", article)
print("\n---\nSummary:\n", summary)
