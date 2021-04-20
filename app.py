import streamlit as st
from PIL import Image
import pytesseract as pt
from textblob import TextBlob
import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import base64
import pandas as pd
import time
from pdf2image import convert_from_path
import pdfplumber
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer


def intro():
    st.markdown("""<h2 style='text-align: left; color: #0e1236;'>Problem Statement</h2>
    <p style='color: #0e1236;'>Plagiarism is defined as taking the work of others as your own without credit. This definition applies to source-lacking essays and simply claiming ideas as your own. 
    One example of plagiarism is like taking the idea of the theory of relativity as your own, or claiming you wrote a book while it was actually made by another author.</p>""",
                unsafe_allow_html=True)
    st.markdown("""<h2 style='text-align: left; color: #0e1236;'>TASK</h2><p style='color: #0e1236;'>
    Need to Input some files in raw format and then check plagiarism among them</p>""",
                unsafe_allow_html=True)
    st.markdown("""<h2 style='text-align: left; color: #0e1236;'>My Approach</h2>
    <p style='color: #0e1236;'>
    <ul style='text-align: left; color: #0e1236;'>
    <li>Create a web Interface using Python</li>
    <li>Take some input files</li>
    <li>Extract Text from them using some of the libraries like docx2txt pdfplumber tesseract etc.</li>
    <li>Compare them using a NLP(Natural Language Processig) or LCS(longest comen Subsequence)</li>
    <li>Classify texts as:<ul><li>Positive if plagiarism is less than 30%,<li>Negative if plagiarism is greater than 75%,<li>Random otherwise</li></ul>
    <li>I am using Streamlit as Backend In this Project Which is an open source app framework</li>
    </ul>
    </p>""",
                unsafe_allow_html=True)
    st.markdown("""<h3 style='text-align: left; color: #0e1236;'>Learn More</h3>""",
                unsafe_allow_html=True)
    if(st.checkbox("What is Tesseract")):
        st.markdown(
            """<div style='background-color: #FEFEFE;padding-left: 50px;border-radius: 10px;'>
# from textblob import TextBlob
            <h3 style='text-align: left; color: #0e1236;'>An optical character recognition (OCR) engine</h3>
            <p style='text-align: left; color: #0e1236;'>Tesseract is an OCR engine with support for unicode and 
            the ability to recognize more than 100 languages out of the box. It can be trained to recognize other languages.</p>
            <p style='text-align: left; color: #0e1236;'>Tesseract is used for text detection on mobile devices, in video, and 
            in Gmail image spam detection.</p>
            <a href = "https://tesseract-ocr.github.io/" style='text-align: left; color:#7870e0;'>
                Tesseract Documentation
                </a >
            </div><p></p>""", unsafe_allow_html=True)
    if(st.checkbox("Lowest commen Subsequence")):
        st.markdown(
            """<div style='background-color: #FEFEFE;border-radius: 10px;padding-left: 50px;'>
            <p style='text-align: left; color: #0e1236;'><span style='font-style: italic;'>LCS Problem Statement:</span> 
            Given two sequences, find the length of longest subsequence present in both of them. A subsequence is a sequence 
            that appears in the same relative order, but not necessarily contiguous. For example, “abc”, “abg”, “bdf”, “aeg”, ‘”acefg”, .. etc are subsequences of “abcdefg”.</p>
            <p style='text-align: left; color: #0e1236;'>We are using Dynamic programming for this.</p>
            <a href = "https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/" style='text-align: left; color:#7870e0;'>
                Geeks For Geeks Link
                </a >
            </div><p></p>""",
            unsafe_allow_html=True)
        if(st.checkbox("Show Code")):
            st.markdown(
                """<h2 style='text-align: left; color: #0e1236;'>Lowest Commen Subsequence Function</h2>""",
                unsafe_allow_html=True)

            with st.echo():
                def lowestcomensubsequence(X, Y):
                    m = len(X)
                    n = len(Y)

                    L = [[None]*(n+1) for i in range(m+1)]

                    for i in range(m+1):
                        for j in range(n+1):
                            if i == 0 or j == 0:
                                L[i][j] = 0
                            elif X[i-1] == Y[j-1]:
                                L[i][j] = L[i-1][j-1]+1
                            else:
                                L[i][j] = max(L[i-1][j], L[i][j-1])

                    return L[m][n]


def Extract(fil):
    text = ""
    if fil.type == "text/plain":
        text = str(fil.read(), "utf-8")

    elif fil.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(fil)

    elif fil.type == "application/pdf":
        with pdfplumber.open(fil) as pdf:
            for j in range(0, len(pdf.pages)):
                page = pdf.pages[j]
                text += page.extract_text()
    else:
        file_bytes = np.asarray(bytearray(fil.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        text = pt.image_to_string(img)
    text = text.replace('\n', ' ').replace('\r', '')
    return text


def lcs(X, Y):
    m = len(X)
    n = len(Y)

    L = [[None]*(n+1) for i in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]*100/max(m, n)


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}" style="color:#FFF;text-decoration:none;">{download_link_text}</a>'


def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            if values[i][j] < 0.5:
                text = ax.text(j, i, "%.2f" % values[i, j],
                               ha="center", va="center", color="w", fontsize=6)
            else:
                text = ax.text(j, i, "%.2f" % values[i, j],
                               ha="center", va="center", color="b", fontsize=6)
    st.pyplot(fig)


def OTM():
    MAJOR = st.sidebar.file_uploader("MAJOR FILE", type=["pdf", "txt", "docx"])
    ALL = st.sidebar.file_uploader("ALL FILE", type=["pdf", "txt", "docx"],
                                   accept_multiple_files=True)

    if MAJOR is not None and ALL is not None:
        key_val = {}
        for i in range(len(ALL)):
            key_val[ALL[i].name] = i
        corpus = [Extract(ALL[key_val[j]]) for j in sorted(key_val.keys())]
        corpus.append(Extract(MAJOR))
        names = [ALL[key_val[j]].name for j in sorted(key_val.keys())]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(corpus)
        pairwise_similarity = tfidf * tfidf.T
        pairwise_similarity = tfidf * tfidf.T
        pairwise_similarity = pairwise_similarity.toarray()
        l = len(ALL)
        # pairwise_similarity = a * b.T

        # print(a.toarray()[0])
        # print(b.toarray())
        # text_major = Extract(MAJOR)
        # key_val = {}
        arr = np.zeros((len(ALL), 1))
        # i = int(0)
        # for fil in ALL:
        #     text = Extract(fil)
        #     key_val[fil.name] = lcs(text_major, text)

        for i in range(l):
            arr[i][0] = pairwise_similarity[l][i]
        chart_data = pd.DataFrame(
            arr,
            columns=["Plagiarism"])
        st.bar_chart(chart_data)
        tmp_download_link = download_link(
            chart_data, 'extracted_text.csv', 'Download as csv')
        st.markdown(f"""<h3 style='
            border-radius: 10px;color: #FFF;background-color:#eb34c9;text-align: center;'>{tmp_download_link}</h3>""", unsafe_allow_html=True)


def MTM():
    ALL = None
    ALL = st.sidebar.file_uploader("ALL FILE's", type=["pdf", "txt", "docx"],
                                   accept_multiple_files=True)
    if ALL:
        key_val = {}
        for i in range(len(ALL)):
            key_val[ALL[i].name] = i
        corpus = [Extract(ALL[key_val[j]]) for j in sorted(key_val.keys())]
        names = [ALL[key_val[j]].name for j in sorted(key_val.keys())]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(corpus)
        pairwise_similarity = tfidf * tfidf.T
        heatmap(names, names, pairwise_similarity.toarray())
        toprint = pd.DataFrame(pairwise_similarity.toarray())
        tmp_download_link = download_link(
            toprint, 'pairwise_similarity.csv', 'Download as csv')
        st.markdown(f"""<h3 style='
            border-radius: 10px;color: #FFF;background-color:#eb34c9;text-align: center;'>{tmp_download_link}</h3>""", unsafe_allow_html=True)


def OTO():
    FILE_1 = st.sidebar.file_uploader(
        "FILE.1", type=["pdf", "txt", "docx", "jpg", "png", "jpeg"])
    FILE_2 = st.sidebar.file_uploader(
        "FILE.2", type=["pdf", "txt", "docx", "jpg", "png", "jpeg"])
    cols0 = st.beta_columns(2)
    if FILE_1 and FILE_2:
        texts = [None, None]
        texts[0] = Extract(FILE_1)
        texts[1] = Extract(FILE_2)
        cols0[0].markdown(f"<pre style='background-color: #FEFEFE;border-radius: 10px;'>{texts[0]}</pre>",
                          unsafe_allow_html=True)
        cols0[1].markdown(f"<pre style='background-color: #FEFEFE;border-radius: 10px;'>{texts[1]}</pre>",
                          unsafe_allow_html=True)
        percentage = round(lcs(texts[0], texts[1]), 2)
        k = 2
        if percentage > 75:
            k = 1
        if percentage < 45:
            k = 0
        percentage = str(percentage)
        if k == 0:
            st.markdown(
                f"""<div style='background-color: #a0f76a;border-radius: 10px;'>
            <h1 style='color:#0e1236;text-align: center;'>Plagiarism: 
            <span style='color:#0e1236;'>{percentage}%</span></h1>
            <h2 style='color:#0e1236;text-align: center;'>These Docs Are Not Copied!</h2>
            </div>
            """, unsafe_allow_html=True)
        if k == 1:
            st.markdown(
                f"""<div style='background-color: #f5af98;border-radius: 10px;'>
            <h1 style='color:#0e1236;text-align: center;'>Plagiarism: 
            <span style='color:#0e1236;'>{percentage}%</span></h1>
            <h2 style='color:#0e1236;text-align: center;'>These Docs Are Copied!</h2>
            </div>
            """, unsafe_allow_html=True)
        if k == 2:
            st.markdown(
                f"""<div style='background-color: #cceaf0;border-radius: 10px;'>
            <h1 style='color:#0e1236;text-align: center;'>Plagiarism: 
            <span style='color:#0e1236;'>{percentage}%</span></h1>
            <h2 style='color:#0e1236;text-align: center;'>These Docs Are somewhat similer!</h2>
            </div>
            """, unsafe_allow_html=True)


def main():

    # Defining Title
    st.markdown(
        "<h1 style='text-align: center;border-radius: 10px;color: #FFF;background-color:#eb34c9;padding-top:30px;padding-bottom:30px;font-size:60px;'>Find Plagiarism</h1>", unsafe_allow_html=True)
    st.markdown(
        """<style>body {background-color: #e6e8dc;margin-left:0;}</style><body></body>""", unsafe_allow_html=True)
    Comparison_Type = ['intro', 'One-to-Many', 'Many-to-Many', 'One-to-One']
    Comparison = st.selectbox("Select Comparison Type", Comparison_Type)
    if Comparison == Comparison_Type[0]:
        intro()
    if Comparison == Comparison_Type[1]:
        OTM()
    elif Comparison == Comparison_Type[2]:
        MTM()
    else:
        OTO()


main()
