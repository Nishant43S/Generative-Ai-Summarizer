import streamlit as st   ### importing liberaries
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
import streamlit.components.v1 as component
from streamlit_lottie import st_lottie, st_lottie_spinner
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
from newspaper import Article
import nltk
import nltk.downloader
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from cleantext import clean
from PyPDF2 import PdfReader
import pdfminer
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
import requests
import json
import numpy as np
import pandas as pd
import random
import base64
import lxml
import lxml_html_clean
import re
import os


###### main app functions

### insert external css
def insert_css(css_file:str):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

### insert external html file
def insert_html(html_file):
    with open(html_file) as f:
        return f.read()

### insert lottie animation json files
def insert_lottie_animation(animation_file:str):
    with open(animation_file, "r") as f:
        return json.load(f)

### app tutorial video function
@st.dialog("App Tutorial")
def watch_tutorial():
    st.subheader("GenAi Summarizer🤖")
    video_file = open("app_tutorial.mp4", "rb")
    video_bytes = video_file.read()
    st.text("")
    st.video(
        data=video_bytes,format="video/mp4",
        loop=True,autoplay=True
    )


def download_text(text, filename):
    """
    download article text 
    in document format
    """
    #### Convert string to bytes
    b64 = base64.b64encode(text.encode()).decode()

    href = f"""
            <a href="data:application/octet-stream;base64,{b64}" download="{filename}">
                <button class="neon-button">Download</button>
            </a>
            """
    
    st.markdown(href, unsafe_allow_html=True)
    if __name__=="__main__":
        insert_css("cssfiles/download-article.css")


def copy_text(text):
    html_code = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
             <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css" integrity="sha512-Kc323vGBEqzTmouAECnVceyQqyqdsSiqLQISBL29aUW4U/M7pSPA/gEUZQqv1cwx4OnYxTxve5UMg5GT6L4JJg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
            <style>
                *{{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                .copy-button{{
                    font-size: 24px;
                    cursor: pointer;
                    color: #5b70f3;
                    transition: 0.3s ease-in-out;
                }}
            </style>
        </head>
        <body>
            <a class="copy-button" onclick="copyText()">
                <i class="fa-solid fa-copy"></i>
            </a>
            <br>
            <br>
            <p id="textToCopy">{text}</p>

            <script>
                function copyText() {{
                    // Get the text from the <p> tag
                    const text = document.getElementById('textToCopy').innerText;

                    // Create a temporary <textarea> element
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    document.body.appendChild(textarea);

                    // Select the text in the <textarea>
                    textarea.select();

                    // Execute the copy command
                    document.execCommand('copy');

                    // Remove the <textarea> element from the DOM
                    document.body.removeChild(textarea);

                    alert('Text copied');
                }}
            </script>
        </body>
        </html>

    """

    component.html(html_code,height=28)


### copy and download button
def Copy_download_button(article_text,article_format,article_file_name):
    try:
                ### column for copy and download article
        Copy_btn_col,download_btn_col, blank_col_copy1, blank_col_copy2= st.columns([1,3,5,5],gap="small")

        with blank_col_copy1:
            st.text("")
        with blank_col_copy1:
            st.text("")
                
        with Copy_btn_col:
            copy_text(article_text)

        with download_btn_col:
            download_text(text=article_format,filename=article_file_name)
    except Exception as e:
        st.warning("Something went wrong...",e,icon="⚠️")


### setting page layout
st.set_page_config( 
    page_title="GenAi Summarizer",
    page_icon="🤗",
    initial_sidebar_state="collapsed",
    layout="wide"
)


#### app settings css
if __name__=="__main__":
    insert_css("cssfiles/app.css")   


### huging face modals
Hugingface_modals = {
    "google-pegasus":"google/pegasus-xsum",
    "facebook-bart":"facebook/bart-large-cnn",
    "t5-base":"t5-base"
}


### summarization modal
def Hugingface_summarization_modal(summary_text,modal_name,maximum_length):
    """
    it is an text summarization modal
    it use hugingface modals for summarization task.
    it generates summarized text output
    """
    def summarization_modal_name(modal)->str:
        if modal == "google-pegasus":
            return "google/pegasus-xsum"
        elif modal == "facebook-bart":
            return "facebook/bart-large-cnn"
        elif modal == "t5-base":
            return "t5-base"
    try:
        use_modal = summarization_modal_name(modal_name)  ### modal name

        auto_tokenizer = AutoTokenizer.from_pretrained(use_modal) ### using autokenizer for pretrained modal
        auto_modal = AutoModelForSeq2SeqLM.from_pretrained(use_modal)

        ### creating pipeline
        summarizer = pipeline("summarization",model=auto_modal,tokenizer=auto_tokenizer)

        summarizer_text = summary_text

        summary_generate = summarizer( ### summarizer
            summarizer_text,max_length=maximum_length+20,
            min_length=maximum_length,
            do_sample=False
        ) 

        return summary_generate[0]['summary_text']
            
    except Exception as e:
        st.warning("Something went wrong...\n\n",e,icon="⚠️")




### displaying modals
@st.cache_data
def Modal_Level(modal_text):
    if modal_text == "google-pegasus":
        st.markdown(
            f"""
                <div class="google-modal">
                <span style="font-size: 17px; color: #fff;">
                    Maodal-
                </span>
                    google/pegasus-xsum
                </div>
            """,unsafe_allow_html=True
        )

    elif modal_text == "facebook-bart":
        st.markdown(
            f"""<div class="facebook-modal">
                <span style="font-size: 17px; color: #fff;">
                    Maodal-
                </span>
                    facebook/bart-large-cnn
                </div>
            """,unsafe_allow_html=True
        )

    elif modal_text == "t5-base":
        st.markdown(
            f"""<div class="t5-modal">
                <span style="font-size: 17px; color: #fff;">
                    Maodal-
                </span>
                    t5-base
                </div>
            """,unsafe_allow_html=True
        )
    if __name__=="__main__":
        insert_css("cssfiles/modal.css")



#### creating sidebar
app_sidebar = st.sidebar

with app_sidebar:
    st.text("")
    st.subheader("GenAi Summarizer🤖")
    st.write("Developer: **Nishant Maity**")
    st.text("")
    st.text("")

    ### creating menu bar
    Main_menu = option_menu(
        menu_title="",
        options=["Article Summarizer","Text Summarizer","PDF Summarizer","App Info"],
        icons=["chat-dots","card-heading","file-earmark-pdf","person-circle"],
        default_index=0,
        key="Menu Bar"
    )
    st.text("")

    ### select modal for text and article summarizer
    if Main_menu == "Article Summarizer" or Main_menu == "Text Summarizer":

        Summarizer_modal = st.selectbox(
            label="Select Modal",
            options=np.array(list(Hugingface_modals.keys())),
            index=1,
            key="Modals"
        )

#### selecting number or paragraph for article summarizer
if Main_menu == "Article Summarizer":
    with app_sidebar:
        st.text("")
        st.text("")

        Number_of_article_paragraph = st.slider(
            label="Number of paragraph",
            min_value=1,max_value=10,
            step=1,value=2,
            key="Number of paragraph"
        )

with app_sidebar:
    st.button(
        label="Watch App Tutorial",
        use_container_width=True,
        on_click=watch_tutorial
    )


##### article summarizer functions

##### naive bayes text classification function

def is_url(text):
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(?:www\.)[^\s]+')
    return bool(url_pattern.match(text))


# Train a model for text vs URL classification
def train_model():
    """
    this function predict the given input
    is a simple text or url,link 
    and generate output.
    """
    #### dataset (normal text and URLs)
    try:
        data = [
            ('This is a normal sentence.', 'text'),
            ('www.google.com', 'url'),
            ('Check out this website', 'text'),
            ('https://www.example.com', 'url'),
            ('Machine learning is fun', 'text'),
            ('http://openai.com', 'url'),
            ('Python is a great language', 'text'),
        ]
        texts = [d[0] for d in data]
        labels = [1 if d[1] == 'url' else 0 for d in data]  ## 1 for url, 0 for text
        
        ##### modal training
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        
        model.fit(X_train, y_train)   #### Train the model
        
        model.score(X_train, y_train)
        model.score(X_test, y_test)
        
        return model
    
    except Exception as e:
        st.error("Error...\n\n",e,icon="⚠️")



###############################    article summarizer


if Main_menu == "Article Summarizer":

    blank_article1, article_column, blank_article2 = st.columns([2,8,2],gap="small")

    with blank_article1:  ### blank space
        pass
    with blank_article2:  ### blank space
        pass

    #### main app column
    with article_column:
        
        #### app title
        st.text("")
        App_Title = colored_header(
            label="Web Article Summarizer 📑",
            color_name="blue-green-70",
            description="Search or paste url"
        )
  
        Text_input = st.text_input(
                label="Search or paste url",
                placeholder="machine learning, java  url- https://www.example.com"
        )  
        
        ### max slider value
        def max_length_slider_value(max_length)->int:
            if max_length == 1:
                return 90
            elif max_length == 2:
                return 150
            elif max_length == 3:
                return 250
            elif max_length == 4:
                return 380
            elif max_length == 5:
                return 470
            elif max_length == 6:
                return 600
            elif max_length == 7:
                return 750
            elif max_length == 8:
                return 900
            elif max_length == 9:
                return 1200
            elif max_length == 10:
                return 1360
        
        @st.cache_data
        def Default_max_length(default_value):
            if default_value == 1:
                random_value = np.random.randint(30,65,6)
                return random.choice(random_value)
            
            elif default_value == 2:
                random_value = np.random.randint(50,130,6)
                return random.choice(random_value)
            
            elif default_value == 3:
                random_value = np.random.randint(70,210,6)
                return random.choice(random_value)
            
            elif default_value == 4:
                random_value = np.random.randint(140,310,6)
                return random.choice(random_value)
            
            elif default_value == 5:
                random_value = np.random.randint(200,390,6)
                return random.choice(random_value)
            
            elif default_value == 6:
                random_value = np.random.randint(230,490,6)
                return random.choice(random_value)
            
            elif default_value == 7:
                random_value = np.random.randint(280,590,6)
                return random.choice(random_value)
            
            elif default_value == 8:
                random_value = np.random.randint(350,750,6)
                return random.choice(random_value)
            
            elif default_value == 9:
                random_value = np.random.randint(450,1050,6)
                return random.choice(random_value)
            
            elif default_value == 10:
                random_value = np.random.randint(560,1100,6)
                return random.choice(random_value)

            

        
        Button_column, Toggle_summary_btn, Modal_display = st.columns([1,1,3],gap="small")  

        
        # article_summarizer(max_length)
        with Button_column:
        ### generate article button
            Generate_btn = st.button(label="Generate Article")

        with Toggle_summary_btn:
            ### if on then it generates summary
            summary_on = st.toggle(
                label="Summarizer",
                value=False,
                key="Summarizer on off"
            )

            if summary_on:
                st.toast(body="Summarizer Mode on",icon="📑")
            else:
                st.toast(body="Scraping Mode",icon="📰")
            
        with Modal_display:
            
            if summary_on:
                Modal_Level(Summarizer_modal)  
            else:
                pass          
        if summary_on:
            max_length_article = st.slider(
                label="max length",
                min_value=10,max_value=max_length_slider_value(Number_of_article_paragraph),
                key="max length",value=Default_max_length(Number_of_article_paragraph)
            ) 
        

################################################################################################

        
        ### article scraper function
        def article_scraper(article_url):
            """
            this function is used to scrap
            web articles and it provide
            text in the clean format
            """
            try:
                article = Article(article_url)  ### article object
                article.download()
                article.parse()
                nltk.download("punkt")
                article.nlp()

                st.markdown("<h4>Article</h4>",unsafe_allow_html=True)
                st.text("")
                st.text("")

                st.markdown(   ### article title
                    f"""
                        <h6><b>{article.title}</b></h6>
                    """,unsafe_allow_html=True
                )

                article_publishdate = article.publish_date   ### article publish date
                if article_publishdate == None:
                    pass
                else:
                    st.text("published on - "+str(article_publishdate))
                
                article_authors = article.authors   #### article authors
                if len(article_authors) == 0:
                    pass
                else:
                    autho_name_print = ", ".join(map(str, article_authors))
                    st.write(autho_name_print)

                
                ### generating article summary
                def get_top_paragraphs(text, num_paragraphs=Number_of_article_paragraph):
                    """
                    this function gives
                    top 1 - 10 paragraph of the 
                    scrap data
                    """
                    paragraphs = text.split('\n\n')

                    valid_paragraphs = [p.strip() for p in paragraphs if len(p.strip().split()) > 12]
                    top_paragraphs = valid_paragraphs[:num_paragraphs]
                    return '\n\n'.join(top_paragraphs)


                article_summary = article.text

                def remove_bracketed_numbers(text)->str:
                    pattern = r'\[\d+\]'
                    cleaned_text = re.sub(pattern, '', text)
                    return cleaned_text

                
                cleaned_article_text = remove_bracketed_numbers(get_top_paragraphs(article_summary))

                if "clean_text" not in st.session_state:
                    st.session_state.clean_text = ""

                st.session_state.clean_text = cleaned_article_text

                def clean_output_text(text:str)->str:
                    """
                    it gives clean text without emojies,
                    no ascii values english text
                    """
                    clean_text = clean(
                        text=text,fix_unicode=True,
                        to_ascii=True,no_emoji=True,
                        lang="en",no_line_breaks=False,
                        keep_two_line_breaks=True
                    )
                    return clean_text
                ### Print the cleaned text
                st.write(clean_output_text(st.session_state.clean_text))
                st.text("")
                st.text("")


                ### copy download button
                Article_filename = f"{article.title}.doc"

                Article_text_format = f"""
                    \n\n\n
{str(article.title)}
published on - {str(article_publishdate)}
Authors - {", ".join(map(str, article_authors))}
        \n\n\n
{str(cleaned_article_text)}
                """

                
                if __name__=="__main__":
                    Copy_download_button(
                        article_text=clean_output_text(cleaned_article_text),
                        article_format=Article_text_format,
                        article_file_name=Article_filename
                    )
                    
                st.text("")
                
                if summary_on:
                    st.markdown("<h4>Article Summary</h4>",unsafe_allow_html=True)

                    #### summarization modal

                    with st.spinner("Generating Summary..."):

                        
                        if __name__=="__main__":
                            summarized_article_text = Hugingface_summarization_modal(
                                summary_text=clean_output_text(cleaned_article_text),
                                modal_name=Summarizer_modal,
                                maximum_length=max_length_article
                            )
                            #### clean ai generated paragraph
                            

                            st.write(summarized_article_text)
                            st.text("")
                            st.text("")

                            summary_format = f"""

\n\n
{article.title}
\n\n\n
{summarized_article_text}
"""
                            #### copy or download summary button
                            if __name__=="__main__":
                                Copy_download_button(
                                    article_text=summarized_article_text,
                                    article_file_name=f"{article.title}-summary.doc",
                                    article_format=summary_format
                                )
                
                if summary_on:

                    ### summarization details
                    summarization_details = {
                        "Summarization Details":["Modal Name","Text Length","Summary Length","Max Tokens"],
                        "Output":[
                            f"{Summarizer_modal}",
                            f"Length - {len(cleaned_article_text.split())}",
                            f"Length - {len(summarized_article_text.split())}",
                            f"Tokens Used - {max_length_article}"
                        ]
                    }

                    summarization_details_df = pd.DataFrame(
                        data=summarization_details,
                        index=["Hugingface Modal","No. words","No. Words","Max Length"]
                    )

                    st.text("")
                    st.text("")
                    st.text("")
                    st.dataframe(summarization_details_df,use_container_width=True)
                        


            except Exception as err:
                ### 404 error animation

                Error_404_col, page_not_found_col = st.columns(2)

                with Error_404_col:

                    try:
                        Error_404 = insert_lottie_animation("lottie_animations/error-404.json")
                        st_lottie(
                            animation_source=Error_404,
                            speed=1,
                            reverse=False,loop=True,
                            quality="high",
                            height=315,
                            width=400,
                            key="404 error"
                        )
                    except Exception as err:
                        st.warning("something went wrong...",err,icon="⚠️")

                with page_not_found_col:    
                    
                    try:
                        page_not_found = insert_lottie_animation("lottie_animations/page-not-found.json")
                        st_lottie(
                            animation_source=page_not_found,
                            speed=1,
                            reverse=False,loop=True,
                            quality="high",
                            height=265,
                            width=400,
                            key="page not found"
                        )
                    except Exception as err:
                        st.warning("something went wrong...",err,icon="⚠️")

                st.warning(f"Something went wrong...\n\n{err}",icon="⚠️")

        def article_summarizer(summary_length):
            st.write(summary_length)

        
        def check_url_exists(url):
            try:
                response = requests.head(url, allow_redirects=True)
                if response.status_code < 400:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException as e:
                # Handle any exception (e.g., connection error, timeout)
                return False


        ###########      link classified article
        def link_classified(text):
            """
            it use url or link to scrap articles
            provide author name, publish date, summary of
            article
            """
            try:
                url_text = text
                article_url_link = f"{url_text}" ### url to scrap
                if __name__=="__main__":
                    article_scraper(article_url_link)
                    st.text("")
                    st.text("")

                    if check_url_exists(article_url_link):
                        st.link_button(label="Visit Article",url=(article_url_link))
                    else:
                        st.warning("Url does not exist...",icon="⚠️")

                    st.text("")
                    st.text("")
                    st.text("")
                    st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)

            except Exception as err:
                st.warning(f"Something went wrong...\n\n{err}",icon="⚠️")



        ####$     text classified article
        def text_classified(text):
            """
            it use wikipedia to scrap articles
            provide author name, publish date, summary of
            article
            """
            try:
                url_text = text.replace(" ","_")
                article_url = f"https://en.wikipedia.org/wiki/{url_text}" ### url to scrap
                if __name__=="__main__":
                    article_scraper(article_url)
                    st.text("")
                    st.text("")

                    if check_url_exists(article_url):
                        st.link_button(label="Visit Article",url=article_url)
                    else:
                        st.warning("Url does not exist...",icon="⚠️")

                    st.text("")
                    st.text("")
                    st.text("")
                    st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)


            except Exception as e:
                st.warning("Something went wrong...",e,icon="⚠️")

        

############################################################################################
        
        ### j query animation
        if not Generate_btn or Text_input.strip() == "":
            
            try:
                def particle(Js_file):
                    with open(Js_file) as f:
                        component.html(f"{f.read()}", height=420)

                if __name__=="__main__":
                    particle("animation/particles.html")
        
            except Exception as e:
                st.error("Something went wrong...\n\n",e)

        if Generate_btn:
            if Text_input.strip() != "":
                st.text("")
                st.text("")

                ### Function to classify the input text
                def classify_input(text, model):
                    try:
                        if is_url(text):
                            link_classified(text)
                        else:
                            #### If it's not detected as a URL
                            prediction = model.predict([text])[0]
                            if prediction == 1:
                                link_classified(Text_input)
                            else:
                                text_classified(Text_input)
                    except Exception as e:
                        st.error("Error...\n\n",e,icon="⚠️")

                with st.spinner("Generating Article..."):
                    if __name__=="__main__":
                        model = train_model()
                        classify_input(Text_input, model)



####################################################################################################


#################################      Text summarizer
                 
        
if Main_menu == "Text Summarizer":
    
    blank_text_sum1, text_summarizer_col, blank_text_sum2 = st.columns([2,8,2],gap="small")

    ### blank columns
    with blank_text_sum1:
        pass
    with blank_text_sum2:
        pass

    ### text summarizer app column

    with text_summarizer_col:
        #### app title
        st.text("")
        text_summarizer_Title = colored_header(
            label="Text Summarizer 📄",
            color_name="violet-70",
            description="enter or paste text hear"
        )

        placeholder_text = """write or paste your text hear 
paragraph length should be greater then 30 words
to generate output tap on screen or press ctrl+enter
        """

        ### input box
        text_summarizer_input = st.text_area(
            label="Enter Text Hear",
            placeholder=placeholder_text,
            height=340,
            key="text summarizer"
        )
        Modal_Level(Summarizer_modal) 
        
        if text_summarizer_input.strip() == "":

            try:
                #### writing animation
                write_hear_animation  = insert_lottie_animation("lottie_animations/write-hear.json")
                st_lottie(
                    animation_source=write_hear_animation,
                    speed=1,
                    reverse=False,loop=True,
                    quality="medium",
                    height=165,
                    width=240,
                    key="write hear"
                )
            except Exception as err:
                st.warning("something went wrong...",err,icon="⚠️")

        ### enter paragraph length greater than 35 words
        elif len(text_summarizer_input.split()) < 20:
            st.warning("paragraph should be greater than 35 words",icon="✏️")   
                 
        else:
            
            def word_token_maxvalue(text:str)->int:
                """
                converting paragraph into
                tokens
                """
                word_para = []
                words = word_tokenize(text)
                for i in words:
                    word_para.append(i)

                return len(word_para)
            
            @st.cache_data
            def random_value_text(text:str)->int:
                random_value = np.random.randint(
                    10,word_token_maxvalue(text),6
                )
                
                return random.choice(random_value)
            
            def clean_data_for_summarization(text:str)->str:
                clean_text = clean(
                        text=text,fix_unicode=True,
                        to_ascii=True,no_emoji=True,
                        lang="en",no_line_breaks=False,
                        keep_two_line_breaks=True
                    )
                return clean_text

        

            text_Max_length = st.slider(
                label="Max length",
                min_value=10,
                max_value=word_token_maxvalue(text_summarizer_input),
                key="text summarizer max length",
                step=1,value=random_value_text(text_summarizer_input)
            )

            Generate_text_summary = st.button(
                label="Generate summary",key="text summary"
            )

            try:
                #### writing loading
                writing_loading_animation  = insert_lottie_animation("lottie_animations/writing-loading.json")
                summary_generating_animation = st_lottie_spinner(
                    animation_source=writing_loading_animation,
                    speed=2,
                    reverse=False,loop=True,
                    quality="medium",
                    height=165,
                    width=240,
                    key="writing generating"
                )
            except Exception as err:
                st.warning("something went wrong...",err,icon="⚠️")


            #### initilization of modal
            if Generate_text_summary:

                if __name__=="__main__":
                    
                    ##### summary generation
                    with summary_generating_animation:

                        ### modal
                        Text_Summary_output = Hugingface_summarization_modal(
                            summary_text=clean_data_for_summarization(text_summarizer_input),
                            modal_name=Summarizer_modal,
                            maximum_length=text_Max_length
                        )

                        ##### summary displaying and copy
                        st.text("")
                        st.text("")
                        st.markdown("<h4>Generated Summary</h4>",unsafe_allow_html=True)
                        st.text("")
                        st.write(Text_Summary_output)
                        st.text("")
                        
                        copy_text(Text_Summary_output)
                        st.text("")
                        st.text("")

                        ###### original text desplay and copy
                        st.markdown("<h4>Original Text</h4>",unsafe_allow_html=True)
                        st.text("")
                        original_text = clean_data_for_summarization(text_summarizer_input)
                        st.write(original_text)
                        st.text("")
                        copy_text(original_text)

                        st.text("")
                        st.text("")
                        st.text("")

                         ### summarization details
                        text_summarization_details = {
                            "Summarization Details":["Modal Name","Text Length","Summary Length","Max Tokens"],
                            "Output":[
                                f"{Summarizer_modal}",
                                f"Length - {len(text_summarizer_input.split())}",
                                f"Length - {len(Text_Summary_output.split())}",
                                f"Tokens Used - {text_Max_length}"
                            ]
                        }

                        summarization_details_df = pd.DataFrame(
                            data=text_summarization_details,
                            index=["Hugingface Modal","No. words","No. Words","Max Length"]
                        )

                        st.text("")
                        st.text("")
                        st.text("")
                        st.dataframe(summarization_details_df,use_container_width=True)
                        st.text("")
                        st.text("")
                        st.text("")
                        st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)



##############################################################################################################

##############################      pdf summarizer


#### pdf and text summarizer functions


#### displaying uploaded pdf file
def display_pdf_file(uploaded_file):
    """
    it is used to display the
    file on screen
    """
    #### saving the uploaded file
    def save_uploadfile(save_file):
        with open(os.path.join("data",save_file.name),"wb") as f:
            f.write(save_file.getbuffer())
            return st.toast("file uploaded: {}".format(save_file.name))
        
    try:
        ### display pdf on screen
        def displayPDF(pdf_file):
            with open(pdf_file,"rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            pdf_display = f"""
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="580" height="700"
                    type="application/pdf"
                >
                </iframe>
            """

            st.markdown(pdf_display,unsafe_allow_html=True)

        ### save and display file
        save_uploadfile(uploaded_file)
        pdf_file = "data/"+uploaded_file.name
        displayPDF(pdf_file)
    except Exception as e:
        st.warning("Something Went wrong...\n\n",e,icon="⚠️")


#### Function to extract text from a specific page using pdfminer
def extract_text_pdfminer(pdf_file, page_number):
    """
    this function extract pdf file
    text by user input page number
    """
    try:
        extracted_text = ''
        for i, page_layout in enumerate(extract_pages(pdf_file)):
            if i == page_number - 1:  
                ### Extract text elements and format them as closely as possible to the original layout
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        for text_line in element:
                            if isinstance(text_line, LTTextLine):
                                line = ''.join([char.get_text() for char in text_line if isinstance(char, LTChar)])
                                extracted_text += line.strip() + '\n'
                return extracted_text
        return st.warning("Invalid page number.",icon="⚠️")
    except Exception as e:
        st.warning("Something Went wrong...\n\n",e,icon="⚠️")


###############################################


##### clean text for summmarization task
def uploaded_Clean_Text_Summarization(clean_text:str)->str:
        """
        it gives clean text for
        summarization task
        """
        try:
            pattern = r'[|`~^$<>]'
            cleaned_paragraph = re.sub(pattern, '', clean_text)

            ### using clean function
            clean_output_para = clean(
                text=cleaned_paragraph,fix_unicode=True,
                to_ascii=True,no_emoji=True,
                lang="en",no_line_breaks=False,
                keep_two_line_breaks=True
            )

        except Exception as e:
            st.warning("Something Went wrong...\n\n",e,icon="⚠️")

        return clean_output_para


### convert paragraph into tokens
def generate_text_para_tokens(text_para:str)->int:
    """
    converting paragraph into
    tokens
    """
    try:
        pattern = r'[|`~#^$<>]'
        cleaned_paragraph = re.sub(pattern, '', text_para)

            #### using clean function
        clean_para = clean(
            text=cleaned_paragraph,fix_unicode=True,
            to_ascii=True,no_emoji=True,
            lang="en",no_line_breaks=False,
            keep_two_line_breaks=True
        )

        word_tokens = []

        for i in word_tokenize(clean_para):
            word_tokens.append(i)
        return len(np.array(word_tokens))
    
    except Exception as e:
        st.warning("Something Went wrong...\n\n",e,icon="⚠️")



    ### generates random value for slider
@st.cache_data
def random_text_para_value(para:str)->int:
    try:
        random_value = np.random.randint(
            20, generate_text_para_tokens(para), 6
        )
        return random.choice(random_value)
    except Exception as e:
        st.warning("Something Went wrong...\n\n",e,icon="⚠️")


####  PDF files summarizer
def process_pdf(file):
    reader = PdfReader(file)
    page_count = len(reader.pages)

    ### pdf display and information column
    pdf_display_tab, pdf_summarizer_tab = st.tabs([f"Displaying {file.name}","Pdf Summarizer"])

    ####### displaying pdf on pdf display tab
    with pdf_display_tab:
        st.markdown(f"<h4>Pdf - {file.name}</h4>",unsafe_allow_html=True)

        pdf_col, pdf_info_col = st.columns([5,3],gap="medium")
        with pdf_col:
            with st.spinner("Displaying file..."):
                if __name__=="__main__":
                    display_pdf_file(file)

        with pdf_info_col:
            st.write("Your File: {}".format(file.name))
            st.write(f"Number of pages: {str(page_count)}")
            st.markdown(insert_html("htmlfiles/pdf-summarizer-info.html"),unsafe_allow_html=True)

           
    ### pdf information and intract with pdf
    with pdf_summarizer_tab:

        st.text("")
        st.markdown("<h4>Extract pdf text</h4>",unsafe_allow_html=True)

        ### toggle button for extracting text
        extract_by_page_all = st.toggle(
            label="Extract whole Text",key="toggle for extract text",
            value=False
        )

        ### extracting all pdf text
        if extract_by_page_all:
            st.write("Extract whole pdf Text")

            if st.button("Extract Whole Pdf",key="whole pdf text extract"):
            
                st.text("")
                st.text("")

                with st.spinner("Extracting pdf..."):
                    whole_pdf_text = extract_text(file)
                    st.markdown("<h4 style='font-size: 26px'>Whole PDF Text</h4>",unsafe_allow_html=True)
                    st.text("")
                    st.write(whole_pdf_text)
        else:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
            st.write("Extract by page Number")

            pdf_page_no_col, pdf_page_noinfo_col = st.columns([3,5],gap="small")

            with pdf_page_no_col:

                ### input page number
                Pdf_page_number_input = st.number_input(
                    label="Select the page number",
                    min_value=1, max_value=total_pages,
                    value=1,key="pdf page number",step=1
                )

            with pdf_page_noinfo_col:
                st.text("")
                st.text("")
                st.write(f"Selected page: {str(Pdf_page_number_input)}")

            Extract_page_no_button = st.button(
                label="Extract Page text",
                key="Extract button for page"
            )
            st.text("")
            st.text("")

            if Extract_page_no_button:
                text_pdfminer = extract_text_pdfminer(file, Pdf_page_number_input)
                st.session_state['extracted_text'] = text_pdfminer  ### Store the extracted text in session state
            
            if 'extracted_text' in st.session_state:
                Pdf_file_text = st.text_area(
                    label=f"Text data of {Pdf_page_number_input} page",
                    value= st.session_state['extracted_text'], 
                    height=400
                )
                st.session_state['extracted_text'] = Pdf_file_text  # Update the text in session state based on user's input

                #### pdf summarizer
                st.text("")
                Max_length_pdf_slider = st.slider(
                    label="Max Length",key="Pdf summarizer slider",
                    min_value=10,max_value=generate_text_para_tokens(Pdf_file_text),
                    value=random_text_para_value(Pdf_file_text)
                )
                st.text("")

                upload_Pdf_summary_btn_col, upload_Pdf_print_btn_col, upload_clean_Pdf_print_btn_col, blank_Pdf_col1, blank_Pdf_col2 = st.columns(
                    [4,4,4,7,3],gap="small"
                )

                with blank_Pdf_col1:
                    pass
                with blank_Pdf_col2:
                    pass

                with upload_Pdf_summary_btn_col:
                    Generate_upload_pdf_summary_btn = st.button(
                        label="Generate Summary",
                        key="Generate summary of uploaded text pdf"
                    )

                with upload_clean_Pdf_print_btn_col:
                    Upload_clean_pdf_btn = st.button(
                        label="Print Clean Text",
                        key="Print clean pdf file"
                    )


                with upload_Pdf_print_btn_col:
                    upload_pdf_print_button = st.button(
                        label="Print Uploaded Text",
                        key="Print uploadded pdf"
                    )
                
                ### clean text
                if Upload_clean_pdf_btn:
                    with st.spinner("Generating Clean Text..."):
                        st.text("")
                        st.text("")
                        st.markdown("<h4 style='font-size: 26px'>Clean Text</h4>",unsafe_allow_html=True)
                        st.text("")
                        st.write(uploaded_Clean_Text_Summarization(Pdf_file_text))
                        st.text("")
                        copy_text(uploaded_Clean_Text_Summarization(Pdf_file_text))
                        st.text("")
                        st.text("")
                        st.text("")
                        st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)

                ### uploaded text
                elif upload_pdf_print_button:
                    with st.spinner("Generating Uploaded Text..."):
                        st.text("")
                        st.text("")
                        st.markdown("<h4 style='font-size: 26px'>Uploaded Text</h4>",unsafe_allow_html=True)
                        st.text("")
                        st.text(Pdf_file_text)
                        st.text("")
                        copy_text(Pdf_file_text)
                        st.text("")
                        st.text("")
                        st.text("")
                        st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)

                ### generating summary
                elif Generate_upload_pdf_summary_btn:
                    st.text("")
                    with st.spinner("Generating Summary..."):
                        st.text("")
                        if __name__=="__main__":
                            Uploded_Pdf_file_Summary = Hugingface_summarization_modal(
                                summary_text=uploaded_Clean_Text_Summarization(Pdf_file_text),
                                maximum_length=Max_length_pdf_slider,
                                modal_name="facebook-bart"
                            )
                            st.markdown("<h4 style='font-size: 26px'>Summary</h4>",unsafe_allow_html=True)
                            st.text("")

                            st.write(Uploded_Pdf_file_Summary)
                            st.text("")
                            copy_text(Uploded_Pdf_file_Summary)
                            st.text("")
                            st.text("")
                            st.text("")
                            st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)


                
                
#################################################


##### text file summarizer
def process_text(file):
    text_file = file.read().decode("utf-8")
    st.text("")
    st.markdown("<h4 style='font-size: 26px'>Text file</h4>",unsafe_allow_html=True)    

    
    ### displaying text you can edit also
    Uploaded_text = st.text_area(
        label=f"{file.name[:-4]} text data",
        value=text_file,key="text file data",
        height=400
    )
    st.write(f"**{file.name[:-4]}** Edit your file press ctrl+enter")

    ###3 if length is less than 20
    if len(Uploaded_text.split()) < 20:
        st.warning("Summarization Task failed\nnot enough amount of text...",icon="⚠️")

    else:
        st.text("")
        #### max length slider
        max_text_para_length = st.slider(
            label="Max Length",min_value=10,
            max_value=generate_text_para_tokens(Uploaded_text),
            step=1,key="paragraph length",
            value=random_text_para_value(Uploaded_text)
        )
        st.text("")
        
        upload_text_summary_btn_col, upload_text_print_btn_col, upload_clean_text_print_btn_col, blank_text_col1, blank_text_col2 = st.columns(
            [4,4,4,7,3],gap="small"
        )

        with blank_text_col1:
            pass
        with blank_text_col2:
            pass

        with upload_text_summary_btn_col:
            Generate_upload_text_summary_btn = st.button(
                label="Generate Summary",
                key="Generate summary of uploaded text"
            )

        with upload_clean_text_print_btn_col:
            Upload_clean_text_btn = st.button(
                label="Print Clean Text",
                key="Print clean text file"
            )


        with upload_text_print_btn_col:
            upload_text_print_button = st.button(
                label="Print Uploaded Text",
                key="Print uploadded text"
            )
        
        ### clean text
        if Upload_clean_text_btn:
            with st.spinner("Generating Clean Text..."):
                st.text("")
                st.text("")
                st.markdown("<h4 style='font-size: 26px'>Clean Text</h4>",unsafe_allow_html=True)
                st.text("")
                st.write(uploaded_Clean_Text_Summarization(Uploaded_text))
                st.text("")
                copy_text(uploaded_Clean_Text_Summarization(Uploaded_text))
                st.text("")
                st.text("")
                st.text("")
                st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",
                            unsafe_allow_html=True)

        ### uploaded text
        elif upload_text_print_button:
            with st.spinner("Generating Uploaded Text..."):
                st.text("")
                st.text("")
                st.markdown("<h4 style='font-size: 26px'>Uploaded Text</h4>",unsafe_allow_html=True)
                st.text("")
                st.text(Uploaded_text)
                st.text("")
                copy_text(Uploaded_text)
                st.text("")
                st.text("")
                st.text("")
                st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",
                            unsafe_allow_html=True)


        ### generating summary
        elif Generate_upload_text_summary_btn:
            st.text("")
            with st.spinner("Generating Summary..."):
                st.text("")
                if __name__=="__main__":
                    Uploded_Text_file_Summary = Hugingface_summarization_modal(
                        summary_text=uploaded_Clean_Text_Summarization(Uploaded_text),
                        maximum_length=max_text_para_length,
                        modal_name="facebook-bart"
                    )
                    st.markdown("<h4 style='font-size: 26px'>Summary</h4>",unsafe_allow_html=True)
                    st.text("")

                    st.write(Uploded_Text_file_Summary)
                    st.text("")
                    copy_text(Uploded_Text_file_Summary)
                    st.text("")
                    st.text("")
                    st.text("")
                    st.markdown("<h6 style='text-align: center;'>Created by Nishant Maity</h6>",unsafe_allow_html=True)



if Main_menu == "PDF Summarizer":
    
    ### blank and app columns
    Blank_pdf1 ,pdf_summarizer_col, Blank_pdf2 = st.columns([1,8,1],gap="small") 

    with Blank_pdf1:
        pass
    with Blank_pdf2:
        pass

    with pdf_summarizer_col:
        st.text("")
        st.header("PDF Summarizer")   ### app heading

        ### File uploader function
        app_file_upload = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

        if app_file_upload is not None:
            
            ### if pdf file
            if app_file_upload.type == "application/pdf":
                if __name__=="__main__":
                    process_pdf(app_file_upload)

            #### if text file
            elif app_file_upload.type == "text/plain":
                if __name__=="__main__":
                    process_text(app_file_upload)
        
        else:
            st.info("Upload your pdf, text file")
            

 #### app info           
if Main_menu == "App Info":
    Blank_app_info1, App_info_col, Blank_app_info2 = st.columns([2,8,2])

    #### blank columns
    with Blank_app_info1:
        pass
    with Blank_app_info2:
        pass

    ### app info column
    with App_info_col:
        st.text("")
        st.header("App Info")
        st.text("")

        if __name__=="__main__":
            st.markdown(insert_html("htmlfiles/app-info.html"),
                unsafe_allow_html=True
            )

