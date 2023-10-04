import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
import base64
from streamlit_option_menu import option_menu
import openai

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function for generating LLM response
def generate_chat_response(chain, prompt):
    response = chain.run(f'answer the following question: {prompt}')
    return response

def analyse_ingredients(gray_img, key, pdf_folder_path):
    #Extract text from image using pytesseract
    d = pytesseract.image_to_data(gray_img, output_type=Output.DICT)
    print(d)

    # Initialize a list to store block and paragraph numbers where 'ingredients' is found
    ingredient_locations = []

    # Iterate through the data to find 'ingredients' and store the block and paragraph numbers
    for i in range(len(d['text'])):
        if d['text'][i] == 'ingredients' or d['text'][i] == 'ingredient' or d['text'][i] == 'INGREDIENTS':
            block_num = d['block_num'][i]
            par_num = d['par_num'][i]
            ingredient_locations.append((block_num, par_num))

    # Print the list of locations where 'ingredients' was found
    print(ingredient_locations)

    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]

    index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
    llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                        model_kwargs={"temperature":0, "max_length":512})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=index.vectorstore.as_retriever(),
                                        input_key="question")


    list_paragraph = []
    for k in ingredient_locations:
        # Block and paragraph numbers you want to print
        block_num = k[0]
        par_num = k[1]
        paragraph = ''
        # Iterate through the d and print the text for block 1, paragraph 1
        for i in range(len(d['block_num'])):
            if d['block_num'][i] == block_num and d['par_num'][i] == par_num:
                paragraph += d['text'][i]
                paragraph += ' '
        list_paragraph.append(paragraph)
        print(paragraph)

    ingredients = ''
    for paragraph in list_paragraph:
        ingredients += chain.run(f'list ingredients from the following text:{paragraph}')

    ingredients_list = ingredients.split(", ")

    ingredient_res ={}
    for ingredient in ingredients_list:
        ingredient_res[ingredient] = chain.run(f'is {ingredient} Good or Okay or Caution or Avoid ?')

    print(ingredient_res)

    ingredient_categorie = {'Okay': [], 'Good': [], 'Caution': [], 'Avoid': []}

    for ingredient, category in ingredient_res.items():
        ingredient_categorie[category].append(ingredient)

    return ingredient_categorie

#image Background
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


#MenuBar
def streamlit_menu():
        # horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "HairGuardianBot", "Contact"],  # required
            #icons=["house", "book", "envelope"],  # optional "icon": {"color": "orange", "font-size": "25px"},
            #menu_icon="cast",  # optional
            #default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "black"},
            },
        )
        return selected


selected = streamlit_menu()

if selected == "Home":
        set_png_as_page_bg('hairGuardian2.webp')

        #working with pdfs
        pdf_folder_path = 'pdfs'

        #Get HUGGINGFACEHUB_API_KEY
        key = "Add you key here"

        image = Image.open('img_logo2.png')
        st.image(image)
        st.markdown('<h2 style="color:gray; font-size: 28px;">| Analyse Your Hair Care Products With a Single Click</h2>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:gray; font-size: 20px">The HairGuardian Classifies ingredients into: </h3>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:black; font-weight: bold; margin-bottom: 15px; border: 4px dashed #000; border-radius: 50px; padding: 20px; background-color: rgba(255, 255, 255, 0.6); font-size: 18px"><span style="color:#50C948; font-weight: bold; text-shadow: 2px 2px 2px #000000; font-size: 22px"> Good </span>means that I like to see this in a list of ingredients. <span style="color:blue; font-weight: bold; text-shadow: 2px 2px 2px #000000; font-size: 22px">Okay </span>means it is a safe ingredient. <span style="color:orange; font-weight: bold; text-shadow: 2px 2px 2px #000000; font-size: 22px">Caution </span>means that this ingredient may not be good in some hair care products, or for some people.<span style="color:red; text-shadow: 2px 2px 2px #000000; font-weight: bold; font-size: 22px"> Avoid</span> means this ingredient may hurt your hair. If you see this ingredient in a hair product, it is best to put it down and walk away.</h3>', unsafe_allow_html=True)



        upload= st.file_uploader('Insert image for classification', type=['png','jpg'])            
        if upload is not None:
            st.markdown('<h3 style="color:Black; font-size: 22px">Input Image </h3>', unsafe_allow_html=True)
            im= Image.open(upload)
            st.image(im)
            st.markdown('<h3 style="color:Black; font-size: 22px">Result</h3>', unsafe_allow_html=True)
            img= np.asarray(im)
            gray_img = get_grayscale(img)
            image= cv2.resize(img,(224, 224))
            img= np.expand_dims(img, 0)
            analyse_result = analyse_ingredients(gray_img, key, pdf_folder_path)
            c1, c2, c3, c4= st.columns(4)
            for category, ingredient in analyse_result.items():
                if category == 'Good':
                    c1.write(f'<span style="font-size: 22px; color: green; font-weight: bold; text-shadow: 2px 2px 2px #000000;">Good</span>', unsafe_allow_html=True)
                    for ing in analyse_result[category]:
                            c1.write(f'<span style="color:black; font-weight: bold;  background-color: rgba(255, 255, 255, 0.6); font-size: 18px">{ing}</span>', unsafe_allow_html=True)
                if category == 'Okay':
                    c2.write(f'<span style="font-size: 22px; color: blue; font-weight: bold; text-shadow: 2px 2px 2px #000000;">Okay</span>', unsafe_allow_html=True)
                    for ing in analyse_result[category]:
                        c2.write(f'<span style="color:black; font-weight: bold;  background-color: rgba(255, 255, 255, 0.6); font-size: 18px">{ing}</span>', unsafe_allow_html=True)
                if category == 'Caution':
                    c3.write(f'<span style="font-size: 22px; color: orange; font-weight: bold; text-shadow: 2px 2px 2px #000000;">Caution</span>', unsafe_allow_html=True)
                    for ing in analyse_result[category]:
                        c3.write(f'<span style="color:black; font-weight: bold;  background-color: rgba(255, 255, 255, 0.6); font-size: 18px">{ing}</span>', unsafe_allow_html=True)     
                if category == 'Avoid':
                    c4.write(f'<span style="font-size: 22px; color: red; font-weight: bold; text-shadow: 2px 2px 2px #000000;">Avoid</span>', unsafe_allow_html=True)
                    for ing in analyse_result[category]:
                        c4.write(f'<span style="color:black; font-weight: bold;  background-color: rgba(255, 255, 255, 0.6); font-size: 18px">{ing}</span>', unsafe_allow_html=True)
     

if selected == "HairGuardianBot":
    #chatBot
    set_png_as_page_bg('hairGuardianBot.webp')
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt   
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)


    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                openai.api_key = "Add OpenAi Key"
                # Create a chatbot using ChatCompletion.create() function
                completion = openai.ChatCompletion.create(
                # Use GPT 3.5 as the LLM
                model="gpt-3.5-turbo",
                # Pre-define conversation messages for the possible roles
                messages=[
                    {"role": "assistant", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                )
                # Print the returned output from the LLM model
                response = completion.choices[0].message 
                st.write(response["content"])                 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message["content"])
                

if selected == "Contact":
    st.title(f"For more information :")
    st.markdown('<h3 style="color:black; font-weight: bold; margin-bottom: 15px; border: 4px dashed #000; border-radius: 0px; padding: 20px; background-color: rgba(255, 255, 255, 0.6); font-size: 18px"><span style="color:gray; font-weight: bold; text-shadow: 1px 1px 1px #000000; font-size: 20px"> Gmail: </span> KhadijaETTOUIL01@gmail.com <div><div><span style="color:gray; font-weight: bold; text-shadow: 1px 1px 1px #000000; font-size: 20px">LinkedIn: </span> <a href="https://ma.linkedin.com/in/khadija-ettouil-b54132241" target="_blank">Khadija ETTOUIL</a></div></div></h3>', unsafe_allow_html=True)









