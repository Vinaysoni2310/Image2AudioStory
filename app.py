from dotenv import find_dotenv, load_dotenv
import os
import requests
from langchain import LLMChain, PromptTemplate, OpenAI
import streamlit as st

_=load_dotenv(find_dotenv())
API_URL = os.getenv("HUGGINGFACE_API_KEY")


# Image to Text
def img2text(filename):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()[0]['generated_text']

#print(query("img.jpg")[0]['generated_text'])

# Text to Story

def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story on a simple narrative, the story should be no more than 50 words;

    CONTEXT = {scenario}
    STORY :
    """
    prompt = PromptTemplate(template=template, input_variables=['scenario'])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1),prompt=prompt,verbose=True
    )

    story = story_llm.predict(scenario=scenario)
    #print(story)

    return story


# Text to Audio
def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}

    payloads ={
            'inputs' : story
    }

    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.flac', "wb") as f:
        f.write(response.content)



def main():
    st.set_page_config(page_title="Image to Auido Story", page_icon="V")
    
    st.header("Turn Image into Audio Story")
    uploaded_file = st.file_uploader("Choose an Image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption = "Uploaded Image.", use_column_width = True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scennario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.flac")
    

if __name__ == '__main__':
    main()
