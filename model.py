from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings,GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
import os
load_dotenv()

class ChatBotUS():
    def __init__(self, k = 30, url = 'local'):
        self.llm = None
        self.prompt = None
        self.db = None
        self.url = url
        
        self.template = 'Hãy trả lời câu hỏi sau bằng tiếng việt (Vietnamese): {question} dựa trên nội dung được cung cấp: {context}, nếu không có thông tin hãy nói tôi không biết, không diễn giả gì thêm'
    
        self.llm_chain = None
        self.k = k
        
        if url != 'local':
            self.db_temp = self.create_vectorstores()
        
        self.load_db()
        self.load_llm()            
        self.create_template()     
        self.create_qa_chain() 
        
        
    
    def get_data(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)

        driver.get(self.url)

        html_content = driver.page_source

        driver.quit()

        soup = BeautifulSoup(html_content, 'html.parser')

        data = soup.getText(separator='\n',strip= True)  
        return data
    
    def load_db(self):
        embeddings_model = GPT4AllEmbeddings(model_file='model/all-MiniLM-L6-v2-f16.gguf')
        
        self.db = FAISS.load_local('vectorstores',embeddings_model,allow_dangerous_deserialization=True)
        return self.db
    

    def create_vectorstores(self):
        data = self.get_data()

        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
        data = data.lower()
        chunks = text_spliter.split_text(data)
        embeddings_model = GPT4AllEmbeddings(model_file='model/all-MiniLM-L6-v2-f16.gguf')
        self.db = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
        return self.db

    def load_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.01,
            max_tokens=None,
            timeout=None,
            api_key=os.getenv('API_KEY')
        )

    def create_template(self):
        self.prompt = PromptTemplate(template=self.template, input_variables=['context', 'question'])

    def create_qa_chain(self):
        if self.url == 'local':
            self.llm_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type='stuff',
                retriever=self.db.as_retriever(search_kwargs={"k": self.k}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': self.prompt}
            )
        else:
            self.llm_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type='stuff',
                retriever=self.db_temp.as_retriever(search_kwargs={"k": self.k}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': self.prompt}
            )

    def make_response(self, question):
        response = self.llm_chain.invoke({'query': question.lower()})
        return response['result']
