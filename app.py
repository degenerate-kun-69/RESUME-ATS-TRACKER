from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
import os
import streamlit as st
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import faiss
import pymupdf
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel(
        model_name='gemini-2.5-pro'
    )
    generation_config ={
        "temperature": 0.15,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "stop_sequences": ["\n"],
        "top_k": 40
    }
    response=model.generate_content(
        [input,pdf_content[0],prompt],
        generation_config=generation_config
    )
    return response.text
