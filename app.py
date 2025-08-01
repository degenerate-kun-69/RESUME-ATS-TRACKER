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
