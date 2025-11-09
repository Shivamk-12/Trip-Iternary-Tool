from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)


st.header("Trip Iterneary Tool")

destination_input = st.text_input(
    "Enter your destination name",
    placeholder="e.g., Paris, Tokyo, New York"
)

Budget_input = st.selectbox(
    "Select your budget ",
    ["10000-15000", "15000-25000", "25000-30000", "8000-10000", "5000-8000"],
)

Days_Selection = st.selectbox("Select no of days", ["1-2 days", "3-5 days", "7-9 days"])


# user_input = st.text_input("Enter your query")

template = PromptTemplate(
    template="""
    You are a smart and helpful travel planner.

Create a personalized trip itinerary for the following user inputs:
- Destination: {destination_input}
- Budget: {Budget_input}
- Duration: {Days_Selection} days

The itinerary should include:
1. Daily activity suggestions (morning, afternoon, evening)
2. Popular attractions
3. Local food experiences
4. Travel and stay recommendations within the given budget
5. Tips for weather, local transport, or safety if relevant

Keep the language friendly and concise. Present the plan day-by-day, numbered clearly.

"""
)

input_variables = ["destination_input", "Budget_input", "Days_Selection"]

prompt = template.invoke(
    {
        "destination_input": destination_input,
        "Budget_input": Budget_input,
        "Days_Selection": Days_Selection,
    }
)

if st.button("Find"):
    result = chat_model.invoke(prompt)
    st.write(result.content)
