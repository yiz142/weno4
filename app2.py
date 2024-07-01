import time
import json
import io
import requests
import streamlit as st
import pandas as pd
from keplergl import keplergl
from langchain_groq import ChatGroq

# Import utility functions and other modules
from util import process_data_request, process_regulation_request, process_off_topic_request, process_data_commons_request
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan
from dataframe_table import render_interface_for_table
from data_commons import get_time_series_dataframe_for_dcid, get_dcid_from_county_name, get_dcid_from_state_name, get_dcid_from_country_name

# Function to load data from API
def load_data_from_api(api_url):
    response = requests.get(api_url)
    response.raise_for_status()  # Check for request errors
    data = pd.read_csv(io.StringIO(response.text))
    return data

# Load the GC3datasets.csv file if not already in session state
if 'datasets_df' not in st.session_state:
    datasets_csv_path = 'GC3datasets2.csv'
    st.session_state.datasets_df = pd.read_csv(datasets_csv_path)

datasets_df = st.session_state.datasets_df

# Setup LLM
Groq_KEY = st.secrets["Groq_KEY"]
# Groq_KEY_2 = st.secrets["Groq_KEY_2"]

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
# llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Set up the title
st.markdown("### &nbsp; WEN-OKN: Dive into Data, Never Easier")

selected_dataset = st.selectbox('Selected Dataset', datasets_df['dataset_name'])

if 'selected_dataset' not in st.session_state or \
    st.session_state.selected_dataset != selected_dataset:
    # Dropdown to select dataset
    st.session_state.selected_dataset = selected_dataset
    print(st.session_state.selected_dataset)

    # Retrieve dataset information based on selected dataset
    st.session_state.selected_dataset_info = datasets_df.loc[datasets_df['dataset_name'] == st.session_state.selected_dataset].iloc[0]

# Load data based on dataset_type

    if st.session_state.selected_dataset_info['dataset_type'] == 'dynamic':
        api_url = st.session_state.selected_dataset_info['file_path']
        if not api_url.startswith(('http://', 'https://')):
            api_url = 'https://' + api_url
        st.session_state.data_df = load_data_from_api(api_url)
        print(len(st.session_state.data_df))
    else:
        st.session_state.data_df = pd.read_csv(st.session_state.selected_dataset_info['file_path'])

data_df = st.session_state.data_df

# Extract schema from the dataframe if available
if 'schema' in datasets_df.columns and pd.notnull(st.session_state.selected_dataset_info['schema']):
    st.session_state.schema = st.session_state.selected_dataset_info['schema'].split(',')
else:
    st.session_state.schema = list(data_df.columns)

# Set up datasets in the session for GeoDataframes
if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Add datasets for tables
if "wen_datasets" not in st.session_state:
    st.session_state.wen_datasets = []
    st.session_state.wen_tables = []
    st.session_state.table_chat_histories = []
    st.session_state.chart_types = []

# Flag for managing rerun
if "rerun" not in st.session_state:
    st.session_state.rerun = False

# Add all generated SPARQL queries with the requests to Streamlit session state
if "sparqls" not in st.session_state:
    st.session_state.requests = []
    st.session_state.sparqls = []

@st.experimental_fragment
def add_map():
    options = {"keepExistingConfig": True}
    _map_config = keplergl(st.session_state.datasets, options=options, config=None, height=410)
    time.sleep(0.5)

    if _map_config:
        map_config_json = json.loads(_map_config)

        map_data_ids = [layer["config"]["dataId"] for layer in map_config_json["visState"]["layers"]]
        indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if dataset.id not in map_data_ids]

        deleted = False
        for i in reversed(indices_to_remove):
            if time.time() - st.session_state.datasets[i].time > 3:
                del st.session_state.datasets[i]
                del st.session_state.requests[i]
                del st.session_state.sparqls[i]
                deleted = True
        if deleted:
            st.rerun()
    return _map_config

# Set up CSS for tables
st.markdown("""
<style>
.tableTitle {
    font-size: 18pt;
    font-weight: 600;
    color: rgb(49, 51, 63);
    padding: 10px 0px 10px 0px;
}
.stDataFrame {
    margin-left: 50px;
}
</style>
""", unsafe_allow_html=True)

# Set up two columns for the map and chat interface
col1, col2 = st.columns([3, 2])

# Show all tables
if st.session_state.wen_datasets:
    for index, pivot_table in enumerate(st.session_state.wen_datasets):
        render_interface_for_table(llm, llm2, index, pivot_table)

# Show all requests and generated SPARQL queries
if len(st.session_state.sparqls) > 0:
    st.write(f"<div class='tableTitle'>Spatial Requests and SPARQL queries</div>", unsafe_allow_html=True)
    info_container = st.container(height=350)
    with info_container:
        for idx, sparql in enumerate(st.session_state.sparqls):
            if st.session_state.sparqls[idx] != "":
                st.markdown(f"**Request:**  {st.session_state.requests[idx]}")
                st.code(sparql)

# Set up the chat interface
with col2:
    chat_container = st.container(height=355)

    for message in st.session_state.chat:
        with chat_container:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    user_input = st.chat_input("What can I help you with?")

    if user_input:
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})
            route = get_question_route(llm, user_input)

            if route['request_type'] == 'WEN-KEN database':
                refined_request = get_refined_question(llm, user_input)
                if refined_request['is_request_data']:
                    plan = get_request_plan(llm, refined_request['request'])
                    count_start = len(st.session_state.datasets)
                    for request in plan['requests']:
                        process_data_request(request, chat_container)
                    count_end = len(st.session_state.datasets)
                    for idx in range(count_start, count_end):
                        st.session_state.datasets[idx].time = time.time()
                    st.session_state.chat.append({"role": "assistant",
                                                  "content": "Your request has been processed."})
                    st.rerun()
                else:
                    message = refined_request['alternative_answer']
                    st.chat_message("assistant").markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
            elif route['request_type'] == 'NPDES regulations':
                message = process_regulation_request(llm, user_input, chat_container)
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()
            elif route['request_type'] == 'Data Commons':
                code = process_data_commons_request(llm, user_input, st.session_state.datasets)
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            exec(code)
                            df.id = user_input
                            st.session_state.wen_datasets.append(df)
                            st.session_state.wen_tables.append(df.copy())
                            st.session_state.table_chat_histories.append([])
                            st.session_state.chart_types.append("bar_chart")
                            message = f"""
                                    Your request has been processed. {df.shape[0]} { "rows are" if df.shape[0] > 1 else "row is"}
                                    found and displayed.
                                    """
                        except Exception as e:
                            message = f"""We are not able to process your request. Please refine your 
                                          request and try it again. \n\nError: {str(e)}"""
                        st.markdown(message)
                        st.session_state.chat.append({"role": "assistant", "content": message})
                        st.rerun()
            else:
                message = process_off_topic_request(llm, user_input, chat_container)
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()

if st.session_state.rerun:
    st.session_state.rerun = False
    st.rerun()
