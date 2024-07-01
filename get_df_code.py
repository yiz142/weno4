from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def get_df_code(llm, question, title, schema):
    prompt = PromptTemplate(
        template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert of {title} data which is loaded in a 
        DataFrame df with the following columns: 
        {schema}.
        Also the data is displayed to the user.

        Users can make queries in natural language to request data from this DataFrame, or they may ask other types of 
        questions.

        Here are some examples of data requests:
            "Show the rows with top 10 SPI values."
            "Show all data from the station Salt."

        And here are some examples of other questions:
            "What questions can I ask?"
            "What data do you have?"

        Please categorize the following user question as either a "Request data" or "Other" in a 
        JSON field "category". 

        For "Request data", return a python statement in the following format:
             st.session_state.df = <your expression with df only>      
        in the JSON field "answer".  Note that you can't use df.resample('Y', on='Time') 
        because the type of df['Time'] is string.
        
        Example 1: for each station, find the highest SPI in 2000, the answer could be
            st.session_state.df = df.loc[df[(df['Time'] >= '2000-01-01') & (df['Time'] <= '2000-12-31')].groupby('Station_Name')['SPI'].idxmax()]
        
        Example 2: find monthly average SPI for the station Salt
            st.session_state.df =  df.loc[(df['Station_Name'] == 'Salt')].groupby(df['Time'].str[:7]).agg({{'SPI': 'mean'}})
        
        For "Other", return a reasonable answer in the JSON field "answer". 

        Return JSON only without any explanations.

        User question:
        {question}

        Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )

    df_code_chain = prompt | llm | JsonOutputParser()
    return df_code_chain.invoke({"question": question})