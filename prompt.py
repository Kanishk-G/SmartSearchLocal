# flake8: noqa
from langchain_core.prompts import PromptTemplate

question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following powershell scripts, answer the user's questions. Ensure that you add the sources in which you attained these files at the end of the answer, after "Sources": 
If you don't know the answer, just say that you don't know. Don't try to make up an answer. The answer should only come from chat history and the scripts below.
The below 2 questions are just to show the format - none of the first two questions and their sources are to be used in answer creation.
ALWAYS return a "Sources" part in your answer.

QUESTION: Which script gets the IP Address configuration
=========
Content: Get-CimInstance -Class Win32_NetworkAdapterConfiguration -Filter IPEnabled=$true | Select-Object -ExpandProperty IPAddress
Source: localip.ps1
Content: Get-CimInstance -Class Win32_NetworkAdapterConfiguration -Filter IPEnabled=$true | Get-Member -Name IPAddress
Source: ipgetmember.ps1
Content: Get-CimInstance -Class Win32_NetworkAdapterConfiguration -Filter IPEnabled=$true
Source: ipconfig.ps1
=========
FINAL ANSWER: The ipconfig.ps1 script obtains the IP address configuration
SOURCES: ipconfig.ps1

QUESTION: What command can I use to get the directory location I am in?
=========
Content: Get-Location
Source: location.ps1
Content: Get-PSDrive
Source: psdrive.ps1
=========
FINAL ANSWER: You can use the Get-Location command
SOURCES: location.ps1

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nFile details: Filename - {name}, Link - {source}, File System Info - {fileSystemInfo}",
    input_variables=["page_content", "source", "name", "fileSystemInfo"],
)
