from langchain.prompts import PromptTemplate

question_prompt_template = """ Use the following portion of a long document and return any text that is relevant to the question.
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""

QUESTION_PROMPT =  PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question/search string, create a final answer with references ("Sources").
If you do not know the answer, say you do not know. Do not try and make up an answer. Use paragraphs where appropriate.
Always return a "Sources" as part of your answer. Each source should be returned as following:

The following is an example, purely to express format. Do not use as information in the generated final answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: file/28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: This is just some content on something else.
Source: ghost-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
\nSources: <a href="file/28-pl">28</a>

QUESTION: LetsCode
=========
Content: LetsCode is a NGO created to help young children from low income backgrounds get into coding.
Source: file/someDirectory/NGOs-pl
Content: Some named primary school is partnered with LetsCode, to encourage children in the school to get into technology where they may not be able to.
Source: file/path/29-pl
Content: The NGO LetsCode was founded in 2023, by univeristy students who discovered the lack of diversity in the tech world.
Source: file/path/to/directory/founded.txt
=========
FINAL ANSWER: LetsCode is an NGO created by university students, for children of low income background to get into coding. Their aim is to improve diversity in the techworld, and have partnered with a named primary school.
\nSources: <a href="file/someDirectory/ngo-pl">NGOs</a>, <a href="file/path/29-pl">29</a>,  <a href="file/path/to/directory/founded.txt">Founded</a>


QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
