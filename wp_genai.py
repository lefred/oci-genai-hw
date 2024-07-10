import oci
import mysql.connector

import wp_config

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean
from bs4 import BeautifulSoup


# Constants
compartment_id = wp_config.COMPARTMENT
config = oci.config.from_file(wp_config.CONFIG_FILE, wp_config.CONFIG_PROFILE)

# Service endpoint
endpoint = wp_config.ENDPOINT


generative_ai_inference_client = (
    oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240),
    )
)


myconfig = {
    "user": wp_config.DB_USER,
    "password": wp_config.DB_PASSWORD,
    "host": wp_config.DB_HOST,
    "port": wp_config.DB_PORT,
    "database": wp_config.DB_SCHEMA,
}


def connectMySQL(myconfig):
    cnx = mysql.connector.connect(**myconfig)
    return cnx


# Used to format response and return references
class Document:

    doc_id: int
    doc_text: str
    wp_post_id: int

    def __init__(self, id, text, wp_post_id) -> None:

        self.doc_id = id
        self.doc_text = text
        self.wp_post_id = wp_post_id

    def __str__(self):
        return f"doc_id:{self.doc_id},doc_text:{self.doc_text},wp_post_id:{self.wp_post_id}"


# OCI-LLM: Used to generate embeddings for question(s)
def generate_embeddings_for_question(question_list):

    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.inputs = question_list
    embed_text_detail.input_type = embed_text_detail.INPUT_TYPE_SEARCH_QUERY
    embed_text_detail.serving_mode = (
        oci.generative_ai_inference.models.OnDemandServingMode(
            model_id="cohere.embed-english-v3.0"
        )
    )
    embed_text_detail.compartment_id = compartment_id
    embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
    return embed_text_response


# OCI-LLM: Used to prompt the LLM
def query_llm_with_prompt(prompt):

    cohere_generate_text_request = (
        oci.generative_ai_inference.models.CohereLlmInferenceRequest()
    )
    cohere_generate_text_request.prompt = prompt
    cohere_generate_text_request.is_stream = False
    cohere_generate_text_request.max_tokens = 1000
    cohere_generate_text_request.temperature = 0.75
    cohere_generate_text_request.top_k = 5
    cohere_generate_text_request.top_p = 0

    generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
    generate_text_detail.serving_mode = (
        oci.generative_ai_inference.models.OnDemandServingMode(
            model_id="cohere.command"
        )
    )
    generate_text_detail.compartment_id = compartment_id
    generate_text_detail.inference_request = cohere_generate_text_request

    generate_text_response = generative_ai_inference_client.generate_text(
        generate_text_detail
    )

    llm_response_result = (
        generate_text_response.data.inference_response.generated_texts[0].text
    )

    return llm_response_result


# Find relevant records from HeatWave using Dot Product similarity.
def search_data(cursor, query_vec, list_dict_docs):

    myvectorStr = ",".join(str(item) for item in query_vec)
    myvectorStr = "[" + myvectorStr + "]"

    relevant_docs = []
    mydata = myvectorStr
    cursor.execute(
        """
        select *
        from {}.wp_embeddings a
        order by vector_distance(vec, string_to_vector(%s)) desc
        LIMIT 20
    """.format(
            wp_config.DB_SCHEMA
        ),
        [myvectorStr],
    )

    for row in cursor:
        id = row[0]
        text = row[1]
        wp_post_id = row[3]
        temp_dict = {id: text}
        list_dict_docs.append(temp_dict)
        doc = Document(id, text, wp_post_id)
        # print(doc)
        relevant_docs.append(doc)

    return relevant_docs


# Perform RAG
def answer_user_question(query):

    question_list = []
    question_list.append(query)

    embed_text_response = generate_embeddings_for_question(question_list)

    question_vector = embed_text_response.data.embeddings[0]

    with connectMySQL(myconfig) as db:
        cursor = db.cursor()
        list_dict_docs = []
        # query vector db to search relevant records
        similar_docs = search_data(cursor, question_vector, list_dict_docs)

        # prepare documents for the prompt
        context_documents = []
        relevant_doc_ids = []
        similar_docs_subset = []

        for docs in similar_docs:
            current_id = str(docs.doc_id)
            relevant_doc_ids.append(current_id)
            similar_docs_subset.append(docs)
            context_documents.append(docs.doc_text)

        context_document = "\n".join(context_documents)
        prompt_template = """
        Text: {documents} \n
        Question: {question} \n
        Answer the question based on the text provided and also return the relevant document numbers where you found the answer. If the text doesn't contain the answer, reply that the answer is not available.
        """

        prompt = prompt_template.format(question=query, documents=context_document)

        llm_response_result = query_llm_with_prompt(prompt)
        response = {}
        response["message"] = query
        response["text"] = llm_response_result
        response["documents"] = [
            {"id": doc.doc_id, "snippet": doc.doc_text, "wp_post_id": doc.wp_post_id}
            for doc in similar_docs_subset
        ]

        return response


# Main Function

cnx = connectMySQL(myconfig)
if cnx.is_connected():
    cursor = cnx.cursor()
    cursor.execute("SELECT @@version, @@version_comment")
    results = cursor.fetchone()
    cnx.close()

    print("You are now connected to {} {}".format(results[1], results[0]))

    question = input("What is your question? ")
    myanswer = answer_user_question(question)
    print(myanswer["text"])
