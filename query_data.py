import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from emb_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответьте на вопрос, основываясь только на контексте, который представлен ниже:

{context}

---

Ответьте на вопрос, исходя из приведенного выше контекста: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #поиск по бд
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    model = Ollama(model="ilyagusev/saiga_llama3:latest")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Ответ: {response_text}\из: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()