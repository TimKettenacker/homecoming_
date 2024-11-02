import os
from openai import OpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# let ChatGPT create property listings
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key = "API_KEY"
)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": "Generate csv formatted listings for real estate. Make up 12 diverse properties, one per row"
                 "of the output. Write it like a real estate agent trying to sell the property. The description"
                 "of the property must include a lively depiction of its neighborhood, its price in US Dollars,"
                 "as well as the number of bedrooms, bathrooms and the overall size of the house in sqft."
                 ""
    }
  ],
  temperature=0.0,
  max_tokens=5000
)

# write listings to file for reproducibility
open('findings.csv', 'w', newline='\n').write(response.choices[0].message.content)

# create embeddings for the listings and put into Chroma vector database
loader = CSVLoader(file_path='./findings.csv')
docs = loader.load()

## embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = "OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "API_KEY"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# define function to collect input from the user to specific property-related questions
questions = [
                "How big do you want your house to be?",
                "What are 3 most important things for you in choosing this property?",
                "Which amenities would you like?",
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?"
            ]

def prompt_user(questions):
    answers = []
    for question in questions:
        val = str(input("Please shortly describe: " + "" + question + ""))
        answers.append(val)
    return answers

# now prompt the user interactively
answers = prompt_user(questions)
open('answers.csv', 'w', newline='\n').write(' '.join(answers))

# implement semantic search
query = """
    Find the most suiting properties based on these criteria:
    """ + ' '.join(answers) + """
    Make sure you do not paraphrase, and only use the information provided, do not make up new properties.
    """

# check the closeness of retrieved listing
closest_match = db.similarity_search(query, k=1)

# now include the closest match as well as the preferences in the prompt with further instructions to the model
prompt = PromptTemplate.from_template(
    """Act as a real estate agent. Emphasize the personal preferences expressed in the context below when
    suggesting the property described in {closest_match}. Make sure you do not paraphrase, and only use the information provided, 
    do not make up new properties.
       Context: {context}"""
)

prompt = prompt.format(closest_match=closest_match, context=query)
model = ChatOpenAI()
chain = LLMChain(llm=model, prompt=prompt)
reply = chain.invoke({"closest_match": closest_match[0].page_content, "context": query})

# note the recommendation for reproducibility
open('recommendation.csv', 'w', newline='\n').write(''.join(reply['text']))

if __name__ == '__main__':
    print(reply['text'])

