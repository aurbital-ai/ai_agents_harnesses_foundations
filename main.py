# see also https://docs.langchain.com/oss/python/integrations/chat/anthropic

from dotenv import load_dotenv
# import os
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic



load_dotenv()

def main():
    print("Hello, World!")
    # print(os.environ.get("ANTHROPIC_API_KEY"))

    information = """
    Richard Buckminster Fuller Jr. (/ˈfʊlər/; July 12, 1895 – July 1, 1983)[1] was an American architect, systems theorist, writer, designer, inventor, philosopher, and futurist. He styled his name as R. Buckminster Fuller in his writings, publishing more than 30 books and coining or popularizing such terms as "Spaceship Earth", "Dymaxion" (e.g., Dymaxion house, Dymaxion car, Dymaxion map), "ephemeralization", "synergetics", and "tensegrity".
    Fuller developed numerous inventions, mainly architectural designs, and popularized the widely known geodesic dome; carbon molecules known as fullerenes were later named by scientists for their structural and mathematical resemblance to geodesic spheres. He also served as the second World President of Mensa International from 1974 to 1983.[2][3]
    Fuller was awarded 28 United States patents[4] and many honorary doctorates. In 1960, he was awarded the Frank P. Brown Medal from the Franklin Institute. He was elected an honorary member of Phi Beta Kappa in 1967, on the occasion of the 50-year reunion of his Harvard class of 1917 (from which he had been expelled in his first year).[5][6] He was elected a Fellow of the American Academy of Arts and Sciences in 1968.[7] The same year, he was elected into the National Academy of Design as an Associate member. He became a full Academician in 1970, and he received the Gold Medal award from the American Institute of Architects the same year. Also in 1970, Fuller received the title of Master Architect from Alpha Rho Chi (APX), the national fraternity for architecture and the allied arts.[8] In 1976, he received the St. Louis Literary Award from the Saint Louis University Library Associates.[9][10] In 1977, he received the Golden Plate Award of the American Academy of Achievement.[11] He also received numerous other awards, including the Presidential Medal of Freedom, presented to him on February 23, 1983, by President Ronald Reagan.[12]
    """

    summary_template="""
    Given the information {information} about a person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=500)

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()