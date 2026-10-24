from src.states.blog_state import BlogState
from langchain_core.messages import SystemMessage, HumanMessage
from src.states.blog_state import Blog

class BlogNode:
    """
    A class to represent the Blog Node.
    """

    def __init__(self, llm):
        self.llm = llm


    def title_creation(self, state:BlogState):
        """
        Create the title of the blog
        """

        if "topic" in state and state["topic"]:
            prompt="""
            You are an expert blog content writer.
            Use Markdown formatting.
            Generate a blog title for the {topic}.
            This title should be creative and SEO friendly.
            """

            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)
            return {"blog":{"title":response.content}}


    def content_generation(self,state:BlogState):
        """
        Generation of the content of blog
        """
        
        if "topic" in state and state["topic"]:
            system_prompt = """
            You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}.
            """

            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}




    def translation(self, state: BlogState):
        """
        Translate both the blog title and content into the target language.
        """

        language = state["current_language"]
        blog = state["blog"]

        title_prompt = f"""
        Translate this blog title into {language}.
        Return only the translated title, without explanations.

        Title: {blog["title"]}
        """

        content_prompt = f"""
        Translate the following blog content into {language}.
        Preserve Markdown formatting, headings, links, and code blocks.
        Return only the translated content.

        Content:
        {blog["content"]}
        """

        translated_title = self.llm.invoke(
            [HumanMessage(content=title_prompt)]
        )

        translated_content = self.llm.invoke(
            [HumanMessage(content=content_prompt)]
        )

        return {
            "blog": {
                "title": translated_title.content.strip(),
                "content": translated_content.content.strip()
            }
        }


    def route(self, state: BlogState):
        return {"current_language": state["current_language"]}


    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """

        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state["current_language"]
