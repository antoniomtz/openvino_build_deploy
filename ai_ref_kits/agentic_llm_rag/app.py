
import argparse
import io
import logging
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

import gradio as gr
import nest_asyncio
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import requests
import yaml
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
# Agent tools
from tools import PaintCalculator, ShoppingCart
from system_prompt import react_system_header_str

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#Filter unnecessary warnings for demonstration
warnings.filterwarnings("ignore")

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

# def qwen_completion_to_prompt(completion, system_prompt=""):
#     return f"system\n{system_prompt}\nuser\n{completion}\nassistant\n"

def setup_models(llm_model_path, embedding_model_path, device):
    # Load the Llama model locally
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=8192,
        max_new_tokens=500,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},
        #completion_to_prompt=qwen_completion_to_prompt,        
        device_map=device,
    )

    # Load the embedding model locally
    embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device=device)

    return llm, embedding


def setup_tools():

    paint_cost_calculator = FunctionTool.from_defaults(
        fn=PaintCalculator.calculate_paint_cost,
        name="calculate_paint_cost",
        description="ALWAYS use this tool when calculating paint cost for a specific area in square feet. Required inputs: area (float, square feet), price_per_gallon (float), add_paint_supply_costs (bool)"
    )

    paint_gallons_calculator = FunctionTool.from_defaults(
    fn=PaintCalculator.calculate_paint_gallons_needed,
    name="calculate_paint_gallons",
    description="Calculate how many gallons of paint are needed to cover a specific area. Required input: area (float, square feet). Returns the number of gallons needed, rounded up to ensure full coverage."
)

    add_to_cart_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.add_to_cart,
        name="add_to_cart",
        description="""
        Use this tool WHENEVER a user wants to add any item to their cart or shopping cart.
        
        PARAMETERS:
        - product_name (string): The exact name of the product (e.g., "Premium Latex Paint")
        - quantity (int): The number of units to add, must be a positive integer (e.g., 2)
        - price_per_unit (float): The price per unit in dollars (e.g., 24.99)
        
        RETURNS:
        - A confirmation message and updated cart contents
        
        EXAMPLES:
        To add 3 gallons of paint at $29.99 each: add_to_cart(product_name="Interior Eggshell Paint", quantity=3, price_per_unit=29.99)
        """
    )
    
    get_cart_items_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.get_cart_items,
        name="view_cart",
        description="""
        Use this tool when a user wants to see what's in their shopping cart.
        No parameters are required.
        
        RETURNS:
        - A list of all items currently in the cart with their details
        
        EXAMPLES:
        To view the current cart contents: view_cart()
        """
    )
    
    clear_cart_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.clear_cart,
        name="clear_cart",
        description="""
        Use this tool when a user asks to empty or clear their shopping cart.
        No parameters are required.
        
        RETURNS:
        - A confirmation message that the cart has been cleared
        
        EXAMPLES:
        To empty the shopping cart: clear_cart()
        """
    )
    return paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, paint_gallons_calculator


def load_documents(text_example_en_path):
    # Check and download document if not present
    if not text_example_en_path.exists():
        text_example_en = "test_painting_llm_rag.pdf"
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())

    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index

# Custom function to handle reasoning failures
def custom_handle_reasoning_failure(callback_manager, exception):
    return "Hmm...I didn't quite that. Could you please rephrase your question to be simpler?"


def run_app(agent):
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout        

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, "")]

    def update_cart_display():
        cart_items = ShoppingCart.get_cart_items()
        if not cart_items:
            return "### 🛒 Your Shopping Cart is Empty"
            
        table = "### 🛒 Your Shopping Cart\n\n"
        table += "| Product | Qty | Price | Total |\n"
        table += "|---------|-----|-------|-------|\n"
            
        for item in cart_items:
            table += f"| {item['product_name']} | {item['quantity']} | ${item['price_per_unit']:.2f} | ${item['total_price']:.2f} |\n"
            
        total = sum(item["total_price"] for item in cart_items)
        table += f"\n**Total: ${total:.2f}**"
        return table

    def _generate_response(chat_history, log_history):
        log.info(f"log_history {log_history}")
        estimated_tokens = sum(len(msg[0].split()) + len(msg[1].split()) for msg in chat_history) * 1.3
    
        # Add checkpoint counter to track interactions
        # if not hasattr(_generate_response, 'interaction_count'):
        #     _generate_response.interaction_count = 0
        
        # _generate_response.interaction_count += 1
        
        # Force reset every few interactions regardless of estimated token count
        # if _generate_response.interaction_count >= 5:  # Reset after 4 interactions
        #     log.info("Performing preventative agent reset after 4 interactions")
            
        #     # Save important state
        #     current_cart = ShoppingCart.get_cart_items()
            
        #     # Reset agent
        #     agent.reset()
            
        #     # Restore cart
        #     for item in current_cart:
        #         ShoppingCart.add_to_cart(
        #             item["product_name"], 
        #             item["quantity"], 
        #             item["price_per_unit"]
        #         )
            
        #     # Reset counter
        #     _generate_response.interaction_count = 0
        if not isinstance(log_history, list):
            log_history = []

        # Capture time for thought process
        start_thought_time = time.time()

        # Capture the thought process output
        with Capturing() as output:
            try:
                response = agent.stream_chat(chat_history[-1][0])
            except ValueError:
                response = agent.stream_chat(chat_history[-1][0])
        formatted_output = []
        for line in output:
            if "Thought:" in line:
                formatted_output.append("\n🤔 **Thought:**\n" + line.split("Thought:", 1)[1])
            elif "Action:" in line:
                formatted_output.append("\n🔧 **Action:**\n" + line.split("Action:", 1)[1])
            elif "Action Input:" in line:
                formatted_output.append("\n📥 **Input:**\n" + line.split("Action Input:", 1)[1])
            elif "Observation:" in line:
                formatted_output.append("\n📋 **Result:**\n" + line.split("Observation:", 1)[1])
            else:
                formatted_output.append(line)
        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time

        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(formatted_output)
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")
        cart_content = update_cart_display() # update shopping cart
        yield chat_history, "\n".join(log_history), cart_content  # Yield after the thought process time is captured

        # Now capture response generation time
        start_response_time = time.time()

        # Gradually yield the response from the agent to the chat
        # Quick fix for agent occasionally repeating the first word of its repsponse
        last_token = "Dummy Token"
        i = 0
        for token in response.response_gen:
            if i == 0:
                last_token = token
            if i == 1 and token.split()[0] == last_token.split()[0]:
                chat_history[-1][1] += token.split()[1] + " "
            else:
                chat_history[-1][1] += token
            yield chat_history, "\n".join(log_history), cart_content  # Ensure log_history is a string
            if i <= 2: i += 1

        end_response_time = time.time()
        response_time = end_response_time - start_response_time

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s)"

        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history), cart_content  # Join logs into a string for display

    def _reset_chat():
        agent.reset()
        ShoppingCart._cart_items = []  # Also clear the cart
        return "", [], "🤔 Agent's Thought Process", update_cart_display()

    def run():
        custom_css= """
            #agent-steps {
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #f9f9f9;
                margin-top: 10px;
            }
            #shopping-cart {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 12px;
                background-color: #f0f8f0;
                margin-top: 10px;
            }
        """        
        with gr.Blocks(css=custom_css) as demo:
            gr.Markdown("# Smart Retail Assistant 🤖: Agentic LLMs with RAG 💭")
            gr.Markdown("Ask me about paint! 🎨")

            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Paint Purchase Helper",
                    avatar_images=(None, "https://docs.openvino.ai/2024/_static/favicon.ico"),
                    height=400,  # Adjust height as per your preference
                    scale=2  # Set a higher scale value for Chatbot to make it wider
                    #autoscroll=True,  # Enable auto-scrolling for better UX
                )            
                log_window = gr.Markdown(
                        label="🤔 Agent's Thought Process",                                            
                        show_label=True,                        
                        value="🤔 Agent's Thought Process",
                        height=400,                        
                        elem_id="agent-steps"
                )
                cart_display = gr.Markdown(
                    value=update_cart_display(),
                    elem_id="shopping-cart"
                )

            with gr.Row():
                message = gr.Textbox(label="Ask the Paint Expert", scale=4, placeholder="Type your prompt/Question and press Enter")
                clear = gr.ClearButton()

            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False                
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],  # Pass individual components, including log_window
                outputs=[chat_window, log_window, cart_display],  # Update chatbot and log window
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window, cart_display])

            gr.Markdown("------------------------------")            

        demo.launch()

    run()


def main(chat_model: str, embedding_model: str, rag_pdf: str, personality: str, device: str):
    # Load models and embedding based on parsed arguments
    llm, embedding = setup_models(chat_model, embedding_model, device)

    Settings.embed_model = embedding
    Settings.llm = llm

    # Set up tools
    paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, paint_gallons_calculator = setup_tools()
    
    text_example_en_path = Path(rag_pdf)
    index = load_documents(text_example_en_path)
    log.info(f"loading in {index}")
 
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="""            
            Use this tool for ANY question about paint products, recommendations, prices, or technical specifications.
            
            WHEN TO USE:
            - User asks about paint types, brands, or products
            - User needs price information before adding to cart
            - User needs recommendations based on their project
            - User has technical questions about painting
            
            EXAMPLES:
            - "What paint is best for kitchen cabinets?"
            - "How much does AwesomePainter Interior Acrylic Latex cost?"
            - "What supplies do I need for painting my living room?"
            """,
        ),
    )
    
    nest_asyncio.apply()

    # Load agent config
    personality_file_path = Path(personality)

    with open(personality_file_path, "rb") as f:
        chatbot_config = yaml.safe_load(f)    
  
    # Define agent and available tools
    agent = ReActAgent.from_tools(
        [paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, vector_tool, paint_gallons_calculator],
        llm=llm,
        max_iterations=5,  # Set a max_iterations value
        handle_reasoning_failure_fn=custom_handle_reasoning_failure,
        verbose=True,
        react_chat_formatter=ReActChatFormatter.from_defaults(
            # ReactAgent uses a default system prompt, this is just to expand the context
            #context=chatbot_config["system_configuration"],
            observation_role=MessageRole.TOOL          
        ),
    ) 
    react_system_prompt = PromptTemplate(react_system_header_str)
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})  
    agent.reset()                     
    run_app(agent)

if __name__ == "__main__":
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="/home/antonio/agent/openvino_build_deploy/ai_ref_kits/agentic_llm_rag/model/qwen2-7B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="/home/antonio/agent/openvino_build_deploy/ai_ref_kits/agentic_llm_rag/model/bge-large-FP32", help="Path to the embedding model directory")
    parser.add_argument("--rag_pdf", type=str, default="data/test_painting_llm_rag.pdf", help="Path to a RAG PDF file with additional knowledge the chatbot can rely on.")
    parser.add_argument("--personality", type=str, default="config/paint_concierge_personality.yaml", help="Path to the yaml file with chatbot personality")
    parser.add_argument("--device", type=str, default="GPU", help="Device for inferencing (CPU,GPU,GPU.1,NPU)")

    args = parser.parse_args()

    main(args.chat_model, args.embedding_model, args.rag_pdf, args.personality, args.device)
