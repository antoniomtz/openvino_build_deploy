
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

from tools import Math, PaintCostCalculator

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#Filter unnecessary warnings for demonstration
warnings.filterwarnings("ignore")

llm_device = "GPU.1"
embedding_device = "GPU.1"
ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

def qwen_completion_to_prompt(completion):
    return f"<|im_start|>system\nYou are a helpful Paint Concierge assistant.<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def setup_models(llm_model_path, embedding_model_path):
    # Load the Llama model locally
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=4096,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
        completion_to_prompt=qwen_completion_to_prompt,
        device_map=llm_device,
    )

    # Load the embedding model locally
    embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device=embedding_device)

    return llm, embedding


def setup_tools():
    multiply_tool = FunctionTool.from_defaults(fn=Math.multiply)
    divide_tool = FunctionTool.from_defaults(fn=Math.divide)
    add_tool = FunctionTool.from_defaults(fn=Math.add, name="add",
        description="Add two numbers and returns the sum. Input: float1 and float2")
    subtract_tool = FunctionTool.from_defaults(fn=Math.add)
    paint_cost_calculator = FunctionTool.from_defaults(
        fn=PaintCostCalculator.calculate_paint_cost,
        name="calculate_paint_cost",
        description="Calculate paint cost for a given area. Required inputs: area (float, square feet), price_per_gallon (float), add_paint_supply_costs (bool)"
    )
    return multiply_tool, divide_tool, add_tool, subtract_tool, paint_cost_calculator


def load_documents(text_example_en_path):
    # Check and download document if not present
    if not text_example_en_path.exists():
        text_example_en = "test_painting_llm_rag.pdf"  # TBD - Replace with valid URL
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())

    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index


# Function to simulate adding items to cart, and to check size of cart
def purchase_click(cart, *components):
    """Update the cart with unique selected items and their quantities."""
    selected_items = []
    quantities = []

    for i in range(0, len(components), 2):
        item_checkbox = components[i]
        quantity_box = components[i + 1]
        if item_checkbox is True:  # Fix to check if checkbox is selected
            selected_items.append(item_components[i // 2][0].label)
            quantities.append(quantity_box)

    updated_cart = cart.copy()
    for item, quantity in zip(selected_items, quantities):
        if item and quantity > 0:
            item_entry = (item, quantity)
            # Update or add the item with quantity in the cart
            for idx, cart_item in enumerate(updated_cart):
                if cart_item[0] == item:
                    updated_cart[idx] = item_entry
                    break
            else:
                updated_cart.append(item_entry)

    # Calculate the total quantity of items in the cart
    cart_size = sum(quantity for _, quantity in updated_cart)
    # Update purchase action to list items and their quantities
    if selected_items:
        item_details = ", ".join([f"{item} (Quantity: {quantity})" for item, quantity in zip(selected_items, quantities)])
        purchase_action = f"Added the following items to cart: {item_details}."
    else:
        purchase_action = "No items selected."

    return updated_cart, purchase_action, cart_size


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


    def _generate_response(chat_history, log_history):
        log.info(f"log_history {log_history}")
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
                formatted_output.append("\n🤔 Thought:\n" + line.split("Thought:", 1)[1])
            elif "Action:" in line:
                formatted_output.append("\n🔧 Action:\n" + line.split("Action:", 1)[1])
            elif "Action Input:" in line:
                formatted_output.append("\n📥 Input:\n" + line.split("Action Input:", 1)[1])
            elif "Observation:" in line:
                formatted_output.append("\n📋 Result:\n" + line.split("Observation:", 1)[1])
            else:
                formatted_output.append(line)
        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time

        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(formatted_output)
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")

        yield chat_history, "\n".join(log_history)  # Yield after the thought process time is captured

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
            yield chat_history, "\n".join(log_history)  # Ensure log_history is a string
            if i <= 2: i += 1

        end_response_time = time.time()
        response_time = end_response_time - start_response_time

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s on {llm_device})"

        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history)  # Join logs into a string for display

    def _reset_chat():
        agent.reset()
        return "", [], []  # Reset both chat and logs (initialize log as empty list)

    def run():
        custom_css= """
            #agent-steps {
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #f9f9f9;
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
            with gr.Row():
                message = gr.Textbox(label="Ask the Paint Expert", scale=4, placeholder="Type your prompt/Question and press Enter")
                clear = gr.ClearButton()

            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False,
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],  # Pass individual components, including log_window
                outputs=[chat_window, log_window],  # Update chatbot and log window
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window])

            gr.Markdown("------------------------------")
            gr.Markdown("### Purchase items")

            cart = gr.State([])

            # Define items with checkbox and numeric quantity
            items = ["Behr Premium Plus", "AwesomeSplash", "TheBrush", "PaintFinish"]

            global item_components
            item_components = []
            for item in items:
                with gr.Row(equal_height=True):
                    item_checkbox = gr.Checkbox(label=f"{item}", value=False)
                    quantity_box = gr.Number(
                        label=f"{item} Quantity", value=1, precision=0, interactive=False, minimum=1
                    )
                    item_checkbox.change(
                        fn=lambda selected, box=quantity_box: gr.update(interactive=selected, value=1 if selected else 0),
                        inputs=item_checkbox,
                        outputs=quantity_box,
                    )
                    item_components.append((item_checkbox, quantity_box))

            purchase = gr.Button(value="Add to Cart")
            cart_size = gr.Number(label="Cart Size", interactive=False)
            purchased_textbox = gr.Textbox(label="Purchase Action", interactive=False)
            # Gather inputs from all item checkbox and number box pairs
            component_inputs = [cart] + [comp for pair in item_components for comp in pair]
            purchase.click(fn=purchase_click, inputs=component_inputs, outputs=[cart, purchased_textbox, cart_size])

        demo.launch()

    run()


def main(chat_model: str, embedding_model: str, rag_pdf: str, personality: str):
    # Load models and embedding based on parsed arguments
    llm, embedding = setup_models(chat_model, embedding_model)

    Settings.embed_model = embedding
    Settings.llm = llm

    # Set up tools
    multiply_tool, divide_tool, add_tool, subtract_tool, paint_cost_calculator = setup_tools()

    # Step 4: Load documents and create the VectorStoreIndex
    text_example_en_path = Path(rag_pdf)
    index = load_documents(text_example_en_path)
    log.info(f"loading in {index}")
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="Use this first for any questions about paint products or recommendations",
        ),
    )

    # Step 5: Initialize the agent with the loaded tools
    nest_asyncio.apply()

    # Load agent config
    personality_file_path = Path(personality)

    with open(personality_file_path, "rb") as f:
        chatbot_config = yaml.safe_load(f)

    react_system_prompt = PromptTemplate(chatbot_config['system_configuration'])
    log.info(f"react_system_prompt {react_system_prompt}")

    react_prompt = PromptTemplate("""
        When responding, STRICTLY follow this format:

    1. For questions about paint quantity:
       Thought: I need the room size in square feet to calculate paint needed
       If size unknown: Ask for room dimensions (length and width in feet)
       If size known: 
           Action: calculate_paint_cost
           Action Input: [room size in square feet]
           Observation: [tool output]
           Final Answer: Based on the calculations...

    2. For product recommendations:
       Thought: Do I have enough details about [room type, color, finish]?
       If no: Ask for missing details
       If yes:
           Action: vector_search
           Action Input: [specific search criteria]
           Observation: [tool output]
           Final Answer: Based on your requirements...

    3. For questions about paint colors or finishes:
       Action: vector_search
       Action Input: [specific color/finish query]
       Observation: [tool output]
       Final Answer: Based on our product database...

    IMPORTANT:
    - NEVER calculate paint quantity without room dimensions
    - ALWAYS ask for missing information before using tools
    - If a question requires calculations, first confirm you have all needed measurements
    - Keep responses focused and directly address the user's question
    """)

    #combined_prompt = PromptTemplate(f"{react_system_prompt}\n\n{react_prompt}\n\nBegin now:\n{{query}}")
    combined_prompt = PromptTemplate(f"{react_system_prompt}\n\nBegin now:\n{{query}}")
    log.info(f"Using combined prompt: {combined_prompt}")
    # Define agent and available tools
    agent = ReActAgent.from_tools(
        [multiply_tool, divide_tool, add_tool, subtract_tool, paint_cost_calculator, vector_tool],
        llm=llm,
        max_iterations=5,  # Set a max_iterations value
        handle_reasoning_failure_fn=custom_handle_reasoning_failure,
        verbose=True,
        system_prompt=combined_prompt
        )

    tool_prompts = {
        "task_decomposition_prompt": "To answer this question, I must: 1. Use vector_search to find information, 2. Use calculation tools if needed",
        "observation_prompt": "Based ONLY on the tool's output above, without adding any information:",
        "solution_prompt": "Using ONLY the information from the tools, I can now answer that:"
    }
    
    agent.update_prompts(tool_prompts)

    # Step 6: Run the app
    run_app(agent)


if __name__ == "__main__":
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/qwen2-7B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-large-FP32", help="Path to the embedding model directory")
    parser.add_argument("--rag_pdf", type=str, default="data/test_painting_llm_rag.pdf", help="Path to a RAG PDF file with additional knowledge the chatbot can rely on.")
    parser.add_argument("--personality", type=str, default="config/paint_concierge_personality.yaml", help="Path to the yaml file with chatbot personality")

    args = parser.parse_args()

    main(args.chat_model, args.embedding_model, args.rag_pdf, args.personality)
