import os
import argparse
os.environ["GTIFF_SRS_SOURCE"]="EPSG"
import json
import logging
import asyncio
from enum import auto
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from logging.handlers import RotatingFileHandler
import hashlib
import math
import re

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Pprint for debugging
from pprint import pprint

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global variables
logger = None
temp_dir_path = None

# Configuration
model_name = 'qwen3-4b-instruct-2507'
sys_prompt = '''
You are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once.

ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide only the choice you think is most appropriate.

Your final answer must use exactly this format:

<Answer>Your choice</Answer>

Do not output any other text after the final answer.
'''

SKILLBANK_DIR = os.environ.get(
    "SKILLBANK_DIR",
    "skillbank-eb/qwen3-4b-instruct-2507"
)
EARTHBENCH_ROOT = os.environ.get(
    "EARTHBENCH_ROOT",
    "SkillEarth/datasets/eb/Earth-Bench"
)
SKILL_SCORE_THRESHOLD = 0.7

LLM_REQUEST_TIMEOUT = 600   # 单次模型请求：10 分钟
QUESTION_TIMEOUT = 900      # 整题总预算：15 分钟

def _features(text: str):
    """
    Same retrieval feature construction as the reference AutoSkill implementation.
    """
    normalized = " ".join(str(text or "").lower().split())

    # word features
    words = re.findall(r"[a-z0-9_]+", normalized)

    # character bigram features
    compact = re.sub(r"\s+", "", normalized)
    chars = [
        compact[i:i + 2]
        for i in range(max(0, len(compact) - 1))
    ]

    return words + chars

def _hash_features(text: str, dims: int = 2048):
    """
    Convert text into a sparse hashed feature vector.
    """
    vector = {}

    for feature in _features(text):
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=8
        ).digest()

        slot = int.from_bytes(digest, "big") % dims
        vector[slot] = vector.get(slot, 0.0) + 1.0

    return vector

def _cosine(left, right):
    """
    Cosine similarity between two sparse vectors.
    """
    if not left or not right:
        return 0.0

    dot = sum(
        value * right.get(key, 0.0)
        for key, value in left.items()
    )

    left_norm = math.sqrt(
        sum(value * value for value in left.values())
    )

    right_norm = math.sqrt(
        sum(value * value for value in right.values())
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)

def load_skillbank(skillbank_dir: str):
    """
    Load all SKILL.md files once before evaluation.
    """
    skillbank_path = Path(skillbank_dir)

    if not skillbank_path.exists():
        raise FileNotFoundError(
            f"Skillbank does not exist: {skillbank_path}"
        )

    skills = []

    for folder in sorted(skillbank_path.iterdir()):
        if not folder.is_dir():
            continue

        # ignore auxiliary entries such as _structure.json etc.
        if folder.name.startswith("_"):
            continue

        skill_path = folder / "SKILL.md"

        if not skill_path.is_file():
            continue

        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        retrieval_text = extract_skill_retrieval_text(content, folder.name)

        skills.append({
            "name": folder.name,
            "path": str(skill_path),
            "content": content,
            "retrieval_text": retrieval_text,
            # Compute this only once.
            "vector": _hash_features(retrieval_text),
        })

    if not skills:
        raise RuntimeError(
            f"No SKILL.md found under: {skillbank_path}"
        )

    print(
        f"Successfully loaded {len(skills)} skills "
        f"from {skillbank_path}"
    )

    return skills

def retrieve_skill(question: str, skills):
    """
    Retrieve exactly one skill using cosine similarity.
    """
    query_vector = _hash_features(question)

    scored = [
        (
            skill,
            _cosine(query_vector, skill["vector"])
        )
        for skill in skills
    ]

    best_skill, best_score = max(
        scored,
        key=lambda item: (
            item[1],
            item[0]["name"]
        )
    )

    return best_skill, best_score

def build_skill_context(skill):
    """
    Inject the complete retrieved SKILL.md into the question context.
    """
    return f"""
    ## Retrieved Skill

    The following skill is retrieved from the skill bank for the current task.

    Follow the skill when solving the question.
    Use its workflow, tool names, argument requirements, and observation dependencies.
    Do not copy sample-specific values from the skill; use the values from the current question and tool observations.

    Skill name: {skill["name"]}

    {skill["content"]}
    """.strip()

def extract_skill_retrieval_text(content: str, folder_name: str) -> str:
    """
    Build retrieval text using only:
    1. skill name
    2. description
    3. when-to-use
    """
    # 1. name
    name = folder_name

    match = re.search(r"(?mi)^name\s*:\s*(.+?)\s*$", content)
    if match:
        name = match.group(1).strip().strip('"').strip("'")

    # 2. description
    description = ""
    match = re.search(r"(?mi)^description\s*:\s*(.+?)\s*$", content)
    if match:
        description = match.group(1).strip().strip('"').strip("'")

    # 3. when-to-use
    when_to_use = ""
    match = re.search(
        r"(?ims)^##\s*when[\s-]*to[\s-]*use\s*\n"
        r"(.*?)(?=^##\s|\Z)",
        content,
    )
    if match:
        when_to_use = match.group(1).strip()

    # Retrieval text
    return "\n".join(part for part in [name, description, when_to_use] if part)

def init_global_params(args):
    """Initialize global parameters and logging"""
    global temp_dir_path, logger
    
    if temp_dir_path is None:
        if args.output_dir:
            temp_dir_path = Path(args.output_dir).expanduser().resolve()
        else:
            temp_dir_path = Path('./evaluate_langchain/{}_{}_{}'.format(
                model_name,
                'AP' if autoplanning else "IF",
                args.method_name,
            )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Simplified logging compatible with original format
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_{}_langchain.log".format(
            model_name, 'AP' if autoplanning else "IF"
        )
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    
    return temp_dir_path, logger

def init_chat_logger():
    """Initialize chat logger for .chat file like AgentScope"""
    global temp_dir_path
    chat_log_path = temp_dir_path / "{}_{}_langchain.chat".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    return chat_log_path

def save_chat_message(chat_log_path, message_data):
    """Save a single chat message to .chat file in AgentScope format"""
    import time
    from datetime import datetime
    import uuid
    
    # Format message in AgentScope style
    chat_record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage", 
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'langchain_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": message_data.get('content', []),
        "metadata": message_data.get('metadata', None),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Append to chat file (one JSON per line, like AgentScope)
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')

def load_langchain_config(config_path='./config/config_qwen3.json'):
    """Load configuration and initialize LangChain components"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize OpenAI-compatible model
    model_config = config['models'][0]
    llm_kwargs = {
        'model': model_config['model_name'],
        'api_key': model_config['api_key'],
        'base_url': model_config['client_args']['base_url'],
        'temperature': 0.1,
        'timeout': LLM_REQUEST_TIMEOUT,
        'max_retries': 2,
    }

    # Add generate_args via extra_body if present in config
    if 'generate_args' in model_config:
        llm_kwargs['extra_body'] = model_config['generate_args']

    llm = ChatOpenAI(**llm_kwargs)

    # Prepare MCP server configuration
    mcp_servers = {}

    for server_name, server_config in config['mcpServers'].items():
        transport = server_config.get('transport', 'stdio')

        if transport in ('http', 'streamable_http'):
            # MCP servers are already running externally.
            mcp_servers[server_name] = {
                "transport": transport,
                "url": server_config['url'],
            }

        elif transport == 'stdio':
            # Backward compatibility with old local-process configuration.
            updated_args = []

            for arg in server_config.get('args', []):
                if 'tmp/tmp/out' in arg:
                    updated_args.append(str(temp_dir_path / 'out'))
                elif arg.startswith('tools/'):
                    updated_args.append('./' + arg)
                else:
                    updated_args.append(arg)

            mcp_servers[server_name] = {
                "command": server_config['command'],
                "args": updated_args,
                "transport": "stdio",
            }

        else:
            raise ValueError(
                f"Unsupported MCP transport for {server_name}: {transport}"
            )

    return llm, mcp_servers

async def create_langchain_agent(llm, mcp_servers):
    """Create LangChain ReAct agent with MCP tools"""
    # Create MCP client
    client = MultiServerMCPClient(mcp_servers)
    
    try:
        # Get tools from all MCP servers
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} tools from MCP servers")
        
        # Create ReAct agent
        agent = create_react_agent(llm, tools)
        
        return agent, client
    except Exception as e:
        print(f"Error creating agent: {e}")
        if hasattr(client, 'close'):
            await client.close()
        raise

def load_questions(test_json_path: str = 'SkillEarth/datasets/eb/Earth-Bench/question.json'):
    """Load evaluation questions"""
    with open(test_json_path, 'r') as f:
        test_json = json.load(f)

    out = []
    for _, (question_idx, question_info) in enumerate(test_json.items()):
        AP_INDEX = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
        data = question_info['evaluation'][AP_INDEX].get('data', None)
        data = question_info['evaluation'][1 - AP_INDEX].get('data', None) if data is None else data

        if data is None:
            continue
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = (EARTHBENCH_ROOT / data_path).resolve()
        out.append({
            "question_id": question_idx,
            "auto": question_info['evaluation'][AP_INDEX]['question'],
            "instruct": question_info['evaluation'][1 - AP_INDEX]['question'],
            "data": str(data_path),
            "choices": question_info.get('choices', None)
        })

    return out

def extract_answer_from_response(response):
    """Extract final answer from agent response"""
    messages = response.get("messages", [])
    
    # Look for the final answer in the last assistant message
    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == 'ai':
            content = message.content
            if '<Answer>' in content and '</Answer>' in content:
                # Extract answer between tags
                start = content.find('<Answer>') + len('<Answer>')
                end = content.find('</Answer>')
                return content[start:end].strip()
            return content
    
    return "No answer found"

def extract_message_tool_calls(message):
    """Normalize LangChain and OpenAI-compatible tool-call payloads for logs."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        tool_calls = getattr(message, "additional_kwargs", {}).get("tool_calls", [])

    normalized_calls = []
    for index, tool_call in enumerate(tool_calls or []):
        if not isinstance(tool_call, dict):
            continue

        # LangChain normalized: {"name": ..., "args": ..., "id": ...}
        # Legacy OpenAI: {"function": {"name": ..., "arguments": ...}}
        function = tool_call.get("function") or {}
        name = tool_call.get("name") or function.get("name")
        arguments = tool_call.get("args", tool_call.get("arguments"))
        if arguments is None:
            arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Retain malformed arguments so the call is not silently dropped.
                pass

        if name:
            normalized_calls.append({
                "id": tool_call.get("id", f"tool_call_{index}"),
                "name": name,
                "input": arguments,
            })

    return normalized_calls

async def handle_question(agent, question, chat_log_path, skills):
    """Handle a single question with the LangChain agent"""
    try:
        # 1. Build retrieval query
        retrieval_query = (question["auto"] if autoplanning else question["instruct"])
        retrieved_skill, retrieval_score = retrieve_skill(retrieval_query, skills)

        use_retrieved_skill = retrieval_score > SKILL_SCORE_THRESHOLD
        if use_retrieved_skill:
            skill_context = build_skill_context(retrieved_skill)
            print(
                f"[Skill Retrieval] "
                f"question={question['question_id']} | "
                f"skill={retrieved_skill['name']} | "
                f"cosine={retrieval_score:.4f}"
            )
        else:
            skill_context = ""
            print(
                f"[Skill Retrieval] "
                f"question={question['question_id']} | "
                f"best_skill={retrieved_skill['name']} | "
                f"cosine={retrieval_score:.4f} | "
                f"used=False"
            )

        # Prepare query
        query = question['auto'] + question['data'] if autoplanning else \
            question['instruct'] + question['data']

        if question['choices']:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice) 
                for i, choice in enumerate(question['choices'])
            ])

        # Add system prompt
        if use_retrieved_skill:
            full_query = (
                f"{sys_prompt}\n\n"
                f"{skill_context}\n\n"
                f"Question: {query}"
            )
        else:
            full_query = (
                f"{sys_prompt}\n\n"
                f"Question: {query}"
            )
        
        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")
        
        # Save user message to chat log
        user_message = {
            "name": "user",
            "role": "user",
            "content": full_query,
            "metadata": {
                "question_id": question["question_id"],
                "retrieved_skill": retrieved_skill["name"],
                "retrieval_score": retrieval_score,
                "skill_used": use_retrieved_skill,
            }
        }
        save_chat_message(chat_log_path, user_message)
        
        # Invoke agent with configuration to prevent infinite loops
        try:
            response = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=full_query)]},
                    config={"recursion_limit": 50},
                ),
                timeout=QUESTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return "FAIL: question execution exceeded 900 seconds"
        
        # Extract final answer
        final_answer = extract_answer_from_response(response)
        
        # Convert response to AgentScope-compatible format for logging
        conversation_log = []
        
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # User message
                    conversation_log.append({
                        "role": "user",
                        "content": message.content
                    })
                
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls
                    assistant_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    if message.content and message.content.strip():
                        assistant_content.append({
                            "type": "text",
                            "content": message.content
                        })
                    
                    # Then check for tool calls
                    # Current LangChain stores normalized calls in ``message.tool_calls``.
                    # Fall back to ``additional_kwargs`` for older providers.
                    for tool_call in extract_message_tool_calls(message):
                        assistant_content.append({
                            "name": tool_call["name"],
                            "input": tool_call["input"],
                        })
                    
                    # Only add to log if there's actual content
                    if assistant_content:
                        conversation_log.append({
                            "role": "assistant", 
                            "content": assistant_content
                        })
                
                elif message.type == 'tool':
                    # Tool result message in AgentScope format
                    conversation_log.append({
                        "role": "tool",
                        "name": message.name,
                        "content": [{
                            "output": [{
                                "text": str(message.content),
                            }],
                        }]
                    })
        
        # Save detailed messages to .chat file (AgentScope format)
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # Skip user message for .chat as it's already saved
                    continue
        
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls for .chat file
                    assistant_chat_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    if message.content and message.content.strip():
                        assistant_chat_content.append({
                            "type": "text",
                            "text": message.content
                        })
                    
                    # Then check for tool calls
                    # Use the same normalized records in the .chat file.
                    for tool_call in extract_message_tool_calls(message):
                        assistant_chat_content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["input"],
                        })
                    
                    # Save assistant message with both thinking and tool calls
                    if assistant_chat_content:
                        assistant_message = {
                            "name": question['question_id'],
                            "role": "assistant",
                            "content": assistant_chat_content,
                            "metadata": None
                        }
                        save_chat_message(chat_log_path, assistant_message)
                
                elif message.type == 'tool':
                    # Tool result message
                    tool_result_message = {
                        "name": "system",
                        "role": "system",
                        "content": [{
                            "type": "tool_result",
                            "id": getattr(message, 'tool_call_id', 'unknown'),
                            "output": [{"type": "text", "text": str(message.content), "annotations": None, "meta": None}],
                            "name": message.name
                        }],
                        "metadata": None
                    }
                    save_chat_message(chat_log_path, tool_result_message)
        
        # Log the conversation in the same format as original code
        logger.info("Chat Content", question['question_id'], conversation_log, final_answer)
        
        print(f"Final Answer: {final_answer}")
        return final_answer
        
    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)
        
        # Save error to chat log
        error_message = {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, error_message)
        
        logger.info(question['question_id'], [], error_msg)
        return f"Error: {e}"

async def main(args):
    """Main evaluation function"""
    print("Initializing LangChain-based Earth Science Agent...")

    global autoplanning
    autoplanning = args.autoplanning

    print(f"Evaluation mode: {'AP' if autoplanning else 'IF'}")

    # Initialize global parameters
    init_global_params(args)
    
    # Initialize chat logger
    chat_log_path = init_chat_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    
    # Load configuration and create agent
    llm, mcp_servers = load_langchain_config(args.config_path)
    agent, client = await create_langchain_agent(llm, mcp_servers)

    # Load skillbank once.
    skills = load_skillbank(SKILLBANK_DIR)
    try:
        # Load questions
        questions = load_questions()[0:args.num_questions]  # First num_questions questions for testing
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Process questions
        results = []
        for question in tqdm(questions, desc="Processing questions"):
            answer = await handle_question(agent, question, chat_log_path, skills)
            results.append({
                "question_id": question['question_id'],
                "answer": answer
            })
            
            # Optional: Add delay between questions to avoid rate limiting
            await asyncio.sleep(1)
        
        # Save results summary
        results_path = temp_dir_path / "results_summary.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\nEvaluation completed! Results saved to {results_path}")
        print(f"Detailed logs available at: {temp_dir_path}")
        print(f"Chat history saved to: {chat_log_path}")
        
    except Exception as e:
        print(f"Error in main evaluation: {e}")
        raise
    
    finally:
        # Clean up
        if hasattr(client, 'close'):
            await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangChain-based Earth evaluation")
    parser.add_argument("--num_questions", type=int, default=249, help="Number of questions to evaluate")
    parser.add_argument("--method_name", type=str, default="skillearth", help="Method name for logging and output directory")
    parser.add_argument("--config_path", type=str, default="./config/config_qwen3.json", help="Path to LangChain configuration JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for logs and inference results")
    parser.add_argument(
        "--autoplanning",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable autonomous planning (True/AP) or instruction following (False/IF)."
    )
    args = parser.parse_args()

    asyncio.run(main(args))
    
