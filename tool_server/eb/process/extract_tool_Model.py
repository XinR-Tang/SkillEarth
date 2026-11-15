#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import argparse
from pathlib import Path


def extract_tool_calls(log_file_path, output_file_path, expected_question_ids_file=None):
    """
    Extract tool calls from log file for each question
    
    Args:
        log_file_path: Input log file path  
        output_file_path: Output JSON file path
    """
    log_path = Path(log_file_path)
    output_path = Path(output_file_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Trajectory log not found: {log_path}")

    extracted_data = []
    
    with log_path.open('r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by question records using regex to find question_index patterns
    import re
    # Find all positions where a new question starts (including error records)
    pattern_normal = r'\{\s*"question_index":\s*"(\d+)"'
    pattern_error = r'\{\s*"question_index":\s*\[\]'
    
    # Find all normal question records
    normal_matches = [(m.start(), m.group(1)) for m in re.finditer(pattern_normal, content)]
    # Find all error records  
    error_matches = [(m.start(), None) for m in re.finditer(pattern_error, content)]
    
    # Combine and sort all matches by position
    all_matches = normal_matches + error_matches
    all_matches.sort(key=lambda x: x[0])
    
    records = []
    for i, (start_pos, question_num) in enumerate(all_matches):
        if i + 1 < len(all_matches):
            end_pos = all_matches[i + 1][0]
            record_content = content[start_pos:end_pos].strip()
        else:
            record_content = content[start_pos:].strip()
        
        # Remove any trailing content after the last }
        # Find the position of the last }
        brace_count = 0
        end_idx = len(record_content)
        for j, char in enumerate(record_content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = j + 1
                    break
        
        if end_idx < len(record_content):
            record_content = record_content[:end_idx]
            
        records.append((record_content, question_num))
    
    for record_content, expected_question_num in records:
        try:
            # Parse the JSON record
            data = json.loads(record_content)
            question_index = data.get('question_index', '')
            
            # Handle error records with empty question_index
            if question_index == [] or question_index == "":
                conversations = data.get('conversations', '')
                # Extract question number from error message
                if isinstance(conversations, str) and 'Error processing question' in conversations:
                    match = re.search(r'Error processing question (\d+):', conversations)
                    if match:
                        question_index = match.group(1)
                        
                # Create error record
                if question_index:
                    question_data = {
                        "question_index": question_index,
                        "query": f"Error processing question {question_index}: {conversations}",
                        "tool_calls": [],
                        "error": True,
                        "timestamp": data.get('timestamp', '')
                    }
                    extracted_data.append(question_data)
                continue
                
            conversations = data.get('conversations', [])
            
            # Extract user query and tool calls
            query = ""
            tool_calls = []
            
            # Process conversations sequentially to maintain order
            for conv in conversations:
                if isinstance(conv, dict):
                    role = conv.get('role', '')
                    
                    # Extract user query
                    if role == 'user':
                        query = conv.get('content', '')
                    
                    # Extract assistant tool calls
                    elif role == 'assistant':
                        content = conv.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'name' in item and 'input' in item:
                                    # Store assistant call info temporarily
                                    tool_calls.append({
                                        'name': item.get('name', ''),
                                        'input': item.get('input', {}),
                                        'output': None  # Will be filled by corresponding tool response
                                    })
                    
                    # Extract tool outputs
                    elif role == 'tool':
                        tool_name = conv.get('name', '')
                        content = conv.get('content', [])
                        
                        # Find the corresponding assistant call
                        for tool_call in reversed(tool_calls):
                            if tool_call['name'] == tool_name and tool_call['output'] is None:
                                # Extract output text from tool response
                                if isinstance(content, list) and len(content) > 0:
                                    output_item = content[0]
                                    if isinstance(output_item, dict) and 'output' in output_item:
                                        output_list = output_item['output']
                                        if isinstance(output_list, list) and len(output_list) > 0:
                                            text_item = output_list[0]
                                            if isinstance(text_item, dict) and 'text' in text_item:
                                                tool_call['output'] = text_item['text']
                                break
            
            # Filter out tool calls that have both input and output
            completed_tool_calls = [tc for tc in tool_calls if tc['output'] is not None]
            
            # Create result data structure - include all questions even without tool calls
            if query:
                question_data = {
                    "question_index": question_index,
                    "query": query,
                    "tool_calls": completed_tool_calls  # Will be empty list if no tool calls
                }
                extracted_data.append(question_data)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing record: {e}")
            continue
    
    # Keep the prediction file aligned with the evaluation split.  A malformed
    # trajectory must be scored as an empty prediction rather than silently
    # removing that test item from the denominator.
    if expected_question_ids_file:
        expected_path = Path(expected_question_ids_file)
        with expected_path.open("r", encoding="utf-8") as handle:
            expected_data = json.load(handle)

        if isinstance(expected_data, dict):
            expected_ids = [str(question_id) for question_id in expected_data]
        elif isinstance(expected_data, list):
            expected_ids = [
                str(item.get("question_index", item.get("question_id", "")))
                for item in expected_data
                if isinstance(item, dict)
            ]
        else:
            raise ValueError(f"Unsupported expected-ID file format: {expected_path}")

        existing_ids = {str(item.get("question_index", "")) for item in extracted_data}
        missing_ids = [question_id for question_id in expected_ids if question_id not in existing_ids]
        for question_id in missing_ids:
            extracted_data.append({
                "question_index": question_id,
                "query": "",
                "tool_calls": [],
                "error": True,
                "error_message": "Trajectory record could not be parsed from the execution log.",
            })
        extracted_data.sort(
            key=lambda item: int(item["question_index"])
            if str(item.get("question_index", "")).isdigit()
            else float("inf")
        )
        print(
            f"Aligned predictions to {len(expected_ids)} expected questions "
            f"({len(missing_ids)} empty predictions added)."
        )

    # Save extracted data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully extracted tool calls from {len(extracted_data)} questions")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tool calls from LangChain execution logs.")
    parser.add_argument("--log_file", type=str, default="qwen3-4b-instruct-2507_IF_langchain.log", help="Path to the input LangChain execution log file.")
    parser.add_argument("--output_file", type=str, default="extracted_tool_calls.json", help="Path to the output JSON file containing extracted tool calls.")
    parser.add_argument("--expected_question_ids_file", type=str, default=None, help="Optional split question JSON; missing trajectories are emitted as empty predictions.")
    args = parser.parse_args()

    extract_tool_calls(args.log_file, args.output_file, args.expected_question_ids_file)
