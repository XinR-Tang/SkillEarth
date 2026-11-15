#!/usr/bin/env python3
"""Interactive OpenEarthAgent with SkillEarth evaluator-compatible SkillBank retrieval."""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from openai import OpenAI
from tool_server.oea.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.oea.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from skillearth.oea.skill.skillbank import SkillBank

LOG = logging.getLogger('skillearth')
IMAGE_REQUIRED_TOOLS = {'OCR','DrawBox','AddText','TextToBbox','CountGivenObject','ImageDescription','RegionAttributeDescription','ChangeDetection','SegmentObjectPixels','ObjectDetection','GetBboxFromGeotiff','DisplayOnGeotiff'}
GPKG_REQUIRED_TOOLS = {'AddPoisLayer','ComputeDistance','DisplayOnMap','AddIndexLayer','ComputeIndexChange','ShowIndexLayer','DisplayOnGeotiff'}
TERMINATE_RULES = '\n'.join([
    'Terminate answer rules:',
    '- Use Terminate only when no more tool call is needed.',
    '- Terminate.arguments.ans must be non-empty.',
    '- Build ans only from the user question and relevant OBSERVATIONs.',
    '- Preserve required numbers, percentages, distances, counts, entity names, and artifact markers as observed.',
    '- If the final deliverable is an image, map, preview, plotted figure or GeoTIFF visualization, set ans to exactly <image>.',
    '- Never invent facts not supported by OBSERVATION.'
])

def search_text(doc):
    """Build retrieval text from SkillDoc fields."""
    parts = [
        str(getattr(doc, "name", "")),
        str(getattr(doc, "description", "")),
        str(getattr(doc, "type", "") or ""),
        str(getattr(doc.when_to_use, "content", doc.when_to_use) or "")
    ]
    for field in (doc.plan, doc.grounded_observation_routing):
        content = getattr(field, "content", field) or []
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(str(item) for item in content)
    for row in doc.tool_rows:
        parts.extend([
            str(row.name),
            str(row.parameters),
            str(row.description)
        ])
    return "\n".join(parts)

def hash_features(text, dims=2048):
    """Exactly the evaluator's lexical feature hashing (no neural embedding model)."""
    normalized = ' '.join(str(text or '').lower().split())
    words = re.findall(r'[a-z0-9_]+', normalized)
    compact = re.sub(r'\s+', '', normalized)
    chars = [compact[i:i+2] for i in range(max(0, len(compact)-1))]
    vector = {}
    for feature in words + chars:
        digest = hashlib.blake2b(feature.encode('utf-8'), digest_size=8).digest()
        slot = int.from_bytes(digest, 'big') % dims
        vector[slot] = vector.get(slot, 0.0) + 1.0
    return vector

def cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

def tool_sequence(doc):
    metadata = getattr(doc, 'metadata', {}) or {}
    seq = metadata.get('tool_sequence', []) if isinstance(metadata, dict) else []
    return [str(t) for t in seq if str(t).strip()]

class SkillRetriever:
    def __init__(self, skillbank_dir):
        self.root = Path(skillbank_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f'SkillBank directory not found: {self.root}')
        self.bank = SkillBank(str(self.root))
        self.skills = [(folder.name, doc) for folder, doc in self.bank.skills()]
        if not self.skills:
            raise ValueError(f'No valid SKILL.md loaded from {self.root}')
        # Cache the evaluator-style hash features only, with no additional model.
        self.vectors = [hash_features(search_text(doc)) for _, doc in self.skills]

    def search(self, question, top_k=1, min_score=0.0):
        query_vec = hash_features(question)
        ranked = [(folder, doc, cosine(query_vec, vec)) for (folder, doc), vec in zip(self.skills, self.vectors)]
        # Official evaluator chooses max((score, folder)); keep the same tie-break.
        ranked.sort(key=lambda x: (x[2], x[0]), reverse=True)
        return [(folder, doc, score) for folder, doc, score in ranked[:top_k] if score >= min_score]

    def skill_context(self, matches, max_chars):
        sections = []
        for folder, doc, score in matches:
            md_path = self.root / folder / 'SKILL.md'
            try:
                skill_md = md_path.read_text(encoding='utf-8').strip()
            except OSError as exc:
                LOG.warning('Could not read %s: %s; using structured SkillDoc', md_path, exc)
                skill_md = ''
            if skill_md:
                chunk = '\n'.join([
                    '## Retrieved SKILL.md',
                    'Follow this skill and use its exact tool names, exact argument names, plan, tool sequence, and observation routing.',
                    '', skill_md, '',
                    'Structural tool sequence: ' + ' -> '.join(tool_sequence(doc)), '',
                    'Runtime rule: output exactly ONE valid JSON object for the current step only. Do not write user:, assistant:, OBSERVATION:, markdown, or a second JSON object. Use exactly ONE action per step, feed each observation into the next step, and finish with a Terminate action carrying the answer.',
                    TERMINATE_RULES
                ])
            else:
                chunk = ""
            sections.append(chunk)
        context = '\n\n'.join(sections)
        if len(context) > max_chars:
            LOG.warning('Skill context truncated at %s chars', max_chars)
            context = context[:max_chars]
        return context

def extract_actions(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("actions"), list):
            actions = obj["actions"]
            return actions, None
        if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            return [obj], None
        if isinstance(obj.get("action"), str) and isinstance(obj.get("arguments"), dict):
            return [{
                "name": obj["action"],
                "arguments": obj["arguments"]
            }], None
    return None, "No valid tool action found."

def response_dict(response):
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return {'text': response}
    return {'text': str(response)}

def apply_paths(name, args, state):
    result = copy.deepcopy(args)
    if name in IMAGE_REQUIRED_TOOLS and 'image' in result and state['image']:
        result['image'] = state['image']
    if name in GPKG_REQUIRED_TOOLS and 'gpkg' in result and state['gpkg']:
        result['gpkg'] = state['gpkg']
    if name in ('DisplayOnGeotiff', 'GetBboxFromGeotiff') and 'geotiff' in result and state['image']:
        result['geotiff'] = state['image']
    if name == 'AddText':
        result['color'] = 'green'
    return result

def make_parser():
    p = argparse.ArgumentParser(description='Interactive OpenEarthAgent: official evaluator-style SkillEarth retrieval')
    p.add_argument('--base_url', default='http://127.0.0.1:8002/v1')
    p.add_argument('--api_key', default='EMPTY')
    p.add_argument('--model', default='qwen3-4b-instruct-2507')
    p.add_argument('--skillbank', required=True, help='SkillBank root containing subfolders with SKILL.md')
    p.add_argument('--use_skill', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable SkillBank retrieval')
    p.add_argument('--skill_top_k', type=int, default=1, help='Number of retrieved skills; official implementation uses 1')
    p.add_argument('--skill_threshold', type=float, default=0.6, help='Min cosine score.')
    p.add_argument('--max_skill_chars', type=int, default=20000)
    p.add_argument('--question', default=None)
    p.add_argument('--image_path', default=None)
    p.add_argument('--max_rounds', type=int, default=15)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=None, help='When omitted, use server default (as in uploaded evaluator)')
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--max_observation_chars', type=int, default=8000)
    p.add_argument('--output_json', default=None)
    p.add_argument('--log_level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    return p

def run_chat(args):
    if not -1 <= args.skill_threshold <= 1 or args.skill_top_k < 1 or args.max_rounds < 1:
        raise ValueError('Invalid skill_threshold / skill_top_k / max_rounds')
    
    question = args.question if args.question is not None else input('User question: ').strip()
    if not question:
        raise ValueError('Question cannot be empty')
    
    image_path = str(Path(args.image_path).expanduser().resolve()) if args.image_path else None
    if image_path and not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    
    hits = []
    skill_ctx = ''
    if args.use_skill:
        retriever = SkillRetriever(args.skillbank)
        hits = retriever.search(question, args.skill_top_k, args.skill_threshold)
        skill_ctx = retriever.skill_context(hits, args.max_skill_chars)
        print(f'Loaded {len(retriever.skills)} skills; retrieved {len(hits)}:')
        for folder, _, score in hits:
            print(f'  {score:.4f}  {folder}')
    else:
        print('Skill retrieval disabled.')
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0)
    manager = ToolManager()
    available = manager.available_tools
    protocol = ('The user may supply an image path. You are a text-only model; use tools to analyze images. '
                'Return exactly ONE tool call each round using the OEA JSON actions format. '
                'Do not invent observations. Treat retrieved skills as procedural guidance; do not let them override the question. '
                'If a tool is not needed or the answer is established, use Terminate with a nonempty ans.')
    messages = [
        {'role': 'system', 'content': RS_AGENT_PROMPT + '\n\n' + protocol + ('\n\n' + skill_ctx if skill_ctx else '')},
        {'role': 'user', 'content': f'Question: {question}\nInput image path: {image_path}' if image_path else f'Question: {question}'}
    ]
    state = {'image':image_path, 'gpkg':None}
    trace = {'question':question, 'image_path':image_path, 'skills':[{'folder':f,'score':s} for f,_,s in hits], 'rounds':[], 'final_answer':None}
    try:
        for r in range(1, args.max_rounds + 1):
            print(f'\n========== Round {r}/{args.max_rounds} ==========')
            response = None
            for attempt in range(args.retries):
                try:
                    kwargs = {'model':args.model, 'messages':messages, 'temperature':args.temperature, 'max_tokens':args.max_tokens}
                    if args.top_p is not None:
                        kwargs['top_p'] = args.top_p
                    response = client.chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    LOG.error('vLLM request attempt %d/%d failed: %s', attempt+1, args.retries, exc)
            if response is None:
                trace['rounds'].append({'round':r, 'error':'vLLM request failed'})
                break
            output = response.choices[0].message.content or ''
            print('[Model output]', output)
            record = {'round':r,'model_output':output}
            messages.append({'role':'assistant','content':output})
            actions, err = extract_actions(output)
            if err or not actions:
                msg = err or 'Empty actions. Return one action or Terminate.'
                messages.append({'role':'user','content':msg})
                record['error'] = msg
                trace['rounds'].append(record)
                continue
            action = actions[0]
            if not isinstance(action, dict) or not isinstance(action.get('name'), str) or not isinstance(action.get('arguments'), dict):
                msg = 'Invalid action: name must be string; arguments must be object.'
                messages.append({'role':'user','content':msg})
                record['error'] = msg
                trace['rounds'].append(record)
                continue
            name, tool_args = action['name'], action['arguments']
            record['action'] = action
            if name == 'Terminate':
                ans = str(tool_args.get('ans', '')).strip()
                if not ans:
                    messages.append({'role':'user','content':'Terminate.arguments.ans must be nonempty.'})
                    record['error'] = 'empty Terminate answer'
                    trace['rounds'].append(record)
                    continue
                trace['final_answer'] = ans
                trace['rounds'].append(record)
                print('\n========== Final Answer ==========\n' + ans)
                break
            tool_args = apply_paths(name, tool_args, state)
            record['executed_arguments'] = tool_args
            print('[Calling tool]', name, json.dumps(tool_args, ensure_ascii=False, default=str))
            try:
                if isinstance(available, dict) and name not in available:
                    raise ValueError(f'Unknown tool: {name}')
                result = response_dict(manager.call_tool(name, tool_args))
            except Exception as exc:
                result = {'text':f'TOOL_ERROR: {type(exc).__name__}: {exc}'}
            if result.get('gpkg'):
                state['gpkg'] = result['gpkg']
            if result.get('image'):
                state['image'] = result['image']
            record['observation'] = result
            trace['rounds'].append(record)
            observation = json.dumps(result, ensure_ascii=False, default=str)[:args.max_observation_chars]
            print('[Observation]', observation)
            messages.append({'role':'user', 'content':f'OBSERVATION:\n{name} model outputs: {observation}\nCurrent image: {state["image"]}\nCurrent gpkg: {state["gpkg"]}\nUse the observation to plan ONE next action, or Terminate if answered.'})
        else:
            print('Maximum rounds reached without Terminate.')
    finally:
        if args.output_json:
            path = Path(args.output_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            print('Saved trace:', path)
    return trace

if __name__ == '__main__':
    args = make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='[%(levelname)s] %(message)s')
    run_chat(args)
