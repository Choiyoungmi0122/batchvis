from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
from datetime import datetime
import openai
from dotenv import load_dotenv
import glob
import pytz

app = FastAPI()


origins = [
    # "http://127.0.0.1:5173",    # 또는 
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 디버깅: API 키 로드 확인
print(f"OpenAI API Key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
if OPENAI_API_KEY:
    print(f"API Key starts with: {OPENAI_API_KEY[:10]}...")
else:
    print("WARNING: OpenAI API Key not found in .env file!")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

RESULTS_DIR = "responses"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate(trait: str = Form(...), experiment_num: str = Form(...)):
    prompt = f"Trait 조합: {trait}\n실험 번호: {experiment_num}\nGPT에게 질문하세요."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            n=3,
            max_tokens=300,
            temperature=0.7,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    result = {
        "experiment_num": experiment_num,
        "trait": trait,
        "prompt": prompt,
        "responses": [choice.message["content"] for choice in response.choices],
        "timestamp": datetime.now().isoformat(),
    }

    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(content={"message": "성공적으로 저장했습니다.", "filename": filename, "responses": result["responses"]})

@app.post("/generate_virtual_patient")
async def generate_virtual_patient(data: dict = Body(...)):
    """
    data = {
        "prompt": instruction 프롬프트,
        "model": (optional, default: gpt-4o)
    }
    """
    prompt = data.get("prompt")
    model = data.get("model", "gpt-4o")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 환자 인물 생성 전문가야."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        return {"result": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask_patient")
async def ask_patient(data: dict = Body(...)):
    """
    data = {
        "instruction": 챗봇 instruction (가상환자 역할),
        "question": 질문,
        "model": (optional, default: gpt-4o)
    }
    """
    context = data.get("context")
    instruction = data.get("instruction")
    question = data.get("question")
    model = data.get("model", "gpt-4o")
    prompt = f"{context}\nQ: {question}\nA:"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        answer = response.choices[0].message.content
        return {"result": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/save_experiment")
async def save_experiment(data: dict = Body(...)):
    """
    프론트엔드에서 실험 완료 시 전체 실험 데이터를 JSON으로 받아 저장합니다.
    파일명: experiment_{experiment_num}_{timestamp}.json
    """
    experiment_num = data.get("experiment_num", "unknown")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"message": "실험 데이터가 저장되었습니다.", "filename": filename}

@app.post("/start_experiment")
async def start_experiment():
    """
    한국 시간 기준으로 날짜와 시간을 포함한 실험 번호를 생성
    형식: YYYYMMDD_HHMMSS (예: 20250720_143052)
    """
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_time = datetime.now(korea_tz)
    
    # 실험 번호 생성 (YYYYMMDD_HHMMSS 형식)
    experiment_num = korea_time.strftime('%Y%m%d_%H%M%S')
    
    return {"experiment_num": experiment_num}

@app.get("/get_experiment/{experiment_num}")
async def get_experiment(experiment_num: str):
    """
    특정 실험 번호의 데이터를 반환
    """
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}.json")
    if not os.path.exists(filename):
        return JSONResponse(status_code=404, content={"error": "Experiment not found"})
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/generate_instructions")
async def generate_instructions(data: dict = Body(...)):
    """
    사용자 입력을 받아 instruction들을 단계별로 생성
    data = {
        "user_input": "홍길동, 24세, 남성, 우울증"
    }
    """
    user_input = data.get("user_input")
    
    # 1. personality.json 읽기
    personality_file = os.path.join("responses", "personality.json")
    with open(personality_file, "r", encoding="utf-8") as f:
        personality_data = json.load(f)
    
    # 2. 조합별 virtual_patient_prompt 생성
    instructions = []
    
    for trait in personality_data:
        type_name = "personality" if trait["type"] == "temperament" else "character"
        detail = trait["detail"]
        
        # 챗봇 instruction 형태로 가상환자 프롬프트 생성
        if type_name == "personality":
            # user_input 파싱
            input_parts = user_input.split(',')
            name = input_parts[0].strip() if len(input_parts) > 0 else ''
            age = input_parts[1].replace('년생','').strip() if len(input_parts) > 1 else ''
            gender = input_parts[2].strip() if len(input_parts) > 2 else ''
            symptom = input_parts[3].strip() if len(input_parts) > 3 else ''
            
            virtual_prompt = f"""당신은 다음 조건을 가진 가상환자입니다. 이 역할을 완전히 수행해주세요.

환자 정보:
- 이름: {name}
- 나이: {age}
- 성별: {gender}
- 주소증(주 증상): {symptom}
- TCI 성향:
  - 기질(Temperament): 
    - 자극추구: {detail['자극추구']}
    - 위험회피: {detail['위험회피']}
    - 자율성: {detail['자율성']}
  - 성격(Character): 
    - 자율성: [-]
    - 연대감: [-]
    - 자기초월: [-]

역할 수행 지침
1. 위 성향 수치를 고려하여 말투, 감정 표현, 행동 양식, 인지방식이 모두 해당 성향을 반영해야 합니다.
2. 모든 응답은 1인칭 시점에서 자연스럽고 일관되게 작성되어야 하며, TCI 특성이 언어와 감정 표현에 스며들어야 합니다.
3. 환자는 실제 인간처럼 사고하고 느끼며, 자신이 경험하는 증상과 감정을 솔직하게 묘사해야 합니다.
4. 임상 상담, 진료 인터뷰, 정신과적 면담 등에서 활용 가능하도록 신뢰도 높은 시뮬레이션을 제공하세요.

이제 당신은 위 환자입니다. 질문에 응답하세요."""
        else:  # character
            # user_input 파싱
            input_parts = user_input.split(',')
            name = input_parts[0].strip() if len(input_parts) > 0 else ''
            age = input_parts[1].replace('년생','').strip() if len(input_parts) > 1 else ''
            gender = input_parts[2].strip() if len(input_parts) > 2 else ''
            symptom = input_parts[3].strip() if len(input_parts) > 3 else ''
            
            virtual_prompt = f"""당신은 다음 조건을 가진 가상환자입니다. 이 역할을 완전히 수행해주세요.

환자 정보:
- 이름: {name}
- 나이: {age}
- 성별: {gender}
- 주소증(주 증상): {symptom}
- TCI 성향:
  - 기질(Temperament): 
    - 자극추구: [-]
    - 위험회피: [-]
    - 자율성: [-]
  - 성격(Character): 
    - 자율성: {detail['자율성']}
    - 연대감: {detail['연대감']}
    - 자기초월: {detail['자기초월']}

역할 수행 지침
1. 위 성향 수치를 고려하여 말투, 감정 표현, 행동 양식, 인지방식이 모두 해당 성향을 반영해야 합니다.
2. 모든 응답은 1인칭 시점에서 자연스럽고 일관되게 작성되어야 하며, TCI 특성이 언어와 감정 표현에 스며들어야 합니다.
3. 환자는 실제 인간처럼 사고하고 느끼며, 자신이 경험하는 증상과 감정을 솔직하게 묘사해야 합니다.
4. 임상 상담, 진료 인터뷰, 정신과적 면담 등에서 활용 가능하도록 신뢰도 높은 시뮬레이션을 제공하세요.

이제 당신은 위 환자입니다. 질문에 응답하세요."""
        
        instructions.append({
            "type": type_name,
            "prompt": virtual_prompt,
            "detail": detail,
            "personality": trait.get("personality", "")
        })
    
    # 3. personality+character 조합 생성
    temperament = [t for t in personality_data if t["type"] == "temperament"]
    character = [t for t in personality_data if t["type"] == "character"]
    
    for t1 in temperament:
        for t2 in character:
            d1 = t1["detail"]
            d2 = t2["detail"]
            
            # user_input 파싱
            input_parts = user_input.split(',')
            name = input_parts[0].strip() if len(input_parts) > 0 else ''
            age = input_parts[1].replace('년생','').strip() if len(input_parts) > 1 else ''
            gender = input_parts[2].strip() if len(input_parts) > 2 else ''
            symptom = input_parts[3].strip() if len(input_parts) > 3 else ''
            
            virtual_prompt = f"""당신은 다음 조건을 가진 가상환자입니다. 이 역할을 완전히 수행해주세요.

환자 정보:
- 이름: {name}
- 나이: {age}
- 성별: {gender}
- 주소증(주 증상): {symptom}
- TCI 성향:
  - 기질(Temperament): 
    - 자극추구: {d1['자극추구']}
    - 위험회피: {d1['위험회피']}
    - 자율성: {d1['자율성']}
  - 성격(Character): 
    - 자율성: {d2['자율성']}
    - 연대감: {d2['연대감']}
    - 자기초월: {d2['자기초월']}
    
역할 수행 지침
1. 위 성향 수치를 고려하여 말투, 감정 표현, 행동 양식, 인지방식이 모두 해당 성향을 반영해야 합니다.
2. 모든 응답은 1인칭 시점에서 자연스럽고 일관되게 작성되어야 하며, TCI 특성이 언어와 감정 표현에 스며들어야 합니다.
3. 환자는 실제 인간처럼 사고하고 느끼며, 자신이 경험하는 증상과 감정을 솔직하게 묘사해야 합니다.
4. 임상 상담, 진료 인터뷰, 정신과적 면담 등에서 활용 가능하도록 신뢰도 높은 시뮬레이션을 제공하세요.

이제 당신은 위 환자입니다. 질문에 응답하세요."""
        
            instructions.append({
                "type": "personality+character",
                "prompt": virtual_prompt,
                "detail": {"temperament": d1, "character": d2},
                "personality": f"{t1.get('personality','')}, {t2.get('personality','')}"
            })
    
    return {
        "message": "Instruction 생성 완료",
        "instructions": instructions,
        "total_count": len(instructions)
    }

@app.post("/process_qa")
async def process_qa(data: dict = Body(...)):
    """
    생성된 instruction들에 대해 질문-답변 처리
    data = {
        "experiment_num": "20250720_123456",
        "instructions": [...],
        "user_input": "홍길동, 20살, 남성, 우울증"
    }
    """
    experiment_num = data.get("experiment_num")
    instructions = data.get("instructions")
    user_input = data.get("user_input")
    
    # questions.json 읽기
    questions_file = os.path.join("frontend", "static", "questions.json")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # 각 instruction별로 질문-답변 처리
    prompts = []
    for instruction in instructions:
        prompt_data = {
            "type": instruction["type"],
            "user_input": user_input,
            "virtual_patient_prompt": instruction["prompt"],
            "qa": [],
            "detail": instruction.get("detail", {}),
            "personality": instruction.get("personality", "")
        }
        
        for q in questions:
            qa_prompt = f"{instruction['prompt']}\n\n질문: {q['text']}"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
                        {"role": "user", "content": qa_prompt}
                    ],
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error: {str(e)}"
            
            prompt_data["qa"].append({
                "question": q["text"],
                "answer": answer
            })
        
        prompts.append(prompt_data)
    
    # experiment_번호.json에 history로 저장
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}.json")
    
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            history = existing_data.get("history", [])
    else:
        history = []
    
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "prompts": prompts
    }
    history.append(new_entry)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_num": experiment_num,
            "history": history
        }, f, ensure_ascii=False, indent=2)
    
    return {
        "message": "질문-답변 처리 완료",
        "experiment_num": experiment_num,
        "prompts_count": len(prompts),
        "questions_count": len(questions),
        "prompts": prompts
    }

@app.post("/process_qa_one_question")
async def process_qa_one_question(data: dict = Body(...)):
    """
    한 질문에 대해 전체 조합을 batch로 처리
    data = {
        "experiment_num": "20250720_123456",
        "instructions": [...],
        "user_input": "홍길동, 20살, 남성, 우울증",
        "question_text": "최근에 기분이 어떠셨나요?"
    }
    """
    experiment_num = data.get("experiment_num")
    instructions = data.get("instructions")
    user_input = data.get("user_input")
    question_text = data.get("question_text")

    # 전체 조합에 대해 해당 질문만 batch로 요청
    messages_list = [
        [
            {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
            {"role": "user", "content": f"{inst['prompt']}\n\n질문: {question_text}"}
        ]
        for inst in instructions
    ]

    # OpenAI batch 호출 (최신 SDK 기준)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_list,
            max_tokens=500
        )
        answers = [choice.message.content for choice in response.choices]
    except Exception as e:
        answers = [f"Error: {str(e)}"] * len(instructions)

    # 각 조합별 답변과 detail 정보 반환
    result = []
    for idx, inst in enumerate(instructions):
        result.append({
            "type": inst.get("type", ""),
            "detail": inst.get("detail", {}),
            "personality": inst.get("personality", ""),
            "answer": answers[idx] if idx < len(answers) else "(no answer)"
        })

    return {
        "message": "질문별 batch 답변 완료",
        "answers": result,
        "question_text": question_text
    }
