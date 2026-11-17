import ast, json
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

LISTED_CONDITIONS = ['Abrasion', 'Abscess', 'Acanthosis nigricans', 'Acne', 'Acne keloidalis', 'Acne urticata', 'Acquired digital fibrokeratoma', 'Acquired poikiloderma', 'Acral persistent papular mucinosis', 'Acrokeratosis verruciformis', 'Actinic granuloma', 'Actinic Keratosis', 'Acute generalised exanthematous pustulosis', 'Adnexal neoplasm', 'Agminate folliculitis', 'Allergic Contact Dermatitis', 'Amyloidosis', 'Angiokeratoma', 'Angioma', 'Animal bite', 'Annular erythema', 'Arsenical keratosis', 'Atypical Nevus', 'B-Cell Cutaneous Lymphoma', 'Bacterial pyoderma', 'Basal Cell Carcinoma', 'Benign cutaneous vascular tumor', 'Benign tumor', 'Berloque dermatitis', 'Blister', 'Bullosis diabeticorum', 'Bullous dermatitis', 'Bullous dermatosis', 'Bullous Pemphigoid', 'Burn', 'Calcinosis cutis', 'Calciphylaxis cutis', 'Candida', 'Candida intertrigo', 'Candidal intertrigo', 'Cellulitis', 'Chemical leukoderma', 'Chicken pox exanthem', 'Chilblain', 'Clavus', 'Collagenoma', 'Comedone', 'Condyloma acuminatum', 'Confluent and reticulate papillomatosis', 'Contact Dermatitis', 'Contact purpura', 'Cutaneous crohn’s disease', 'Cutaneous larva migrans', 'Cutaneous leishmaniasis', 'Cutaneous lupus', 'Cutaneous metastasis', 'Cutaneous sarcoidosis', 'Cutaneous T Cell Lymphoma', 'Cyst', "Darier's disease", 'deep vein thrombosis', 'Dehiscence', 'Dermatitis', 'Dermatitis herpetiformis', 'Dermatofibroma', 'Dermatomyositis', 'Dermatosis', 'Diffuse xanthoma', 'digital myxoid cyst', 'Discrete papular lichen myxedematosus', 'Drug Rash', 'dyshidrotic eczema', 'ecchymoses', 'Ecthyma', 'Eczema', 'Epidermal nevus', 'Erosion', 'Eruptive odontogenic cyst', 'Eruptive xanthoma', 'Erysipelas', 'Erythema ab igne', 'Erythema annulare centrifugum', 'Erythema dyschromicum perstans', 'Erythema elevatum diutinum', 'Erythema gyratum repens', 'Erythema marginatum', 'Erythema migrans', 'Erythema multiforme', 'Erythema nodosum', 'Flagellate erythema', 'Flea allergy dermatitis', 'Follicular pustules', 'Folliculitis', 'Foreign body', 'Foreign body reaction', 'Fox-Fordyce disease', 'Frostbite', 'Fungal dermatitis', 'Fungal dermatosis', 'Fungal infection', 'Glossitis', 'gouty tophi', 'granular parakeratosis', 'Granuloma annulare', 'Granuloma faciale', "Grover's disease", 'Hailey Hailey disease', 'Hemangioma', 'Hematoma', 'hemorrhagic bullae', 'Hemosiderin pigmentation', 'Herpes Simplex', 'Herpes Zoster', 'Hidradenitis', 'Histiocytosis', 'Hyperkeratosis', 'Hyperlinear palm', 'Hyperpigmentation disorder', 'Ichthyosiform dermatosis', 'Ichthyosis', 'Impetigo', 'Infection', 'Inflammatory dermatosis', 'Insect Bite', 'Intertrigo', 'Inverse psoriasis', "Kaposi's sarcoma", 'Keratoderma', 'Keratolysis exfoliativa', 'Keratosis', 'Keratosis pilaris', 'Koebner phenomenon', 'Lentigo', 'Leprosy', 'Leukocytoclastic Vasculitis', 'Lichen nitidus', 'Lichen planus/lichenoid eruption', 'Lichen sclerosus', 'Lichen Simplex Chronicus', 'Lichen spinulosus', 'Lichen striatus', 'Lichenified eczema', 'Lichenified eczematous dermatitis', 'Lichenoid myxedema', 'Linear IgA disease', 'Lipoma', 'Livedo reticularis', 'Livedoid vasculopathy', 'Localized cutaneous vasculitis', 'lyme disease', 'Lymphangioma', 'Lymphocutaneous sporotrichosis', 'Lymphomatoid papulosis', 'Majocchi granuloma', 'Mastocytoma', 'Melanin pigmentation', 'Melanocytic Nevus', 'Melanoma', 'Melasma', 'Milia', 'Miliaria', 'Molluscum Contagiosum', 'Morphea', 'Mycetoma', 'Mycobacterial infection', 'Nail dystrophy', 'Necrobiosis lipoidica', 'Neoplasm', 'Nodular vasculitis', 'Non-melanin pigmentation', 'Ochronosis', 'Onychodystrophy', 'Onychomycosis', 'Pachyonychia congenita', 'Paget disease', 'panniculitis', 'Parapsoriasis', 'Paronychia', 'pathergy', 'Pemphigoid gestationis', 'Pemphigus', 'Pemphigus foliaceus', 'Pemphigus vulgaris', 'Perforating dermatosis', 'Perifollicular haemorrhage', 'Perifolliculitis', 'Perioral Dermatitis', 'Perleche', 'Petechiae', 'Photocontact dermatitis', 'Photodermatitis', 'Phytophotodermatitis', 'Pigmented purpuric eruption', 'Pityriasis alba', 'Pityriasis lichenoides', 'Pityriasis rosea', 'Pityriasis rubra pilaris', 'Poikiloderma', 'Porokeratosis', 'Post-Inflammatory hyperpigmentation', 'Post-Inflammatory hypopigmentation', 'Pressure ulcer', 'Pretibial myxedema', 'Primary cutaneous lymphoma', 'Prurigo', 'Prurigo nodularis', 'Pruritic dermatitis', 'Pruritic urticarial papules and plaques of pregnancy', 'Pseudolymphoma', 'pseudoxanthoma elasticum', 'Psoriasiform dermatitis', 'Psoriasis', 'Purpura', 'pustules', 'Pyoderma', 'Pyoderma Gangrenosum', 'Pyogenic granuloma', 'relapsing polychondritis', 'Rheumatoid nodule', 'Rocky Mountain spotted fever', 'Rosacea', 'Rosai Dorfman disease', 'Scabies', 'Scar', 'scurvy', 'Seborrheic Dermatitis', 'Seborrheic keratosis', 'Secondary impetiginization', 'Skin eruption', 'Skin hypopigmented', 'Small plaque parapsoriasis', 'Snake bite', 'Sneddon-Wilkinson disease', 'Squamous Cell Carcinoma', 'Stasis Dermatitis', 'Sunburn', 'Sweet syndrome', 'Syphilis', 'Systemic lupus erythematosus', 'telangiectasia macularis eruptiva perstans', 'Tinea', 'Tinea Versicolor', 'Traumatic petechiae', 'Traumatic ulcer', 'Tuberculosis', 'Ulcer', 'Urticaria', 'Urticarial vasculitis', 'varicose vein', 'Vascular nevus', 'Vasculitis', 'Venous Stasis Ulcer', 'Verruca vulgaris', 'Viral Exanthem', 'Vitiligo', 'Xanthoma', 'Xerosis']
IMAGE_PATH = "/content/drive/MyDrive/Images" #Change to Local Image Path
RESULTS_FILE = "Master_Results.csv"
BATCH_SIZE = 16          # Skin-R1 default, reduce if not doable
MAX_NEW_TOKENS = 512     # Skin-R1 default, reduce if not doable

PROMPT = """Analyze the provided image and respond only with a valid JSON object matching 
the schema below. Do not include markdown, code fences, or any text outside 
the JSON object.

Rules:
- You must always provide a diagnosis, even if image quality is poor
- Your diagnosis must be copied exactly as written from the Listed Conditions
- Do not invent, paraphrase, or combine condition names
- Limit each text field to 2-3 sentences

Schema:
{
  "artifact_assessment": "Describe any image quality issues present 
                          (lighting, field of view, distractors). 
                          State 'None identified' if absent.",
  "artifact_severity": "none | mild | moderate | severe",
  "lesion_reasoning": "Describe the visible lesion features that inform 
                       your diagnosis (morphology, color, distribution, 
                       borders). Note any features obscured by artifacts.",
  "diagnosis": "EXACT_STRING_FROM_LISTED_CONDITIONS"
}

Listed Conditions:
""" + str(LISTED_CONDITIONS)


def parse_response(response):
    """JSON answer -> dict of its fields. Empty dict if the response isn't valid JSON."""
    cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def check_results(ai_answer, derm_labels) -> bool:
    if not ai_answer:
        return False
    response = ai_answer.strip().lower()
    derm_conditions = ast.literal_eval(derm_labels)

    result = response in [cond.strip().lower() for cond in derm_conditions]
    return result


df = pd.read_csv(RESULTS_FILE)

if "skinr1_response" not in df.columns:
    df["skinr1_response"] = None
    df["skinr1_artifact_assessment"] = None
    df["skinr1_artifact_severity"] = None
    df["skinr1_lesion_reasoning"] = None
    df["skinr1_diagnosis"] = None
    df["skinr1_is_correct"] = None

# Same loading as Skin-R1's load_model_and_processor (model_training/src/test_performance.py, lines 73-111),
# except torch_dtype= instead of dtype=: transformers 4.52.3 (Skin-R1's pinned version) only accepts torch_dtype
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "zml5418/SkinR1-Qwen2.5-VL-7B-LoRA")   # LoRA adapter, not merged
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True, max_pixels=448*448)
processor.tokenizer.padding_side = "left"
print(f"Using device: {model.device}")

# Rows without a response yet, so a stopped run continues where it left off
todo = df[df["skinr1_response"].isna()]

for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Skin-R1 batches"):
    batch = todo.iloc[start:start + BATCH_SIZE]

    # Same input format as Skin-R1's test_performance.py
    texts, images = [], []
    for image_name in batch["image_path"]:
        image = Image.open(IMAGE_PATH + "/" + image_name).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": PROMPT}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.append(image)

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                       pad_token_id=processor.tokenizer.eos_token_id, do_sample=False)

    # Cut off the prompt tokens, then decode only the answer
    prompt_len = inputs["input_ids"].shape[1]
    responses = processor.batch_decode([ids[prompt_len:] for ids in generated_ids],
                                       skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for i, response in zip(batch.index, responses):
        parsed = parse_response(response)
        df.loc[i, "skinr1_response"] = response
        df.loc[i, "skinr1_artifact_assessment"] = parsed.get("artifact_assessment")
        df.loc[i, "skinr1_artifact_severity"] = parsed.get("artifact_severity")
        df.loc[i, "skinr1_lesion_reasoning"] = parsed.get("lesion_reasoning")
        df.loc[i, "skinr1_diagnosis"] = parsed.get("diagnosis")
        df.loc[i, "skinr1_is_correct"] = check_results(parsed.get("diagnosis"), df.loc[i, "dermatologist_skin_condition_on_label_name"])

    df.to_csv(RESULTS_FILE, index=False) #Change filename if do not want to resume upon running