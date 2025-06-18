import requests
import json
import os

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:instruct"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output", type=str, required=True)  # parsed.json 경로
args = parser.parse_args()

user_input = args.prompt
JSON_PATH = args.output
CONTROLNET_PROMPT_PATH = os.path.join(os.path.dirname(JSON_PATH), "controlnet_prompts.json")


def call_ollama_extract_parts(prompt):
    system_prompt = f"""
        You are an assistant that extracts structured information in JSON format.
        Given a sentence describing an object and its parts, return a JSON object with:
            "object": the main object name (e.g., "chair", "sofa")
            "parts": a list of part entries. Each part must have:
                "part_name": name of the part (e.g., "seat", "back part", "armrest") — use the exact phrase as given, don't split or modify.
                "style": style or material for that part. If not mentioned, return an empty string "".
                    If the object itself has a style (e.g., "modern sofa"), treat it as a part entry with "part_name" equal to the object name (e.g., "sofa"), and "style" filled appropriately.

        Format:
        {{
        "object": "OBJECT_NAME",
        "parts": [
            {{ "part_name": "PART1", "style": "STYLE1" }},
            ...
        ]
        }}

        Examples:

        Input:
        "A leather sofa with soft blue cushions, polished wooden legs, and gold decorative studs."

        Output:
        {{
        "object": "sofa",
        "parts": [
            {{ "part_name": "cushions", "style": "soft blue" }},
            {{ "part_name": "legs", "style": "polished wooden" }},
            {{ "part_name": "studs", "style": "gold decorative" }},
            {{ "part_name": "sofa", "style": "leather" }}
        ]
        }}


        Input:
        "A sofa with soft blue cushions, polished wooden legs, and gold decorative studs."

        Output:
        {{
        "object": "sofa",
        "parts": [
            {{ "part_name": "cushions", "style": "soft blue" }},
            {{ "part_name": "legs", "style": "polished wooden" }},
            {{ "part_name": "studs", "style": "gold decorative" }},
            {{ "part_name": "sofa", "style": "" }}
        ]
        }}

        Now extract the JSON for:

        "{prompt}"

        Respond only with valid JSON and Requested format. Do not add any annotation or explaination.
    """

    response = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False
    })

    try:
        content = response.json()["response"]
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
    except Exception:
        print("Failed to parse JSON:")
        print(response.text)
        raise

    with open(JSON_PATH, "w") as f:
        json.dump(parsed, f, indent=2)

    return parsed


def prepare_grounded_prompt_and_style_map(parsed_json):
    parts = parsed_json.get("parts", [])
    object_name = parsed_json.get("object")

    part_names = [p["part_name"] for p in parts]
    if object_name and object_name not in part_names:
        part_names.append(object_name)

    grounded_prompt = ", ".join(part_names)
    style_map = {p["part_name"]: p["style"] for p in parts}

    if isinstance(parsed_json.get("object"), dict):
        obj = parsed_json["object"]
        if "style" in obj and "name" in obj:
            style_map[obj["name"]] = obj["style"]
    elif isinstance(object_name, str) and object_name not in style_map:
        style_map[object_name] = " "

    return grounded_prompt, style_map


def generate_controlnet_prompts(parsed_json, output_path):
    prompts = []
    object_name = parsed_json.get("object", "object")
    parts = parsed_json.get("parts", [])

    for part in parts:
        part_name = str(part.get("part_name") or "").strip()
        style = str(part.get("style") or "").strip()


        if not style.strip() or style.strip().lower() in ["none", "na", "n/a"]:
            continue


        reference_examples = """
        Style Enhancement Examples:
        - white fabric → pristine snow-white fabric with soft cotton texture and subtle weave patterns
        - black leather → deep obsidian leather with glossy finish and natural grain detail
        - gray plastic → industrial matte gray plastic with fine textured surface
        - red velvet → rich burgundy velvet with plush pile and luxurious sheen
        - wooden surface → warm honey-toned wood with visible grain and smooth finish
        """

        user_prompt = (
            f"Generate a detailed texture description for image editing.\n"
            f"Target: Apply '{style}' specifically to the **{part_name}** of a {object_name}.\n"
            f"Requirements:\n"
            f"- Focus ONLY on the {part_name}, ignore other parts\n"
            f"- Enhance the '{style}' with realistic material properties\n"
            f"- Include surface texture, finish type, and visual characteristics\n"
            f"- Use 3-5 descriptive adjectives that specify material qualities\n\n"
            f"Reference examples for style enhancement:\n{reference_examples}\n"
            f"Output format: 'a {object_name} with {part_name} in {style}, [enhanced material description], photorealistic, high detail'\n"
            f"Provide only the final description without quotes or additional text."
        )




        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": user_prompt,
                "stream": False
            })
            raw = response.json().get("response", "").strip()
            content = raw.strip('"').replace('\\"', '"')

            if not content:
                raise ValueError("Empty response")

            print(f"[✓] {part_name}: {content}")
            prompts.append({
                "part_name": part_name,
                "prompt": content
            })

        except Exception as e:
            print(f"[Error] Failed to generate prompt for '{part_name}': {e}")

    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\n[Done] Saved prompts to: {output_path}")


if __name__ == "__main__":
   
    parsed = call_ollama_extract_parts(user_input)
    grounded_prompt, style_map = prepare_grounded_prompt_and_style_map(parsed)

    print("\nGrounded-SAM Prompt:")
    print(grounded_prompt)

    print("\nStyle mapping:")
    for part, style in style_map.items():
        print(f"{part}_mask.png → \"{style}\"")

    print(f"\nSaved JSON: {JSON_PATH}")

    generate_controlnet_prompts(parsed, CONTROLNET_PROMPT_PATH)
