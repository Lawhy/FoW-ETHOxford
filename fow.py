from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
import json

EXAMPLES = [
    "Project Description: Mystic Realms is an immersive, open-world RPG set in a fantastical universe where magic and technology coexist. The game invites players to explore the vast, enchanting world of Eldoria, a land filled with ancient mysteries, dynamic ecosystems, and a tapestry of cultures. Players assume the role of a customizable protagonist, who is a member of the Arcane Guild, tasked with investigating a series of mysterious events that threaten the balance of the world.",
    "Project Description: Decentralized Art Gallery (DAG) is a cutting-edge Web3 application designed to revolutionize the art world by leveraging blockchain technology. DAG provides a platform for artists to showcase their digital artworks, ensuring authenticity, provenance, and ownership through non-fungible tokens (NFTs). Art enthusiasts can discover, purchase, and collect unique pieces of art directly from creators worldwide, fostering a direct artist-to-collector relationship.",
]


class ChatAgent:

    def __init__(
        self,
        pretrained: str = "mistralai/Mistral-7B-v0.1",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        freelancer_database: str = "",
    ):
        self.pretrained_path = pretrained
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        if not load_in_4bit and not load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_path).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_path, quantization_config=self.quantization_config
            )
        with open(freelancer_database) as f:
            self.freelancers = json.load(f)

    def response(self, query: str, max_new_tokens: int = 32):
        input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0])

    def parse(self, description: str, max_new_tokens: int = 64):
        description = f"{description}\n\nWhat is the skill set required for completing this project?"
        return self.response(description, max_new_tokens)[len(description) + 5 :]

    def interface(self):

        def match(msg: str):
            skills = self.parse(msg, max_new_tokens=64)
            parsed_skills = [x.strip() for x in skills.split("-")[1:]]
            # print(parsed_skills)
            results = find_combinations(self.freelancers, parsed_skills)
            return skills, format_combinations(self.freelancers, results)

        desc = gr.Textbox(label="Project Description")
        skillset = gr.Textbox(label="Skill Sets")
        freelancer = gr.Textbox(label="Matched Freelancers")
        demo = gr.Interface(
            fn=match,
            inputs=desc,
            outputs=[skillset, freelancer],
            examples=EXAMPLES,
            title="Future of Work",
        )
        demo.launch(share=True)


def find_combinations(profiles, required_skills):
    
    def is_useful(skill_set, required_skills):
        return any(skill in required_skills for skill in skill_set)

    def covers_all_skills(combined_skills, required_skills):
        return all(skill in combined_skills for skill in required_skills)

    def backtrack(current_index, current_set, combined_skills):
        if covers_all_skills(combined_skills, required_skills):
            valid_combinations.append(current_set.copy())
            return

        for i in range(current_index, len(profile_ids)):
            profile_id = profile_ids[i]
            skills = profiles[profile_id]["skills"]

            if not is_useful(skills, required_skills):
                continue

            updated_skills = combined_skills.union(skills.keys())
            current_set.append(profile_id)

            backtrack(i + 1, current_set, updated_skills)

            current_set.pop()

    profile_ids = list(profiles.keys())
    valid_combinations = []
    backtrack(0, [], set())

    return valid_combinations


def format_combinations(profiles, combinations):
    formatted_combinations = []
    for i, combo in enumerate(combinations):
        combo_text = f"Combination ({i+1}):\n"
        for profile_id in combo:
            profile = profiles[profile_id]
            combo_text += f"- ID: {profile_id}, Name: {profile['name']}, Profession: {profile['profession']}\n"
            combo_text += "  Skills:\n"
            for skill, level in profile["skills"].items():
                combo_text += f"    - {skill}: Level {level}\n"
            combo_text += f"  Review: ${profile['review']}\n\n"
        formatted_combinations.append(combo_text.strip())
    return "\n\n".join(formatted_combinations)
