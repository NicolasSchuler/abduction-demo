import json
from pathlib import Path

import dspy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from problog import get_evaluatable
from problog.program import PrologString

from src.data import load_data
from src.img_utils import encode_base64_resized

TESTING = 1
EPSILON_PROB = 0.0001


def reasoning():
    template_reasoning = ChatPromptTemplate.from_messages([
        ("system", "{role_reasoning}"),
        ("human", "Question: {question_reasoning}"),
    ])
    model_reasoning = ChatOpenAI(model="qwen/qwen3-30b-a3b-2507", base_url="http://127.0.0.1:1234/v1", api_key="")
    reasoning_chain = template_reasoning | model_reasoning | StrOutputParser()

    description = reasoning_chain.invoke({
        "role_reasoning": "You are a scientific expert in the classification of whether an animal is a cat or a dog. If tasked to answer questions, you shall ADHERE TO scientific facts, THINK STEP-BY-STEP, and explain your decision-making process. Focus on 'WHY' something is done, especially for complex logic, rather than 'WHAT' is done. Your answer SHOULD BE concise and direct, but still exhaustive, and avoid conversational fillers. Format your answer appropriately for better understanding. DO NOT rely on fillers like 'e.g.,', instead spell it out and try to be as complete as possible. Your reasoning MAY NOT be mutually exclusive. This is OK.",
        "question_reasoning": "I want you to do a comparative analysis of cats and dogs. Your analysis must use the inherent traits and biological characteristics of each species. You should list each of these characteristics so that an informed decision can be made about whether a given animal depicted in an image is a cat or a dog. Please provide a detailed analysis, focusing on traits and characteristics that can be extracted from a given image. For formatting please use a list.",
    })
    return description


def coding(description: str):
    template_coding = ChatPromptTemplate.from_messages([
        ("system", "{role_coding}"),
        ("human", "Instructions: {instruction}\n Description: {description}"),
    ])

    def gemini():
        # dummy implementation for calling gemini api
        pass

    model_coding = gemini()
    coding_chain = template_coding | model_coding | StrOutputParser()

    coding_description = coding_chain.invoke({
        "role_coding": """You are an expert Problog programmer with extended knowledge in reasoning and probabilities. Given instructions and a description, you write a correct logical Problog program that expresses the given question with probabilistic theory in mind. You SHALL format your answer so that it can be directly used as an input for a Problog interpreter. DO NOT incorporate example facts or queries into the knowledge base; these will be added later by the user. If necessary, add comments to your program to provide explanations to the user. You should follow roughly the following form:
            1. Core Causal Model (LOGIC)
            2. Knowledge Base (P(Feature | Animal))
            3. Observation Model (PER-OBSERVATION)
        """,
        "instruction": "Write a logical program for the following description:",
        "description": description,
    })
    return coding_description


def extract_feature_list(problog_program: str) -> list[str]:
    lm = dspy.LM("huggingface/qwen/qwen3-coder-30b", api_base="http://127.0.0.1:1234/v1", api_key="")
    dspy.configure(lm=lm)

    class FeatureNames(dspy.Signature):
        """Extracting Feature names from a Problog program."""

        problog_program: str = dspy.InputField(desc="Problog program")
        features: list[str] = dspy.OutputField(desc="List of Feature names (atoms) of the Problog program")

    prompt = dspy.ChainOfThought(FeatureNames)
    return prompt(problog_program=problog_program).features


def grounding(feature_list: list[str], image_path: Path):
    lm = dspy.LM("huggingface/qwen/qwen3-vl-8b", api_base="http://127.0.0.1:1234/v1", api_key="")
    dspy.configure(lm=lm)

    class Grounding(dspy.Signature):
        feature_list: list[str] = dspy.InputField(desc="List of possible features present in the image")
        image: dspy.Image = dspy.InputField(desc="Image to be analyzed for present features")
        grounding: dict[str, float] = dspy.OutputField(
            desc="A json mapping feature names to the probability of the feature being present in the input image. Example: {'feature_A': 0.9, 'feature_B': 0.1}"
        )

    prompt = dspy.ChainOfThought(Grounding)
    return prompt(
        feature_list=feature_list,
        image=dspy.Image(
            url=f"data:image/jpg;base64,{encode_base64_resized(image_path, max_width=512, max_height=512, quality=80)}"
        ),
    ).grounding


def execute_logic_program(problog_program: str, grounding: dict[str, float]) -> tuple[float, float]:
    """
    Execute a ProbLog program with evidence from grounding results.

    This function takes the base ProbLog program and adds evidence based on the
    grounding results (feature probabilities extracted from images). It then runs
    ProbLog inference to determine the probability of the animal being a cat or dog.

    Args:
        problog_program: Base ProbLog program as string containing the knowledge base
        grounding: Dictionary mapping feature names to probabilities (0.0 to 1.0)

    Returns:
        Tuple of (cat_probability, dog_probability)

    Example:
        >>> grounding = {"small_muzzle": 0.9, "retractable_claws": 0.8}
        >>> p_cat, p_dog = execute_logic_program(program, grounding)
        >>> print(f"Cat: {p_cat:.3f}, Dog: {p_dog:.3f}")
    """
    # Remove the example evidence section from the base program
    lines = problog_program.split("\n")
    filtered_lines = []
    skip_section = False

    for line in lines:
        # Skip the example evidence section that starts with "% --- Example Scenario:"
        if "% --- Example Scenario:" in line:
            skip_section = True
            continue
        elif (
            line.startswith("% =============================================================================")
            and skip_section
        ):
            # Stop skipping when we reach the query section
            if "5. QUERY" in line:
                skip_section = False
            continue
        elif skip_section and (
            line.startswith("confidence(")
            or line.startswith("evidence(")
            or line.startswith("% confidence(")
            or line.startswith("% evidence(")
        ):
            # Skip example confidence and evidence statements (both active and commented)
            continue
        elif skip_section and line.strip().startswith("%"):
            # Skip comments in the example section
            continue
        elif skip_section and not line.strip():
            # Skip empty lines in the example section
            continue
        # Also filter out commented queries since we add them back
        elif line.strip().startswith("% query("):
            continue
        else:
            skip_section = False
            filtered_lines.append(line)

    # Reconstruct the program without example evidence
    base_program = "\n".join(filtered_lines)

    # Add evidence based on grounding results
    evidence_lines = []
    evidence_lines.append("\n% =============================================================================")
    evidence_lines.append("% ==  4. EVIDENCE FROM GROUNDING")
    evidence_lines.append("% =============================================================================")

    for feature_name, probability in grounding.items():
        # Clamp probability to avoid numerical issues with extreme values
        prob = max(EPSILON_PROB, min(1.0 - EPSILON_PROB, probability))

        # Set up observation model parameters
        # TPR (True Positive Rate): probability of reporting the feature when it's present
        # FPR (False Positive Rate): probability of reporting the feature when it's absent
        tpr = prob  # High confidence means high TPR
        fpr = 1.0 - prob  # High confidence means low FPR

        evidence_lines.append(f"confidence(grounding_observer, {feature_name}, {tpr:.6f}, {fpr:.6f}).")
        evidence_lines.append(f"evidence(report(grounding_observer, {feature_name}), true).")

    # Add queries section
    evidence_lines.append("\n% =============================================================================")
    evidence_lines.append("% ==  5. QUERY")
    evidence_lines.append("% =============================================================================")
    evidence_lines.append("% Asks the program for the final probability of each hypothesis.")
    evidence_lines.append("query(is_cat).")
    evidence_lines.append("query(is_dog).")

    # Combine base program with evidence
    full_program = base_program + "\n".join(evidence_lines)

    # Execute ProbLog inference
    try:
        # Create ProbLog program from string
        problog_model = PrologString(full_program)

        # Get evaluatable model for inference
        evaluatable = get_evaluatable().create_from(problog_model)

        # Evaluate all queries in the program
        results = evaluatable.evaluate()

        # Extract probabilities for cat and dog classifications
        cat_prob = 0.0
        dog_prob = 0.0

        for query, probability in results.items():
            query_str = str(query)
            if "is_cat" in query_str:
                cat_prob = float(probability)
            elif "is_dog" in query_str:
                dog_prob = float(probability)

        return cat_prob, dog_prob

    except Exception as e:
        print(f"Error executing ProbLog program: {e}")
        print("Returning default equal probabilities (0.5, 0.5)")
        # Return default equal probabilities in case of error
        return 0.5, 0.5


def main():
    print("Starting Abduction Demo")
    labeled_images = load_data()
    labeled_image = labeled_images[1]

    if TESTING == 1:
        reasoning_description = Path("result-prompts/reasoning.md").open("r").read()
        problog_program = Path("result-prompts/coding.pl").open("r").read()
        feature_list = Path("result-prompts/feature-list.txt").open("r").readlines()
        grounding_res = json.loads(Path("result-prompts/grounding.json").open("r").read())
    else:
        reasoning_description = reasoning()
        with open("result-prompts/reasoning.md", "w") as f:
            f.write(reasoning_description)

        # ONLY FOR DEMONSTRATION PURPOSES
        problog_program = coding(reasoning_description)
        with open("result-prompts/coding.pl", "w") as f:
            f.write(problog_program)

        feature_list = extract_feature_list(problog_program=problog_program)
        with open(file="result-prompts/feature-list.txt", mode="w") as f:
            f.write("\n".join(feature_list))

        grounding_res = grounding(feature_list=feature_list, image_path=labeled_image.image_path)
        with open(file="result-prompts/grounding.json", mode="w") as f:
            json.dump(grounding_res, f, indent=2)

    p_cat, p_dog = execute_logic_program(problog_program, grounding_res)
    print(f"Cat Probability: {p_cat}")
    print(f"Dog Probability: {p_dog}")
    print("End Abduction Demo")


if __name__ == "__main__":
    main()
