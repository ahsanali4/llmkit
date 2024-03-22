from typing import Dict, Union

from langchain.evaluation import Criteria, load_evaluator
from langchain.evaluation.schema import EvaluatorType  # , LLMEvalChain, StringEvaluator

# more relevant to us
_SUPPORTED_CRITERIA = {
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?",
    Criteria.DEPTH: "Does the submission demonstrate depth of thought?",
    Criteria.CREATIVITY: "Does the submission demonstrate novelty or unique ideas?",
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
}

custom_criteria = {
    "simplicity": "Is the language straightforward and unpretentious?",
    "clarity": "Are the sentences clear and easy to understand?",
    "precision": "Is the writing precise, with no unnecessary words or details?",
    "truthfulness": "Does the writing feel honest and sincere?",
    "subtext": "Does the writing suggest deeper meanings or themes?",
}


def criteria_evaluator(
    query: str, answer: str, reference: str, criteria: Union[str, Dict] = "correctness"
):

    evaluator = load_evaluator(EvaluatorType.LABELED_CRITERIA, criteria=criteria)

    # We can even override the model's learned knowledge using ground truth labels
    return evaluator.evaluate_strings(
        input=query,
        prediction=answer,
        reference=reference,
    )


def string_pair_evaluater(evaluator, query: str, model1_response: str, model2_response: str):
    return evaluator.evaluate_string_pairs(
        prediction=model1_response,
        prediction_b=model2_response,
        input=query,
    )


def model_comparison(
    query: str,
    model1_response: str,
    model2_response: str,
    criteria: Union[str, Dict] = "correctness",
):
    evaluator = load_evaluator(EvaluatorType.PAIRWISE_STRING, criteria=criteria)
    string_pair_evaluater(
        evaluator=evaluator,
        query=query,
        model1_response=model1_response,
        model2_response=model2_response,
    )


def evaluate_embedding_distance(answer: str, reference: str):

    evaluator = load_evaluator(EvaluatorType.EMBEDDING_DISTANCE)
    return evaluator.evaluate_strings(prediction=answer, reference=reference)
