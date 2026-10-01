"""BABILong answer matching, from babilong/metrics.py (booydar/babilong, Apache-2.0).

Only the first sentence of the answer counts. It is correct when the target location is
the one and only qa3 label it mentions, after dropping labels the question itself names
(those are never the answer). Hedging across two rooms is therefore wrong.
"""

import re

LABELS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]  # TASK_LABELS["qa3"]


def preprocess_output(output):
    output = output.lower()
    # take only the first sentence from output
    output = output.split('.')[0]
    # filter responses when model tries to generate examples
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]
    return output


def compare_answers(target, output, question, task_labels=LABELS):
    output = preprocess_output(output)
    target = target.lower()
    task_labels = {label.lower() for label in task_labels}

    # if a label contains another label (qa7: 'none' contains 'one'), match labels as whole words,
    # otherwise a correct 'none' also mentions 'one' and the answer is rejected
    whole_words = any(a != b and a in b for a in task_labels for b in task_labels)

    def mentioned(label, text):
        if whole_words:
            return re.search(r'\b' + re.escape(label) + r'\b', text) is not None
        return label in text

    # extract labels that were mentioned in the question
    labels_in_question = {label for label in task_labels if mentioned(label, question.lower())}
    # extract labels that were mentioned in the model output
    labels_in_output = {label for label in task_labels if mentioned(label, output)}
    # filter labels in the output to exclude mentioned in the question
    # mentions in questions are never targets
    labels_in_output = labels_in_output - labels_in_question

    # check if the target is the only prediction
    if ',' in target and len(target) > 3:
        # if target contains multiple subtargets in qa8
        subtargets = target.split(',')
        num_subtargets = len(subtargets)
        if all([t in labels_in_output for t in subtargets]) and len(labels_in_output) == num_subtargets:
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True

    return False
