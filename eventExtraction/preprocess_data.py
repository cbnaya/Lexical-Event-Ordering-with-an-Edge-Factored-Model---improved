"""
Add the word 'you' in the beginning of every sentence.
"""

TARGET_VERB_FILE = 'resources/verb_list.txt'
g_verb_list = []


def _init_verb_list():
    global g_verb_list
    g_verb_list = [line.strip().lower() for line in open(TARGET_VERB_FILE) if line.strip() != ""]


def add_you_preprocessing(text):
    if not g_verb_list:
        _init_verb_list()
    result = ""
    sentences = text.strip().split('.')
    for sent in sentences:
        words = sent.strip().split()
        if len(words) > 0 and words[0].lower() in g_verb_list:
            words.insert(0, 'you')
            words[1] = words[1].lower()
        result += ' '.join(words) + '. '
    return result

def run_preprocessing(text):
    return add_you_preprocessing(text)
