"""
Execute the event extraction pipeline for a particular file.
"""
import os
import preprocess_data
import tempfile
import re
import pycorenlp
from recipesResources import ffts_recipes


def build_dep_string(dependency_result):
    deps = [u"{0}({1}-{2}, {3}-{4})".format(dep["dep"],
                                            dep["governorGloss"],
                                            dep["governor"],
                                            dep['dependentGloss'],
                                            dep['dependent']) for dep in dependency_result]
    return os.linesep.join(deps)


def parse_recipe(recipe_text):
    result = []
    nlp = pycorenlp.StanfordCoreNLP('http://localhost:9000')
    annotate_result = nlp.annotate(recipe_text, properties={'annotators': 'parse, depparse',
                                                            'outputFormat': 'json',
                                                            'timeout': '50000'})
    if type(annotate_result) in (str, unicode):
        raise Exception(annotate_result)

    for trees in annotate_result['sentences']:
        parse_tree = re.sub("\s+", " ", trees['parse']).encode("utf8")
        dependecy_tree = build_dep_string(trees['basicDependencies']).encode("utf8")

        result.append((parse_tree, dependecy_tree))
    return result


def add_discourse(parse_tree_str):
    if not os.path.isdir("addDiscourse"):
        raise Exception("addDiscourse not found")
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(parse_tree_str)
    f.close()
    with_discourse = os.popen("perl addDiscourse/addDiscourse.pl --parses %s" % f.name).read().strip()
    os.remove(f.name)
    return with_discourse


def run_pipeline(recipes):
    preprocessed_recipes = [preprocess_data.run_preprocessing(recipe) for recipe in recipes]

    for recipe in preprocessed_recipes:
        sentences = parse_recipe(recipe)
        sentences = [(add_discourse(parse_tree), dependecies_tree) for parse_tree, dependecies_tree in sentences]

        for parse_tree, dependecies_tree in sentences:
            ptree = extract_events.read_tree(parse_tree)
            standep = extract_events.read_sds_from_string(dependecies_tree)
            

        # # reads the stanford deps, the ptree and write it to f_out
        # f_out = open(out_filename, 'w')
        # f_out_parses = open(out_filename + '.parses', 'w')
        # for recipe in list_of_tree_sd_string_pairs:
        #     for tree_sd_string_pair in recipe:
        #         f_out_parses.write(tree_sd_string_pair[0] + '\n' + tree_sd_string_pair[1] + '\n')
        #         ptree = extract_events.read_tree(tree_sd_string_pair[0])
        #         standep = extract_events.read_sds_from_string(tree_sd_string_pair[1])
        #         extract_events.write_events_linkages(standep, ptree, f_out)
        #     f_out.write('=========\n')
        #     f_out_parses.write('=========\n')
        # f_out.close()
        # f_out_parses.close()


if __name__ == "__main__":
    recipes = ffts_recipes.get_recipe_by_name("2000.mmf")["2000.mmf"]
    recipes_text = [recipe['direction_lines'] for recipe in recipes]
    run_pipeline(recipes_text)
    # if len(sys.argv) != 2:
    #     print('Usage: run_pipeline.py <input file>')
    #     sys.exit(-1)
    # run_pipeline(sys.argv[1], sys.argv[1] + '.events')
