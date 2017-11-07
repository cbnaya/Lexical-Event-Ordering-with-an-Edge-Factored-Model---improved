"""
Execute the event extraction pipeline for a particular file.
"""
import os
import preprocess_data
import tempfile
import re
import pycorenlp
import extract_events
from recipesResources import ffts_recipes


def build_dep_string(dependency_result):
    deps = [u"{0}({1}-{2}, {3}-{4})".format(dep["dep"],
                                            dep["governorGloss"],
                                            dep["governor"],
                                            dep['dependentGloss'],
                                            dep['dependent']) for dep in dependency_result]
    return os.linesep.decode("ascii").join(deps)


def parse_recipe(recipe_text):
    result = []
    nlp = pycorenlp.StanfordCoreNLP('http://localhost:9000')
    annotate_result = nlp.annotate(recipe_text, properties={'annotators': 'parse, depparse',
                                                            'outputFormat': 'json',
                                                            'timeout': '50000'})
    if type(annotate_result) in (str, unicode):
        raise Exception(annotate_result)

    for trees in annotate_result['sentences']:
        parse_tree = re.sub(u"\s+", u" ", trees['parse']).encode("utf8")
        dependecy_tree = build_dep_string(trees['basicDependencies']).encode("utf8")
        result.append((parse_tree, dependecy_tree))
    return result


def add_discourse(parse_trees):
    text = os.linesep.join(parse_trees)
    if not os.path.isdir("addDiscourse"):
        raise Exception("addDiscourse not found")
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(text)
    f.close()
    with_discourse = os.popen("perl addDiscourse/addDiscourse.pl --parses %s" % f.name).read().strip()
    # TODO: handle the encoding
    os.remove(f.name)
    return with_discourse.splitlines()


def run_pipeline(recipes):
    preprocessed_recipes = [preprocess_data.run_preprocessing(recipe) for recipe in recipes]
    f = open("res", "w")
    f_out = open("events", "w")
    for recipe in preprocessed_recipes:
        try:
            sentences = parse_recipe(recipe)
            parse_tree_with_discourse = add_discourse([parse_tree for parse_tree, dependecies_tree in sentences])
            # replace the parse tree with parse tree with discourse
            sentences = [(parse_tree_with_discourse[i], sentences[i][1]) for i in range(len(sentences))]

            for parse_tree, dependecies_tree in sentences:
                f.write(parse_tree + "\n" + dependecies_tree + "\n")

                ptree = extract_events.read_tree(parse_tree)
                standep = extract_events.read_sds_from_string(dependecies_tree)
                extract_events.write_events_linkages(standep, ptree, f_out)

                # ptree = extract_events.read_tree(parse_tree)
                # standep = extract_events.read_sds_from_string(dependecies_tree)
        except Exception as e:
            print "===========\n{0}\nerror message:{1}\n".format(recipe, e)
            continue

        f.write('=========\n')

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
