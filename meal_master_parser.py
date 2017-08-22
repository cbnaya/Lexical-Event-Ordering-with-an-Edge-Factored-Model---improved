import re

# according http://www.ffts.com/mmformat.txt

RECIPE_HEADER_FORMAT = "-----.*Meal-Master.*?\n"
TITLE_FORMAT = "\s*Title: (.*)"
CATEGORIES_FORMAT = "\s*Categories: (.*)"
SERVINGS_FORMAT = ".*?(\d+).*"
INGREDIENT_FORMAT = "[ \d\./]{7} [a-zA-Z ]{2}"
END_OF_RECIPE = "-----[ -]*"


def split_to_recipes(text):
    return re.split(RECIPE_HEADER_FORMAT, text)


def parse_single_recipe(recipe_text):
    result = {}
    recipe_lines = recipe_text.splitlines()
    recipe_lines = [line for line in recipe_lines if line.strip() != '']
    result['title'] = re.match(TITLE_FORMAT, recipe_lines[0]).group(1)
    result['categories'] = re.match(CATEGORIES_FORMAT, recipe_lines[1]).group(1)
    result['servings'] = re.match(SERVINGS_FORMAT, recipe_lines[2]).group(1)
    assert re.match(END_OF_RECIPE, recipe_lines[-1])

    ingredient_lines = []
    direction_lines = []
    for line in recipe_lines[3:-1]:
        if re.match(INGREDIENT_FORMAT, line):
            ingredient_lines.append(line)
        else:
            direction_lines.append(line)
    # TODO : handle credits

    result['ingredient'] = "\n".join(ingredient_lines)
    result['direction_lines'] = "\n".join(direction_lines)

    return result


def parse(text):
    recipes = split_to_recipes(text)
    recipes = [recipe for recipe in recipes if recipe.strip() != ""]
    return [parse_single_recipe(recipe) for recipe in recipes]
