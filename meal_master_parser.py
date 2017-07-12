import re

# according http://www.ffts.com/mmformat.txt

RECIPE_HEADER_FORMAT = "-----.*Meal-Master.*?\n"
TITLE_FORMAT = "\s*Title: .*"
CATEGORIES_FORMAT = "\s*Categories: .*"
SERVINGS_FORMAT = ".*?\d+.*"
INGREDIENT_FORMAT = "[ \d\./]{7} [a-zA-Z ]{2}"
END_OF_RECIPE = "-----[ -]*"


def split_to_recipes(text):
    return re.split(RECIPE_HEADER_FORMAT, text)


def extract_direction_lines_from_recipes(recipe_text):
    recipe_lines = recipe_text.splitlines()
    recipe_lines = [line for line in recipe_lines if line.strip() != '']
    assert re.match(TITLE_FORMAT, recipe_lines[0])
    assert re.match(CATEGORIES_FORMAT, recipe_lines[1])
    assert re.match(SERVINGS_FORMAT, recipe_lines[2])
    assert re.match(END_OF_RECIPE, recipe_lines[-1])

    direction_lines = [line for line in recipe_lines[3:-1] if not re.match(INGREDIENT_FORMAT, line)]
    # TODO : handle credits
    return "\n".join(direction_lines)


def parse(text):
    recipes = split_to_recipes(text)
    recipes = [recipe for recipe in recipes if recipe.strip() != ""]
    return [extract_direction_lines_from_recipes(recipe) for recipe in recipes]
