import StringIO
import argparse
import os
import re
import urllib2
import zipfile

from bs4 import BeautifulSoup
import meal_master_parser

FFTS_RECIPES_PAGE_URL = "http://www.ffts.com/recipes.htm"
LINK_TAG = 'a'
URL_ATTRIBUTE = 'href'
ZIP_LINK_PATTERN = ".*?\.zip"
MMDF_FILE_EXTENSION = "mmf"
DEFAULT_RECIPES_FOLDER = "FFTS_RECIPES"

TRAIN_RECIPES = ["29000KZ.mmf", "18000KZ.mmf", "Ethiopia.mmf", "12000.mmf", "barley.mmf", "3000KZ.mmf", "11000.mmf",
                 "8000.mmf", "1000KZ.mmf", "33961KZ.mmf", "12000KZ.mmf", "13000.mmf", "32000.mmf", "Mm13000f.mmf",
                 "15000.mmf", "2000.mmf", "cakes02.mmf", "Mmmicwv1.mmf", "allrecip.mmf", "30000KZ.mmf", "Mmmicwv2.mmf",
                 "1000.mmf", "Mm13000b.mmf", "23000.mmf", "16000KZ.mmf", "14000.mmf", "24000KZ.mmf", "Fruits.mmf",
                 "17000.mmf", "7000KZ.mmf", "Pasta.mmf", "26000.mmf", "4000.mmf", "11000KZ.mmf", "Mm13000g.mmf",
                 "18000.mmf", "Mm13000k.mmf", "28000KZ.mmf", "22000.mmf", "Mm13000h.mmf", "28000.mmf", "21000KZ.mmf",
                 "Porkgr.mmf", "Turkey.mmf", "19000KZ.mmf", "32000KZ.mmf", "30000.mmf", "Ccakes.mmf", "8000KZ.mmf",
                 "20000.mmf", "Dips.mmf", "13000KZ.mmf", "24000.mmf", "7000.mmf", "14000KZ.mmf", "27000KZ.mmf",
                 "10000.mmf", "Soup.mmf", "6000KZ.mmf", "Wildrice.mmf", "9000.mmf", "19000.mmf", "German1.mmf",
                 "misc2600.mmf", "Mm13000c.mmf", "23000KZ.mmf", "5000.mmf", "9000KZ.mmf", "Vegan2.mmf", "29000.mmf",
                 "Swordfis.mmf", "Noodles.mmf", "biscotti.mmf", "Diab722.mmf", "31000.mmf", "16000.mmf", "Lentil.mmf",
                 "Chickenb.mmf", "0222-2.mmf", "2000KZ.mmf", "Vegan.mmf", "Soup2.mmf", "Cheese.mmf", "27000.mmf",
                 "stephen2.mmf", "Chilis.mmf", "Mmchick.mmf", "breadmaker.mmf", "33000KZ.mmf", "Mm13000e.mmf",
                 "welsh.mmf", "asparagu.mmf", "25000KZ.mmf"]
DEV_RECIPES = ["26000KZ.mmf", "Sour.mmf", "6000.mmf", "15000KZ.mmf", "Diabetic.mmf", "25000.mmf", "Mmvegy.mmf",
               "brownric.mmf", "Spaghett.mmf", "Mm13000j.mmf", "Mmice.mmf", "31000KZ.mmf", "Mmfilip.mmf", "0222-1.mmf",
               "mm2155re.mmf", "Drinks.mmf", "Porktend.mmf", "Kids.mmf"]
TEST_RECIPES = ["cheddar.mmf", "21000.mmf", "Canada.mmf", "Porkroas.mmf", "5000KZ.mmf", "17000KZ.mmf", "Greek.mmf",
                "Porkchop.mmf", "Mmcyber5.mmf", "wildgame.mmf", "Garlic.mmf", "Mushside.mmf", "22000KZ.mmf",
                "20000KZ.mmf", "10000KZ.mmf", "Mm13000a.mmf", "Usenet.mmf", "Filetmig.mmf", "Usdafood.mmf",
                "Mm13000d.mmf", "3000.mmf", "4000KZ.mmf", "apetizer.mmf", "Mmgsotw1.mmf", "Londontn.mmf"]


def get_recipes_page():
    response = urllib2.urlopen(FFTS_RECIPES_PAGE_URL)
    return response.read()


def find_all_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    return [link.get(URL_ATTRIBUTE) for link in soup.find_all(LINK_TAG)]


def find_all_zip_links(html):
    return [link for link in find_all_links(html) if re.match(ZIP_LINK_PATTERN, link)]


def download_all_recipes(result_folder_path):
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    recipes_page_html = get_recipes_page()
    zip_links = find_all_zip_links(recipes_page_html)
    for link in zip_links:
        response = urllib2.urlopen(link)
        zip_data_stream = StringIO.StringIO(response.read())
        zip_object = zipfile.ZipFile(zip_data_stream)
        for file_name in zip_object.namelist():
            if file_name.lower().endswith(MMDF_FILE_EXTENSION):
                result_path = os.path.join(result_folder_path, file_name)
                open(result_path, 'wb').write(zip_object.read(file_name))


def get_command_line_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--result_folder_path",
                                 help="path to folder for the recipes zip files",
                                 default=DEFAULT_RECIPES_FOLDER)
    return argument_parser.parse_args()


def create_recipes_batch(recipes_folder, files_list):
    result = {}
    for recipe_file_name in files_list:
        recipe_file_data = open(os.path.join(recipes_folder, recipe_file_name), "r").read()
        result[recipe_file_name] = meal_master_parser.parse(recipe_file_data)

    return result


def get_all_recipes(recipes_folder=DEFAULT_RECIPES_FOLDER):
    if not os.path.isdir(recipes_folder):
        download_all_recipes(recipes_folder)

    train = create_recipes_batch(recipes_folder, TRAIN_RECIPES)
    dev = create_recipes_batch(recipes_folder, DEV_RECIPES)
    test = create_recipes_batch(recipes_folder, TEST_RECIPES)

    return train, dev, test


def get_recipe_by_name(name, recipes_folder=DEFAULT_RECIPES_FOLDER):
    if not os.path.isdir(recipes_folder):
        download_all_recipes(recipes_folder)
    return create_recipes_batch(recipes_folder, [name])


def main():
    args = get_command_line_arguments()
    download_all_recipes(args.result_folder_path)


if __name__ == "__main__":
    main()
