import urllib2
import argparse
from bs4 import BeautifulSoup
import re
import os

FFTS_RECIPES_PAGE_URL = "http://www.ffts.com/recipes.htm"
LINK_TAG = 'a'
URL_ATTRIBUTE = 'href'
ZIP_LINK_PATTERN = ".*?\.zip"

def get_recipes_page():
	response = urllib2.urlopen(FFTS_RECIPES_PAGE_URL)
	return response.read()

def find_all_links(html):
	soup = BeautifulSoup(html,'html.parser')
	return [link.get(URL_ATTRIBUTE) for link in soup.find_all(LINK_TAG)]
	
def find_all_zip_links(html):
	return [link for link in find_all_links(html) if re.match(ZIP_LINK_PATTERN, link)]

def download_all_recipes(result_folder_path):
	if not os.path.exists(result_folder_path):
		os.makedirs(result_folder_path)
	
	recipes_page_html = get_recipes_page()
	zip_links = find_all_zip_links(recipes_page_html)
	print ("find %d zip files"%(len(zip_links)))
	for link in zip_links:
		response = urllib2.urlopen(link)
		zip_data = response.read()
		zip_file_path = os.path.join(result_folder_path, os.path.basename(link))
		print "download %s to %s"%(link, zip_file_path)
		open(zip_file_path, 'wb').write(zip_data)
		
def get_command_line_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("result_folder_path",
                                 help="path to folder for the recipes zip files")
    return argument_parser.parse_args()
    
def main():
	args = get_command_line_arguments()
	download_all_recipes(args.result_folder_path)

if __name__ == "__main__":
	main()
	
