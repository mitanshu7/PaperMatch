import json

import requests
from bs4 import BeautifulSoup

# The page where arXiv has listed all their category information
taxonomy_url = "https://arxiv.org/category_taxonomy"

# Get the page
taxonomy_html = requests.get(taxonomy_url).content

# Create a soup
soup = BeautifulSoup(taxonomy_html, "html.parser")

# category_taxonomy_list has the full block under which the taxonomy is defined
# accordion-body class has different blocks divided by subject
subjects = soup.find(id="category_taxonomy_list").find_all(
    attrs={"class": "accordion-body"}
)

# Create empty dict to hold the taxonomy with the following conventions:
# the keys hold the subject names and the values are a list of categories
taxonomy = {}

# Iterating over different subjects
for subject in subjects:
    # Get the name of the subject stored as: id="accordion-panel-grp_cs"
    subject_id = subject.get("id")
    subject_name = subject_id.split("_")[-1]

    print("#" * 80)
    print(subject_name)

    # Create an empty list to store the tags
    taxonomy[subject_name] = []

    # Get all the category tags stored as: <h4>cs.AI <span>(Artificial Intelligence)</span></h4>
    subject_subcategories = subject.find_all("h4")

    # Iterate over all subcategories found
    for subject_subcategory in subject_subcategories:
        # Get the text of the html as: cs.AI (Artificial Intelligence)
        category = subject_subcategory.text.split(" ")[0]
        taxonomy[subject_name].append(category)

    filter_list = [
        f'categories LIKE "%{category_code}%"'
        for category_code in taxonomy[subject_name]
    ]

    filter = " OR ".join(filter_list)

    print("#" * 40)
    print(filter)


# print(taxonomy)

with open("backend/arxiv_taxonomy.json", "w") as file:
    json.dump(taxonomy, file, indent=2)
