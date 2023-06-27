import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def get_links(url, level=0, max_level=1):
    # Parse the base URL
    base_url = urlparse(url).scheme + "://" + urlparse(url).hostname

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the <a> tags in the HTML content
    links = soup.find_all("a")

    # Extract the href attribute from each <a> tag
    urls = []
    for link in links:
        href = link.get("href")
        if href is not None:
            # Join the URL with the base URL to get the absolute URL
            absolute_url = urljoin(base_url, href)
            # Check if the URL is internal
            if urlparse(absolute_url).hostname == urlparse(url).hostname:
                urls.append(absolute_url)

    # Recursively call the function for each URL found on the first level
    if level < max_level:
        for url in urls:
            if "folder" in url:
                urls += get_links(url, level=level+1, max_level=level+2)

    return urls


links = get_links("https://thola.freshdesk.com/support/solutions/")

# exclude all urls containing the word "folders"
links = [link for link in links if "folders" not in link]

# make links unique
links = list(set(links))

# only keep links containing the word "articles"
links = [link for link in links if "articles" in link]

# count the links
print(len(links))

# get each link and convert the HTML to text
for link in links:
    response = requests.get(link)

    # Parse the HTML content using BeautifulSoup only for div that contains fw-content
    soup = BeautifulSoup(response.content, "html.parser").find("div", {"class": "fw-content"})
    
    if soup:
        # save the text to a file named after the article id
        article_id = link.split("/")[-1]

        # remove white lines and double new lines from the text
        article = soup.get_text().replace("\n\n", "\n").strip()

        with open(f"articles/{article_id}.txt", "w") as f:
            f.write(article)




