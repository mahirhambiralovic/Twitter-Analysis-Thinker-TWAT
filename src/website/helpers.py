import urllib,json
from random import randint

def get_gif_url(search_term):
    data = json.loads(urllib.request.urlopen("http://api.giphy.com/v1/gifs/search?q="+search_term+"&api_key=LYw5jwHW9fkeOihU4xJCxJQEt31DhC5u&limit=10").read())
    return data['data'][randint(0, 9)]['images']['fixed_height']['url']
